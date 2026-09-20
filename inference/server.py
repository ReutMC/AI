#!/usr/bin/env python3
"""ARION ALPHA 1 — on-demand local inference server (stdlib only).

Started from the dashboard when (and only when) free RAM allows.
Endpoints:
  GET  /health -> {"ready": bool, "model": str}
  POST /ask    -> {"ok": true, "text": str, "tokens": int, "seconds": float}
                  with body {"prompt", "history", "stream": true} it responds with
                  newline-delimited JSON: {"t": piece}... then {"done": true, tokens, seconds}
Body: {"prompt": str, "history": [{"role","content"}, ...]}
"""
import argparse, json, os, re, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty

os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

SYSTEM = ("You are Arion Alpha 1, a professional bilingual (Persian/English) web development AI "
          "specialized in HTML, CSS, JavaScript and modern frontend development. Answer in the "
          "same language as the user, with complete working code and clear explanations.")

STATE = {"model": None, "tok": None, "ready": False, "name": "", "busy": False}
MAX_NEW = 240


def load_model(model_dir):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(2)
    print(f"[demo] loading {model_dir} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    model.eval()
    STATE["model"] = model
    STATE["tok"] = tok
    STATE["name"] = model_dir
    STATE["ready"] = True
    print("[demo] ready", flush=True)


def build_inputs(prompt, history):
    tok = STATE["tok"]
    msgs = [{"role": "system", "content": SYSTEM}]
    for h in history[-6:]:
        if h.get("role") in ("user", "assistant") and h.get("content"):
            msgs.append({"role": h["role"], "content": str(h["content"])[:1200]})
    msgs.append({"role": "user", "content": prompt[:1200]})
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(text, return_tensors="pt")


def generate(prompt, history):
    import torch
    tok, model = STATE["tok"], STATE["model"]
    ids = build_inputs(prompt, history)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=True,
                             temperature=0.7, top_p=0.9, repetition_penalty=1.05,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][ids["input_ids"].shape[1]:]
    answer = tok.decode(gen, skip_special_tokens=True).strip()
    dt = time.time() - t0
    return answer, int(gen.shape[0]), round(dt, 1)


def generate_stream(prompt, history, emit):
    """Token-streamed generation. emit(dict) is called per piece and once with done=true."""
    import torch
    from transformers import TextIteratorStreamer
    tok, model = STATE["tok"], STATE["model"]
    ids = build_inputs(prompt, history)
    # timeout guards against a worker crash never calling end() (blocks forever otherwise)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True,
                                    timeout=120)
    kwargs = dict(max_new_tokens=MAX_NEW, do_sample=True, temperature=0.7,
                  top_p=0.9, repetition_penalty=1.05, pad_token_id=tok.eos_token_id,
                  streamer=streamer)
    t0 = time.time()
    worker = threading.Thread(target=lambda: model.generate(**ids, **kwargs), daemon=True)
    worker.start()
    pieces = []
    aborted = False
    try:
        for piece in streamer:
            if not piece:
                continue
            pieces.append(piece)
            try:
                emit({"t": piece})
            except (BrokenPipeError, ConnectionResetError):
                aborted = True  # client went away — keep generating, stop writing
    except Empty:
        pass  # worker died without ending the streamer
    worker.join()
    dt = time.time() - t0
    answer = "".join(pieces).strip()
    ntok = len(tok(answer, add_special_tokens=False)["input_ids"]) if answer else 0
    if not aborted:
        try:
            emit({"done": True, "tokens": ntok, "seconds": round(dt, 1)})
        except (BrokenPipeError, ConnectionResetError):
            pass
    return answer, ntok, round(dt, 1)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._send(200, {"ready": STATE["ready"], "model": "arion-alpha-1-merged"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.startswith("/ask"):
            return self._send(404, {"error": "not found"})
        if not STATE["ready"]:
            return self._send(503, {"error": "model loading"})
        if STATE["busy"]:
            return self._send(429, {"error": "busy — one request at a time on this CPU"})
        try:
            length = int(self.headers.get("content-length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            prompt = str(body.get("prompt", "")).strip()
            if not prompt:
                return self._send(400, {"error": "empty prompt"})
            STATE["busy"] = True
            try:
                if body.get("stream"):
                    self._stream_ask(prompt, body.get("history", []))
                else:
                    answer, ntok, secs = generate(prompt, body.get("history", []))
                    self._send(200, {"ok": True, "text": answer, "tokens": ntok, "seconds": secs})
            finally:
                STATE["busy"] = False
        except (BrokenPipeError, ConnectionResetError):
            STATE["busy"] = False
        except Exception as e:
            STATE["busy"] = False
            try:
                self._send(500, {"error": str(e)[:300]})
            except Exception:
                pass

    def _stream_ask(self, prompt, history):
        # no content-length: tell the client the body ends at connection close
        self.close_connection = True
        self.send_response(200)
        self.send_header("content-type", "application/x-ndjson; charset=utf-8")
        self.send_header("cache-control", "no-cache")
        self.send_header("connection", "close")
        self.end_headers()

        def emit(obj):
            data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
            self.wfile.write(data)
            self.wfile.flush()

        generate_stream(prompt, history, emit)

    def log_message(self, fmt, *args):
        sys.stderr.write("[demo-http] " + (fmt % args) + "\n")

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="arion_demo_server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=3099)
    args = ap.parse_args()
    import threading
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=load_model, args=(args.model,), daemon=True).start()
    print(f"[demo] listening on {args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
