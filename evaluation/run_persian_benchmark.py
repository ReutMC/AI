#!/usr/bin/env python3
"""ARION ALPHA 1 — Persian benchmark runner.

Runs evaluation/persian_fluency_benchmark.jsonl against either:
  - a HF transformers model (--model path), or
  - a llama.cpp server (--server http://127.0.0.1:8080)

Writes responses JSONL: {"id", "category", "prompt", "response", "latency_s"}.

Usage:
  python3 run_persian_benchmark.py --model /path/to/merged --benchmark ... --out responses.jsonl
  python3 run_persian_benchmark.py --server http://127.0.0.1:8080 --benchmark ... --out responses.jsonl
"""
import argparse, json, os, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def chat_prompt_server(messages, temperature=0.0):
    return {"messages": messages, "temperature": temperature, "max_tokens": 256, "stream": False}


def run_server(server, benchmark, out_path, max_new=256):
    results = []
    for b in benchmark:
        msgs = list(b.get("history") or []) + [{"role": "user", "content": b["prompt"]}]
        body = json.dumps(chat_prompt_server(msgs)).encode("utf-8")
        req = urllib.request.Request(f"{server}/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        resp_txt = ""
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read())
                resp_txt = (data["choices"][0]["message"].get("content") or "").strip()
        except Exception as e:
            print(f"[ERR] {b['id']}: {str(e)[:120]}", flush=True)
        results.append({"id": b["id"], "category": b["category"], "prompt": b["prompt"],
                        "response": resp_txt, "latency_s": round(time.time() - t0, 2)})
        if len(results) % 20 == 0:
            print(f"  [{len(results)}/{len(benchmark)}] last: {resp_txt[:60]!r}", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return results


def run_transformers(model_path, benchmark, out_path, max_new=256, batch_size=8, sys_prompt=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
                                                 device_map="cuda")
    model.eval()
    results = []
    t_all = time.time()
    for i, b in enumerate(benchmark):
        msgs = list(b.get("history") or []) + [{"role": "user", "content": b["prompt"]}]
        if sys_prompt:
            msgs = [{"role": "system", "content": sys_prompt}] + msgs
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to("cuda")
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = out[0][ids["input_ids"].shape[1]:]
            resp_txt = tok.decode(gen, skip_special_tokens=True).strip()
            lat = time.time() - t0
        except Exception as e:
            print(f"[ERR] {b['id']}: {str(e)[:120]}", flush=True)
            resp_txt, lat = "", 0.0
        results.append({"id": b["id"], "category": b["category"], "prompt": b["prompt"],
                        "response": resp_txt, "latency_s": round(lat, 2)})
        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(benchmark)}] {round((time.time() - t_all) / 60, 1)} min, "
                  f"last: {resp_txt[:60]!r}", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--server", default=None)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--system", default=None)
    args = ap.parse_args()

    benchmark = [json.loads(l) for l in open(args.benchmark, encoding="utf-8") if l.strip()]
    print(f"benchmark: {len(benchmark)} prompts", flush=True)
    if args.server:
        run_server(args.server, benchmark, args.out, args.max_new)
    else:
        run_transformers(args.model, benchmark, args.out, args.max_new, sys_prompt=args.system)
    print(f"[done] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
