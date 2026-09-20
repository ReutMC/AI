#!/usr/bin/env python3
"""ARION ALPHA 1 — interactive chat in Persian or English.

Usage:
  python inference/chat.py
  python inference/chat.py --model model/arion-alpha-1-merged --max-new-tokens 400
Commands inside chat: /exit  /clear  /lang fa|en  /temp <float>
The model automatically answers in the language you write in.
"""
import argparse, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.environ.get("ARION_MODEL", os.path.join(ROOT, "model", "arion-alpha-1-merged"))

SYSTEM = ("You are Arion Alpha 1, a professional bilingual (Persian/English) web development AI "
          "specialized in HTML, CSS, JavaScript and modern frontend development. Answer in the "
          "same language as the user, with complete working code and clear explanations.")

WELCOME = """
================ ARION ALPHA 1 ================
Bilingual Web-Development AI (Persian / English)
Type your question in Persian or English.
Commands: /exit  /clear  /lang fa|en  /temp <0.0-1.5>
==============================================
"""

def main():
    ap = argparse.ArgumentParser(description="Chat with ARION ALPHA 1")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--no-stream", action="store_true")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    torch.set_num_threads(max(1, os.cpu_count() - 1))

    history = []
    temperature = args.temperature

    print(WELCOME)
    while True:
        try:
            user = input("You ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye 👋")
            break
        if not user:
            continue
        if user == "/exit":
            print("bye 👋")
            break
        if user == "/clear":
            history = []
            print("history cleared.")
            continue
        if user.startswith("/lang"):
            print("Arion Alpha 1 follows your language automatically — no switch needed.")
            continue
        if user.startswith("/temp"):
            try:
                temperature = float(user.split()[1])
                print(f"temperature = {temperature}")
            except Exception:
                print("usage: /temp 0.7")
            continue

        messages = [{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": user}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        inputs = tok(prompt, return_tensors="pt")

        import time as _t
        t0 = _t.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=temperature > 0.05,
                temperature=max(0.05, temperature),
                top_p=args.top_p,
                repetition_penalty=1.05,
                pad_token_id=tok.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(gen, skip_special_tokens=True).strip()
        dt = _t.time() - t0
        ntok = gen.shape[0]
        print(f"\nArion ▸ {text}\n\n[generated {ntok} tokens in {dt:.1f}s ≈ {ntok/max(0.1, dt):.1f} tok/s]\n")
        history += [{"role": "user", "content": user}, {"role": "assistant", "content": text}]
        if len(history) > 12:
            history = history[-12:]

if __name__ == "__main__":
    main()
