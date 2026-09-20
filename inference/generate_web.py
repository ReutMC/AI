#!/usr/bin/env python3
"""ARION ALPHA 1 — generate a complete web project from a natural-language prompt.

The model is asked for a multi-file project; the answer's fenced code blocks
are parsed into a real folder on disk.

Usage:
  python inference/generate_web.py "یک landing page برای کافه بساز" --out ./output/my-cafe
  python inference/generate_web.py "Build a responsive portfolio" --out ./output/portfolio
"""
import argparse, json, os, re, sys, datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.environ.get("ARION_MODEL", os.path.join(ROOT, "model", "arion-alpha-1-merged"))

SYS = ("You are Arion Alpha 1, a professional bilingual (Persian/English) web development AI "
       "specialized in HTML, CSS, JavaScript and modern frontend development. Answer in the "
       "same language as the user, with complete working code and clear explanations.")

PROJECT_INSTRUCTION = """
Generate a COMPLETE multi-file static web project for the request above.
Output format (mandatory):
1) A ```text fenced block with the file tree.
2) Then one fenced code block per file, each immediately preceded by a bold path header like **index.html** or **css/style.css**.
Rules: production-quality, responsive, semantic HTML, modern CSS (custom properties, flex/grid, clamp), vanilla JS with defer, correct relative paths between files, Google Fonts via preconnect when fonts are used, no TODOs, no placeholders. If the user writes in Persian use lang="fa" dir="rtl" with the Vazirmatn font.
"""

FENCE_RE = re.compile(r"```([a-zA-Z0-9]*)\s*\n([\s\S]*?)```", re.M)
PATH_RE = re.compile(r"\*\*\s*([\w./\-]+)\*\*")

LANG_EXT = {"html": "index.html", "css": "styles.css", "js": "app.js", "javascript": "app.js"}


def extract_files(answer):
    """Return list of (path, code). Uses bold path headers when present."""
    files, pos = [], 0
    spans = list(FENCE_RE.finditer(answer))
    for i, m in enumerate(spans):
        lang, code = m.group(1).lower(), m.group(2).strip()
        if lang == "text":
            continue
        # path header = bold token between previous fence end and this fence start
        header_zone = answer[pos:m.start()]
        pm = PATH_RE.findall(header_zone)
        if pm:
            path = pm[-1]
        else:
            path = LANG_EXT.get(lang, f"file{i}.{lang or 'txt'}")
        files.append((path, code))
        pos = m.end()
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", help="natural-language website description (Persian or English)")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=1600)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    outdir = args.out or os.path.join(
        ROOT, "output", datetime.datetime.now().strftime("web-%Y%m%d-%H%M%S"))
    os.makedirs(outdir, exist_ok=True)

    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    model.eval()
    torch.set_num_threads(max(1, os.cpu_count() - 1))

    messages = [{"role": "system", "content": SYS},
                {"role": "user", "content": args.prompt + "\n" + PROJECT_INSTRUCTION}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                             do_sample=True, temperature=0.6, top_p=0.9,
                             repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    raw_answer_path = os.path.join(outdir, "_model_answer.md")
    open(raw_answer_path, "w", encoding="utf-8").write(answer)

    files = extract_files(answer)
    if not files:
        print("No code blocks detected — raw answer saved to", raw_answer_path)
        sys.exit(2)
    for path, code in files:
        path = path.replace("\\", "/").lstrip("/")
        if ".." in path:
            continue  # safety
        full = os.path.join(outdir, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        open(full, "w", encoding="utf-8").write(code + "\n")
        print(f"  wrote {path} ({len(code)} chars)")
    json.dump({"prompt": args.prompt, "files": [p for p, _ in files],
               "generated_at": datetime.datetime.now().isoformat()},
              open(os.path.join(outdir, "_project.json"), "w"), indent=2)
    print(f"\nProject ready → {outdir}")
    print("Open index.html in a browser to view it.")


if __name__ == "__main__":
    main()
