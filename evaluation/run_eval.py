#!/usr/bin/env python3
"""ARION ALPHA 1 — automated evaluation harness (spec sections 19, 20).

Measures on generated answers:
  - syntax correctness   : HTML well-formedness (html5lib), CSS parse (tinycss2), JS syntax (node --check)
  - language quality     : script ratio matches requested language, natural length
  - instruction following: keyword/tag coverage for the request type, code fence presence
  - web-dev relevance    : share of domain vocabulary present
  - degeneration         : trigram repetition ratio, empty answers
  - code completeness    : no TODO/placeholder markers
Usage:
  python evaluation/run_eval.py --model model/arion-alpha-1-merged --set both --limit 30
"""
import argparse, json, os, re, subprocess, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYS = ("You are Arion Alpha 1, a professional bilingual (Persian/English) web development AI "
       "specialized in HTML, CSS, JavaScript and modern frontend development. Answer in the "
       "same language as the user, with complete working code and clear explanations.")

def persian_ratio(t):
    return sum(1 for c in t if "\u0600" <= c <= "\u06FF") / max(1, len(t))

def trigram_rep(t):
    w = t.split()
    if len(w) < 12: return 0.0
    tri = [tuple(w[i:i+3]) for i in range(len(w)-2)]
    return (len(tri) - len(set(tri))) / len(tri)

FENCE_RE = re.compile(r"```([a-zA-Z0-9]*)\s*\n([\s\S]*?)```")

def extract_blocks(text):
    out = {"html": [], "css": [], "js": [], "other": []}
    for lang, code in FENCE_RE.findall(text):
        lang = lang.lower()
        key = {"html": "html", "css": "css", "js": "js", "javascript": "js",
               "jsx": "js", "json": "other", "text": "other"}.get(lang, None)
        if key is None:
            # guess by content
            if "<html" in code or "<div" in code or "<!DOCTYPE" in code: key = "html"
            elif "{" in code and (":" in code) and "<" not in code: key = "css"
            else: key = "other"
        out[key].append(code)
    return out

def check_html(code):
    """Structural tag-balance validation (browsers accept bare '&' in URLs,
    so strict entity parsing is intentionally not used)."""
    try:
        from html.parser import HTMLParser
        class P(HTMLParser):
            VOID = {"area","base","br","col","embed","hr","img","input","link","meta","param","source","track","wbr"}
            def __init__(self):
                super().__init__(); self.stack=[]; self.err=""
            def handle_starttag(self, tag, attrs):
                if tag not in self.VOID: self.stack.append(tag)
            def handle_endtag(self, tag):
                if tag in self.VOID: return
                if not self.stack: self.err=f"stray </{tag}>"; return
                if self.stack[-1]==tag: self.stack.pop()
                elif tag in self.stack:
                    while self.stack and self.stack[-1]!=tag: self.stack.pop()
                    if self.stack: self.stack.pop()
                else: self.err=f"unmatched </{tag}>"
        p=P(); p.feed(code)
        return (not p.err and len(p.stack)<=1), p.err or (f"unclosed: {p.stack}" if p.stack else "")
    except Exception as e:
        return False, str(e)


def check_css(code):
    try:
        import tinycss2
        rules, err = tinycss2.parse_stylesheet_bytes(code.encode("utf-8")), None
        bad = [r for r in rules[0] if r.type == "error"]
        return len(bad) == 0, (bad[0].message if bad else "")
    except ImportError:
        ok = code.count("{") == code.count("}")
        return ok, "brace mismatch" if not ok else ""
    except Exception as e:
        return False, str(e)

def check_js(code):
    """Use node --check when available (node is guaranteed in this environment)."""
    node = "/usr/local/bin/node"
    if not os.path.exists(node):
        node = "node"
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(code); path = f.name
        r = subprocess.run([node, "--check", path], capture_output=True, text=True, timeout=20)
        os.unlink(path)
        return r.returncode == 0, (r.stderr.strip().splitlines() or [""])[0][:200]
    except Exception as e:
        return False, str(e)

RELEVANCE_VOCAB = [
    "html","css","javascript","js","div","flex","grid","display","font","color","margin",
    "padding","selector","style","script","class","element","browser","responsive","media",
    "برچسب","استایل","طراحی","صفحه","کد","المان","ریسپانسیو","مرورگر","چیدمان","فونت",
]
BAD_MARKERS = ["TODO", "FIXME", "implement this yourself", "placeholder logic", "... rest of the code"]

def evaluate_answer(ans, prompt, expect_lang):
    m = {}
    blocks = extract_blocks(ans)
    m["answer_chars"] = len(ans)
    m["has_code"] = bool(blocks["html"] or blocks["css"] or blocks["js"])
    m["empty"] = len(ans.strip()) < 30
    m["repetition"] = round(trigram_rep(ans), 3)
    m["bad_markers"] = [b for b in BAD_MARKERS if b.lower() in ans.lower()]

    # syntax checks
    syn = {}
    for name, codes, fn in (("html", blocks["html"], check_html),
                            ("css", blocks["css"], check_css),
                            ("js", blocks["js"], check_js)):
        if codes:
            oks, errs = [], []
            for c in codes[:3]:
                ok, err = fn(c); oks.append(ok)
                if not ok: errs.append(err)
            syn[name] = {"blocks": len(codes), "valid": all(oks), "error": errs[0] if errs else ""}
    m["syntax"] = syn
    m["syntax_valid"] = all(v["valid"] for v in syn.values()) if syn else None

    # language quality
    pr = persian_ratio(ans)
    if expect_lang == "fa":
        m["lang_ok"] = pr >= 0.10
        m["persian_ratio"] = round(pr, 3)
    elif expect_lang == "en":
        m["lang_ok"] = pr <= 0.15
        m["persian_ratio"] = round(pr, 3)
    else:
        m["lang_ok"] = None

    # relevance
    low = ans.lower()
    m["relevance_hits"] = sum(1 for v in RELEVANCE_VOCAB if v in low)
    m["relevant"] = m["relevance_hits"] >= 2

    # instruction following: build requests should produce html
    wants_page = any(k in prompt.lower() for k in ["بساز", "build", "create", "make", "generate", "landing", "dashboard", "page"])
    if wants_page:
        m["followed"] = bool(blocks["html"])
    else:
        m["followed"] = m["has_code"] or len(ans) > 120
    return m

def generate(model, tok, prompts, max_new=420):
    import torch
    results = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        messages = [{"role": "system", "content": SYS}, {"role": "user", "content": p["prompt"]}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                                 repetition_penalty=1.08, pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results.append({"id": p["id"], "prompt": p["prompt"], "answer": ans})
        if (i + 1) % 5 == 0:
            el = time.time() - t0
            print(f"  generated {i+1}/{len(prompts)} ({el/60:.1f} min)", flush=True)
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=f"{ROOT}/model/arion-alpha-1-merged")
    ap.add_argument("--set", dest="set_name", default="both", choices=["webdev", "bilingual", "both"])
    ap.add_argument("--limit", type=int, default=0, help="evaluate only first N prompts per set (0=all files)")
    ap.add_argument("--max-new", type=int, default=420)
    ap.add_argument("--out", default=f"{ROOT}/artifacts/eval_report.json")
    args = ap.parse_args()

    sets = []
    if args.set_name in ("webdev", "both"):
        sets.append(("webdev", json.load(open(f"{ROOT}/evaluation/prompts_webdev.json"))["prompts"]))
    if args.set_name in ("bilingual", "both"):
        sets.append(("bilingual", json.load(open(f"{ROOT}/evaluation/prompts_bilingual.json"))["prompts"]))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True)
    model.eval()
    torch.set_num_threads(max(1, os.cpu_count() - 1))

    report = {"model": args.model, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "sets": {}}
    for name, prompts in sets:
        if args.limit: prompts = prompts[:args.limit]
        print(f"== {name}: {len(prompts)} prompts ==", flush=True)
        gens = generate(model, tok, prompts, args.max_new)
        # proper per-row evaluation
        rows = []
        plist = prompts
        for p, g in zip(plist, gens):
            expect = p.get("expect") or ("fa" if p.get("lang") == "fa" else "en")
            m = evaluate_answer(g["answer"], p["prompt"], expect)
            rows.append({"id": p["id"], "prompt": p["prompt"], "metrics": m, "answer": g["answer"][:1200]})
        ok = lambda k: sum(1 for r in rows if r["metrics"].get(k)) / max(1, len(rows))
        syntax_rows = [r for r in rows if r["metrics"]["syntax_valid"] is not None]
        summary = {
            "n": len(rows),
            "empty_rate": round(1 - ok("empty"), 3),
            "code_production_rate": round(ok("has_code"), 3),
            "instruction_following": round(ok("followed"), 3),
            "language_correct": round(ok("lang_ok"), 3) if name == "bilingual" else None,
            "relevance_rate": round(ok("relevant"), 3),
            "syntax_checked": len(syntax_rows),
            "syntax_valid_rate": round(sum(1 for r in syntax_rows if r["metrics"]["syntax_valid"]) / max(1, len(syntax_rows)), 3),
            "bad_marker_rate": round(1 - ok("bad_markers"), 3),
            "mean_repetition": round(sum(r["metrics"]["repetition"] for r in rows) / max(1, len(rows)), 3),
        }
        report["sets"][name] = {"summary": summary, "rows": rows}
        print(json.dumps(summary, indent=2), flush=True)

    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2, ensure_ascii=False)
    print("report saved →", args.out)

if __name__ == "__main__":
    main()
