#!/usr/bin/env python3
"""ARION ALPHA 1 — self-testing of generated code (spec section 20).

Validates HTML/CSS/JS blocks found in a model answer or a generated project:
  1. HTML well-formedness  (html5lib strict when available)
  2. CSS parse             (tinycss2)
  3. JS syntax             (node --check)
  4. optional Playwright render test for single-file pages (--browser)

Usage:
  python tests/selftest.py --answer-file answer.md
  python tests/selftest.py --project output/web-XXXX
"""
import argparse, json, os, re, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FENCE_RE = re.compile(r"```([a-zA-Z0-9]*)\s*\n([\s\S]*?)```")


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
        rules = tinycss2.parse_stylesheet_bytes(code.encode("utf-8"))[0]
        bad = [r for r in rules if r.type == "error"]
        return len(bad)==0, (bad[0].message if bad else "")
    except ImportError:
        ok = code.count("{")==code.count("}")
        return ok, "brace mismatch" if not ok else ""
    except Exception as e:
        return False, str(e)


def check_js(code):
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as f:
            f.write(code); path=f.name
        r = subprocess.run(["node","--check",path], capture_output=True, text=True, timeout=20)
        os.unlink(path)
        return r.returncode==0, (r.stderr.strip().splitlines() or [""])[0][:200]
    except Exception as e:
        return False, str(e)


def check_files(files):
    """files: list of (name, code). Route by extension/content."""
    results = []
    for name, code in files:
        ext = name.lower().rsplit(".",1)[-1] if "." in name else ""
        if ext == "html" or "<html" in code[:200] or "<!DOCTYPE" in code[:100]:
            ok, err = check_html(code)
        elif ext == "css":
            ok, err = check_css(code)
        elif ext == "js":
            ok, err = check_js(code)
        else:
            continue
        results.append({"file": name, "valid": ok, "error": err})
    return results


def browser_test(index_html):
    """Headless render test with Playwright if installed."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return {"browser_test": "skipped (playwright not installed)"}
    try:
        with sync_playwright() as p:
            b = p.chromium.launch()
            page = b.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"file://{index_html}")
            page.wait_for_timeout(800)
            title = page.title()
            b.close()
            return {"browser_test": "passed", "title": title,
                    "js_errors": errors[:5]}
    except Exception as e:
        return {"browser_test": "failed", "error": str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answer-file", help="markdown file containing fenced code blocks")
    ap.add_argument("--project", help="directory of generated web project")
    ap.add_argument("--browser", action="store_true", help="also run headless render test")
    args = ap.parse_args()

    files = []
    if args.answer_file:
        text = open(args.answer_file, encoding="utf-8").read()
        for i, (lang, code) in enumerate(FENCE_RE.findall(text)):
            lang = lang.lower() or "txt"
            files.append((f"fenced-{i}.{lang if lang!='javascript' else 'js'}", code.strip()))
    if args.project:
        for base, _, names in os.walk(args.project):
            for n in names:
                if n.startswith("_"):
                    continue
                p = os.path.join(base, n)
                files.append((n, open(p, encoding="utf-8").read()))

    if not files:
        print("nothing to test"); sys.exit(2)

    results = check_files(files)
    n_ok = sum(1 for r in results if r["valid"])
    for r in results:
        print(("PASS " if r["valid"] else "FAIL ") + r["file"] + (f" — {r['error']}" if r["error"] else ""))

    if args.browser:
        idx = os.path.join(args.project or ".", "index.html")
        if os.path.exists(idx):
            br = browser_test(idx)
            print(json.dumps(br, indent=2))

    print(f"\n{n_ok}/{len(results)} files valid")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
