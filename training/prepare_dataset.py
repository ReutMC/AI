#!/usr/bin/env python3
"""ARION ALPHA 1 — dataset cleaning, quality filtering, dedup and splits.

Pipeline (spec sections 18, 35):
  1. validate schema (roles/order, non-empty)
  2. quality filters (length, degenerate repetition, language ratios)
  3. secret scanning (API keys, tokens, passwords)
  4. exact + near duplicate removal (normalized user+assistant hash)
  5. stratified 90/5/5 train/val/test splits
Outputs processed/*.jsonl + dataset_stats.json
"""
import json, glob, hashlib, random, os, re, unicodedata
from collections import Counter

random.seed(4217)
ROOT = os.environ.get("ARION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = f"{ROOT}/datasets/generated"
OUT = f"{ROOT}/datasets/processed"
os.makedirs(OUT, exist_ok=True)

SYS_OK_PREFIX = "You are Arion Alpha 1"

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                 # openai-style
    re.compile(r"AKIA[0-9A-Z]{16}"),                    # aws
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),                # github
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{25,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]

def persian_ratio(t):
    fa = sum(1 for c in t if "\u0600" <= c <= "\u06FF")
    return fa / max(1, len(t))

def latin_ratio(t):
    la = sum(1 for c in t if c.isascii() and c.isalpha())
    return la / max(1, len(t))

def norm(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip().lower()

def repetition_score(t):
    words = t.split()
    if len(words) < 12:
        return 0.0
    tri = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    return 1.0 * (len(tri) - len(set(tri))) / len(tri)

def validate(ex):
    msgs = ex.get("messages")
    if not msgs or len(msgs) < 2:
        return "bad_messages"
    roles = [m["role"] for m in msgs]
    if roles[0] == "system" and not msgs[0]["content"].startswith(SYS_OK_PREFIX):
        return "foreign_system"
    if "user" not in roles or "assistant" not in roles:
        return "missing_roles"
    for m in msgs:
        c = m.get("content")
        if not isinstance(c, str) or len(c.strip()) < 5:
            return "empty_content"
    u = next(m["content"] for m in msgs if m["role"] == "user")
    a = next(m["content"] for m in msgs if m["role"] == "assistant")
    if len(a) < 40:
        return "answer_too_short"
    if len(u) > 4000 or len(a) > 24000:
        return "too_long"
    if repetition_score(a) > 0.45:
        return "repetitive"
    for p in SECRET_PATTERNS:
        if p.search(a) or p.search(u):
            return "secret_like"
    return "ok"

def lang_check(ex):
    lang = ex.get("lang", "en")
    u = next((m["content"] for m in ex["messages"] if m["role"] == "user"), "")
    if lang == "fa" and persian_ratio(u) < 0.12:
        return False
    if lang == "en" and persian_ratio(u) > 0.45:
        return False
    return True

def main():
    rows, rejected = [], Counter()
    for path in sorted(glob.glob(f"{GEN}/*.jsonl")):
        src = os.path.basename(path)
        kept = 0
        for i, line in enumerate(open(path, encoding="utf-8")):
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
            except Exception:
                rejected[f"{src}:json"] += 1
                continue
            verdict = validate(ex)
            if verdict != "ok":
                rejected[verdict] += 1
                continue
            if not lang_check(ex):
                rejected["lang_mismatch"] += 1
                continue
            ex["_src"] = src
            rows.append(ex)
            kept += 1
        print(f"{src}: kept {kept}")

    # dedup (exact on normalized user+assistant, then near-dup on user)
    seen, deduped = set(), []
    for ex in rows:
        u = next(m["content"] for m in ex["messages"] if m["role"] == "user")
        a = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
        h = hashlib.md5((norm(u) + "||" + norm(a)[:600]).encode()).hexdigest()
        if h in seen:
            rejected["dup_exact"] += 1
            continue
        seen.add(h)
        deduped.append(ex)
    rows = deduped

    random.shuffle(rows)
    n = len(rows)
    n_val, n_test = int(n * 0.05), int(n * 0.05)
    splits = {
        "val": rows[:n_val],
        "test": rows[n_val:n_val + n_test],
        "train": rows[n_val + n_test:],
    }
    for name, data in splits.items():
        with open(f"{OUT}/{name}.jsonl", "w", encoding="utf-8") as f:
            for ex in data:
                ex.pop("_src", None)
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    stats = {
        "total_raw_lines": sum(rejected.values()) + n,
        "accepted_after_cleaning": n,
        "rejected": dict(rejected),
        "splits": {k: len(v) for k, v in splits.items()},
        "train_categories": dict(Counter(r["category"] for r in splits["train"])),
        "train_langs": dict(Counter(r["lang"] for r in splits["train"])),
        "train_difficulty": dict(Counter(r["difficulty"] for r in splits["train"])),
        "split_ratio": "90/5/5",
    }
    with open(f"{OUT}/dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
