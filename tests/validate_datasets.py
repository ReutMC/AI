#!/usr/bin/env python3
"""ARION ALPHA 1 — dataset validation utility (tests/)."""
import json, glob, sys

ROOT = "/home/z/my-project/arion-alpha-1"

def main():
    ok = True
    for path in sorted(glob.glob(f"{ROOT}/datasets/*/*.jsonl")):
        n, bad = 0, 0
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            try:
                ex = json.loads(line)
                assert "messages" in ex and isinstance(ex["messages"], list)
                for m in ex["messages"]:
                    assert m.get("role") in ("system", "user", "assistant")
                    assert isinstance(m.get("content"), str) and len(m["content"]) > 0
                n += 1
            except Exception:
                bad += 1
        status = "OK " if bad == 0 else "BAD"
        if bad: ok = False
        print(f"[{status}] {path}: {n} valid, {bad} invalid")
    # splits exist?
    for split in ("train", "val", "test"):
        import os
        p = f"{ROOT}/datasets/processed/{split}.jsonl"
        if not os.path.exists(p):
            print(f"[MISSING] {p}"); ok = False
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
