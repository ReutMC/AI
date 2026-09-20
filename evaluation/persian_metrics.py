#!/usr/bin/env python3
"""ARION ALPHA 1 — Persian fluency metrics.

Input:  benchmark responses JSONL: {"id", "category", "prompt", "response"}
        (produced by the model runner — see run_persian_benchmark)
Output: metrics JSON + markdown report; misclassified examples dumped.

Metrics (spec "AUTOMATED METRICS"):
  1. persian_response_rate        — share of responses classified fa (fa-expect items)
  2. language_compliance          — response language == expected language
  3. arabic_leakage_rate          — fa-expected items whose response classified ar
  4. empty_response_rate
  5. repetition_rate              — trigram repetition > 0.45
  6. malformed_response_rate      — unclosed code fence / placeholder artifacts
  7. average_response_length      — chars + est tokens
  8/9. validation loss / training loss — read from train_summary.json if present

Usage:
  python3 evaluation/persian_metrics.py --responses out/responses.jsonl \
      --benchmark evaluation/persian_fluency_benchmark.jsonl \
      --out out/persian_metrics.json [--label ARION]
"""
import argparse, json, os, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from persian_langid import classify  # noqa: E402

PLACEHOLDER_RE = re.compile(r"\{\{\s*\w+\s*\}\}|<نام>|\[نام\]|Lorem ipsum|\bTODO\b")


def trigram_rep(t):
    w = t.split()
    if len(w) < 12:
        return 0.0
    tri = [tuple(w[i:i + 3]) for i in range(len(w) - 2)]
    return (len(tri) - len(set(tri))) / len(tri)


def malformed(t):
    if t.count("```") % 2 == 1:
        return "unclosed_fence"
    if PLACEHOLDER_RE.search(t):
        return "placeholder"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="model")
    ap.add_argument("--train-summary", default=None,
                    help="optional train_summary.json to attach losses")
    args = ap.parse_args()

    bench = {}
    for line in open(args.benchmark, encoding="utf-8"):
        if line.strip():
            b = json.loads(line)
            bench[b["id"]] = b

    n = 0
    fa_expect = ar_expect = en_expect = 0
    fa_got_fa = ar_got_ar = en_got_en = 0
    fa_leak_ar = 0
    empty = 0
    repetitive = 0
    malformed_n = 0
    len_sum = 0
    by_cat = defaultdict(Counter)
    wrong = []

    for line in open(args.responses, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        b = bench.get(r["id"], {})
        expect = b.get("expect_lang", "fa")
        resp = (r.get("response") or "").strip()
        # defense: classify the ANSWER, not a Qwen3 thinking trace
        if "</think>" in resp:
            tail = resp.split("</think>", 1)[1].strip()
            if tail:
                resp = tail
            else:
                resp = ""  # thinking-only output (no answer produced)
        elif resp.startswith("<think>"):
            resp = ""  # unclosed think block = no answer
        cat = b.get("category", "unknown")
        n += 1
        by_cat[cat]["n"] += 1
        if not resp:
            empty += 1
            by_cat[cat]["empty"] += 1
            wrong.append({"id": r["id"], "expect": expect, "got": "empty",
                          "prompt": b.get("prompt", r["id"])[:120], "response": ""})
            continue
        len_sum += len(resp)
        m = malformed(resp)
        if m:
            malformed_n += 1
            by_cat[cat]["malformed"] += 1
        if trigram_rep(resp) > 0.45:
            repetitive += 1
            by_cat[cat]["repetitive"] += 1
        got = classify(resp)[0]
        if expect == "fa":
            fa_expect += 1
            by_cat[cat]["fa_expect"] += 1
            if got == "fa":
                fa_got_fa += 1
                by_cat[cat]["fa_ok"] += 1
            elif got == "ar":
                fa_leak_ar += 1
                wrong.append({"id": r["id"], "expect": "fa", "got": "ar",
                              "prompt": b.get("prompt", "")[:120],
                              "response": resp[:200]})
            else:
                wrong.append({"id": r["id"], "expect": "fa", "got": got,
                              "prompt": b.get("prompt", "")[:120],
                              "response": resp[:200]})
        elif expect == "ar":
            ar_expect += 1
            by_cat[cat]["ar_expect"] += 1
            if got == "ar":
                ar_got_ar += 1
                by_cat[cat]["ar_ok"] += 1
            else:
                wrong.append({"id": r["id"], "expect": "ar", "got": got,
                              "prompt": b.get("prompt", "")[:120],
                              "response": resp[:200]})
        elif expect == "en":
            en_expect += 1
            by_cat[cat]["en_expect"] += 1
            if got == "en":
                en_got_en += 1
                by_cat[cat]["en_ok"] += 1
            else:
                wrong.append({"id": r["id"], "expect": "en", "got": got,
                              "prompt": b.get("prompt", "")[:120],
                              "response": resp[:200]})

    compliant = fa_got_fa + ar_got_ar + en_got_en
    total_expected = fa_expect + ar_expect + en_expect
    metrics = {
        "model": args.label,
        "n_prompts": n,
        "persian_response_rate": round(fa_got_fa / fa_expect, 4) if fa_expect else None,
        "language_compliance": round(compliant / total_expected, 4) if total_expected else None,
        "arabic_leakage_rate_on_fa": round(fa_leak_ar / fa_expect, 4) if fa_expect else None,
        "empty_response_rate": round(empty / n, 4) if n else None,
        "repetition_rate": round(repetitive / n, 4) if n else None,
        "malformed_response_rate": round(malformed_n / n, 4) if n else None,
        "average_response_length_chars": round(len_sum / max(1, n - empty), 1),
        "average_response_length_tokens_est": round(len_sum / max(1, n - empty) / 2.2, 1),
        "expect_counts": {"fa": fa_expect, "ar": ar_expect, "en": en_expect},
        "by_category": {k: dict(v) for k, v in sorted(by_cat.items())},
        "misclassified_examples": wrong[:40],
        "misclassified_total": len(wrong),
    }

    if args.train_summary and os.path.exists(args.train_summary):
        ts = json.load(open(args.train_summary))
        metrics["final_train_loss"] = ts.get("final_train_loss") or ts.get("last_train_loss")
        metrics["best_validation_loss"] = ts.get("best_val_loss")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(metrics, open(args.out, "w"), ensure_ascii=False, indent=2)

    # markdown report
    md = f"""# Persian Benchmark — {args.label}

| metric | value |
|---|---|
| prompts | {n} |
| persian_response_rate | {metrics['persian_response_rate']} |
| language_compliance | {metrics['language_compliance']} |
| arabic_leakage_rate_on_fa | {metrics['arabic_leakage_rate_on_fa']} |
| empty_response_rate | {metrics['empty_response_rate']} |
| repetition_rate | {metrics['repetition_rate']} |
| malformed_response_rate | {metrics['malformed_response_rate']} |
| avg_response_length (chars) | {metrics['average_response_length_chars']} |
| final_train_loss | {metrics.get('final_train_loss')} |
| best_validation_loss | {metrics.get('best_validation_loss')} |
"""
    open(args.out.replace(".json", ".md"), "w").write(md)
    print(json.dumps({k: v for k, v in metrics.items() if k != "misclassified_examples"},
                     ensure_ascii=False, indent=2))
    print(f"[metrics] {len(wrong)} misclassified examples (first 40 in {args.out})")


if __name__ == "__main__":
    main()
