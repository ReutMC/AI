#!/usr/bin/env python3
"""ARION ALPHA 1 — llama.cpp-backed evaluation re-runner (dashboard "Re-run eval" button).

Talks to a running llama-server (OpenAI-compatible /v1/chat/completions) instead of loading
torch/transformers — RAM footprint ~20 MB, so it can run alongside the ~600 MB llama-server
even on this 4 GB machine. Metrics (HTML/CSS/JS syntax, language, relevance, instruction
following, repetition) are imported unchanged from run_eval.py.

Writes progress to artifacts/eval_progress.json after each prompt so the dashboard can poll.
Before overwriting artifacts/eval_report.json, archives the previous report to
artifacts/eval_history/eval-<timestamp>.json (keeps newest 5) so the UI can show deltas.

Usage:
  python3 evaluation/run_eval_llama.py --port 3099 --set both --limit 6 \
      --out artifacts/eval_report.json
"""
import argparse, json, os, re, shutil, sys, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_eval import SYS, evaluate_answer  # reuse identical metrics

# if the model server dies mid-run (OOM kill, crash), further requests fail with these:
FATAL_PATTERNS = ("Connection refused", "closed connection without response", "Remote end closed")


def chat(port, prompt, max_new, timeout=240):
    """Greedy generation via llama-server (temperature 0 ≈ run_eval.py do_sample=False)."""
    body = json.dumps({
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": max_new,
        "frequency_penalty": 1.08,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.loads(r.read().decode("utf-8"))
    return (j["choices"][0]["message"].get("content") or "").strip(), \
           (j.get("usage") or {}).get("completion_tokens", 0)


def write_progress(path, **kw):
    try:
        json.dump(kw, open(path, "w"), ensure_ascii=False)
    except Exception:
        pass


def archive_previous(report_path, hist_dir, keep=5):
    if not os.path.exists(report_path):
        return None
    os.makedirs(hist_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(hist_dir, f"eval-{ts}.json")
    try:
        shutil.copy2(report_path, dest)
        # prune old archives beyond `keep`
        olds = sorted(f for f in os.listdir(hist_dir) if f.startswith("eval-") and f.endswith(".json"))
        for f in olds[:-keep]:
            try:
                os.unlink(os.path.join(hist_dir, f))
            except OSError:
                pass
        return dest
    except Exception:
        return None


def summarize(name, rows):
    """Same summary math as run_eval.py, computed over whatever rows exist so far."""
    ok = lambda k: sum(1 for r in rows if r["metrics"].get(k)) / max(1, len(rows))
    syntax_rows = [r for r in rows if r["metrics"]["syntax_valid"] is not None]
    return {
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


def stratified(prompts, limit):
    """Pick `limit` prompts spread across categories (id prefix minus -NNN) by round-robin,
    so a short run covers every skill area / language instead of only the file's first N.
    Order within each category and category order stay stable across runs."""
    if not limit or limit >= len(prompts):
        return prompts[: limit] if limit else list(prompts)
    groups: dict[str, list] = {}
    for p in prompts:
        cat = re.sub(r"-?\d+$", "", p["id"]) or "misc"
        groups.setdefault(cat, []).append(p)
    cats = sorted(groups)
    out, i = [], 0
    while len(out) < limit:
        added = False
        for c in cats:
            if i < len(groups[c]):
                out.append(groups[c][i])
                added = True
                if len(out) >= limit:
                    break
        if not added:
            break
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=3099)
    ap.add_argument("--set", dest="set_name", default="both", choices=["webdev", "bilingual", "both"])
    ap.add_argument("--limit", type=int, default=6, help="first N prompts per set (0 = all — very slow on CPU)")
    ap.add_argument("--max-new", type=int, default=420)
    ap.add_argument("--out", default=f"{ROOT}/artifacts/eval_report.json")
    ap.add_argument("--progress", default=f"{ROOT}/artifacts/eval_progress.json")
    ap.add_argument("--tag", default="arion_eval_runner",
                    help="process tag so pgrep -f arion_eval_[r]unner can find this runner (ignored otherwise)")
    args = ap.parse_args()

    sets = []
    if args.set_name in ("webdev", "both"):
        sets.append(("webdev", json.load(open(f"{ROOT}/evaluation/prompts_webdev.json"))["prompts"]))
    if args.set_name in ("bilingual", "both"):
        sets.append(("bilingual", json.load(open(f"{ROOT}/evaluation/prompts_bilingual.json"))["prompts"]))
    if args.limit:
        sets = [(n, stratified(ps, args.limit)) for n, ps in sets]

    total = sum(len(ps) for _, ps in sets)
    write_progress(args.progress, running=True, done=0, total=total, set=args.set_name,
                   limit=args.limit, started=time.strftime("%Y-%m-%d %H:%M:%S"))

    done = 0
    t0 = time.time()
    consec_fail = 0  # consecutive server-dead failures → abort without overwriting the report
    report = {"model": "llama-server Q4_K_M (re-run)", "started": time.strftime("%Y-%m-%d %H:%M:%S"), "sets": {}}
    partial_rows: dict[str, list] = {}  # live rows per set → running summaries in progress file
    for name, prompts in sets:
        print(f"== {name}: {len(prompts)} prompts ==", flush=True)
        rows = []
        partial_rows[name] = rows
        for p in prompts:
            expect = p.get("expect") or ("fa" if p.get("lang") == "fa" else "en")
            try:
                ans, toks = chat(args.port, p["prompt"], args.max_new)
            except Exception as e:
                ans, toks = "", 0
                print(f"  !! {p['id']} generation failed: {e}", flush=True)
                if any(pat in str(e) for pat in FATAL_PATTERNS):
                    consec_fail += 1
                    if consec_fail >= 2:
                        # server is gone (OOM/crash) — abort WITHOUT touching the report
                        msg = (f"model server died mid-run ({e}) after {done}/{total} prompts — "
                               "run aborted, existing report NOT overwritten")
                        print("FATAL:", msg, flush=True)
                        write_progress(args.progress, running=False, done=done, total=total,
                                       set=args.set_name, limit=args.limit,
                                       started=report["started"], elapsed_sec=round(time.time() - t0),
                                       error=msg, fatal=True)
                        sys.exit(1)
                    continue
            else:
                consec_fail = 0
            m = evaluate_answer(ans, p["prompt"], expect)
            m["completion_tokens"] = toks
            rows.append({"id": p["id"], "prompt": p["prompt"], "metrics": m, "answer": ans[:1200]})
            done += 1
            el = time.time() - t0
            eta = (el / done) * (total - done) if done else 0
            print(f"  [{done}/{total}] {p['id']} {toks} tok ({el/60:.1f} min, eta {eta/60:.1f})", flush=True)
            # live partial summaries so the dashboard can show metric bars mid-run
            write_progress(args.progress, running=True, done=done, total=total, set=args.set_name,
                           limit=args.limit, current=p["id"],
                           started=report["started"], elapsed_sec=round(el),
                           partial={n: summarize(n, rs) for n, rs in partial_rows.items() if rs})
        summary = summarize(name, rows)
        report["sets"][name] = {"summary": summary, "rows": rows}
        print(json.dumps(summary, indent=2), flush=True)

    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    archived = archive_previous(args.out, os.path.join(os.path.dirname(args.out), "eval_history"))
    json.dump(report, open(args.out, "w"), indent=2, ensure_ascii=False)
    write_progress(args.progress, running=False, done=done, total=total, set=args.set_name,
                   limit=args.limit, started=report["started"], finished=report["finished"],
                   elapsed_sec=round(time.time() - t0), archived_to=os.path.basename(archived) if archived else None)
    print("report saved →", args.out, f"(previous archived: {archived})", flush=True)


if __name__ == "__main__":
    main()
