#!/usr/bin/env python3
"""ARION ALPHA 1 — Persian dataset preparation pipeline.

Source: ParsBench/PersianSyntheticQA (Apache-2.0, ~100k Persian QA conversations,
50 domains) + locally generated Persian booster data (colloquial / multi-turn /
language-switching / fa+en technical, via scripts under training/).

Stages (all counts logged into persian_stats.json):
  1. load raw parquet (+ booster jsonl)
  2. Persian normalization (training/persian_norm.py — code/URL-safe)
  3. structural validation (roles, empty, malformed)
  4. quality filters (too long, corrupted Unicode, repetitive, placeholders,
     unfinished, secrets, Arabic-language leakage)
  5. exact-duplicate removal (hash)
  6. near-duplicate removal (MinHash-LSH, Jaccard >= 0.85)
  7. quality-ranked subsample to --target (default 45000), domain-balanced
  8. deterministic 90/10 split with near-dup-group isolation (NO leakage)
  9. outputs: datasets/processed/persian_train.jsonl, persian_validation.jsonl,
     persian_stats.json

Usage:
  python3 training/prepare_persian_dataset.py [--target 45000] [--seed 42]
      [--raw-dir datasets/raw/persiansyntheticqa] [--subset N]
"""
import argparse, glob, hashlib, json, os, random, re, sys
from collections import Counter, defaultdict

import numpy as np

# deterministic hashing across runs (hash() is process-randomized)
if os.environ.get("PYTHONHASHSEED") != "0":
    _env = {**os.environ, "PYTHONHASHSEED": "0"}
    os.execve(sys.executable, [sys.executable] + sys.argv, _env)
ROOT = os.environ.get("ARION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from persian_norm import normalize_conversation, normalize_text, corruption_score  # noqa: E402

OUT = f"{ROOT}/datasets/processed"

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ghp_[A-Za-z0-9]{30,}"),
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{16,}['\"]"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-\.]{25,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"KGAT_[A-Za-z0-9]{20,}"),
]

PLACEHOLDER_RE = re.compile(r"\{\{?\s*\w+\s*\}?\}|\[[نام توضیح موضوع]\]|<نام>|TODO|Lorem ipsum")

# Arabic MSA function words (rare in Persian prose; strong Arabic signal)
ARABIC_MARKERS = ["في", "من", "على", "هذا", "هذه", "التي", "الذي", "ذلك", "تلك",
                  "كما", "بينما", "إن", "أو", "لقد", "قد", "عندما", "حيث", "إلى",
                  "التى", "كان", "يجب أن", "لكن", "كل", "بعض", "هناك"]
# Persian function words / morphology (strong Persian signal)
PERSIAN_MARKERS = ["است", "را", "برای", "که", "های", "هاي", "بود", "هستند", "نیست",
                   "می\u200c", "نمی\u200c", "خیلی", "بسیار", "این", "آن", "با", "تا",
                   "اما", "هم", "یک", "شود", "شد", "می\u200cشود", "خود", "کرد", "کند",
                   "باشد", "چگونه", "چطور", "چه", "هر"]

PERSIAN_CHARS = set("\u06CC\u06A9\u06AF\u067E\u0686\u0698")  # ی ک گ پ چ ژ — Persian-only letters


def has_arabic_only_letters(t):
    """Letters that exist in Arabic but NOT in Persian alphabet usage."""
    return sum(t.count(c) for c in "\u0636\u0637\u0638\u0639\u063A\u0641\u0642\u062D\u062E\u062B\u0635\u0634\u0633\u062C\u0631\u0630\u062F\u0632\u0633")


def lang_of_text(text):
    """Heuristic language classification: fa | ar | en | mixed | other.

    Word-level function-word scoring (NOT mere character overlap), per spec.
    """
    if not text:
        return "other"
    words = text.split()
    n = len(words)
    arabic_hits = sum(text.count(m) for m in ARABIC_MARKERS)
    persian_hits = sum(text.count(m) for m in PERSIAN_MARKERS)
    persian_letters = sum(1 for c in text if c in PERSIAN_CHARS)
    ascii_letters = sum(1 for c in text if c.isascii() and c.isalpha())
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF" or "\u0750" <= c <= "\u077F")
    if ascii_letters > 0.7 * max(1, len(text)):
        return "en"
    if persian_letters >= 1 and persian_hits >= arabic_hits:
        return "fa"
    if arabic_hits >= 3 and persian_hits == 0 and persian_letters == 0:
        return "ar"
    if arabic_chars > 0.3 * max(1, len(text)):
        return "fa" if (persian_hits or persian_letters) else "mixed"
    if ascii_letters > 0.3 * max(1, len(text)):
        return "mixed"
    return "other"


def conversation_lang(msgs):
    """Language of user turns decides the conversation's language label."""
    langs = [lang_of_text(m["content"]) for m in msgs if m["role"] == "user"]
    fa = sum(1 for l in langs if l == "fa")
    ar = sum(1 for l in langs if l == "ar")
    en = sum(1 for l in langs if l == "en")
    if ar and ar >= fa:
        return "ar"
    if fa and en:
        return "fa_en"
    if fa:
        return "fa"
    if en:
        return "en"
    return "mixed"


def repetition_score(t, n=3):
    words = t.split()
    if len(words) < n + 2:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return (len(grams) - len(set(grams))) / len(grams)


def max_gram_repeat(t, n=12):
    words = t.split()
    if len(words) < n:
        return 1
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return max(grams.values())


def hash64(s):
    b = s.encode("utf-8") if isinstance(s, str) else s
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


# ---------------- MinHash near-duplicate detection ----------------
class MinHasher:
    """Memory-frugal MinHash: signatures only (no shingle-set retention).
    A,B < 2^31 and ids < 2^32 keep uint64 products below 2^63 (no overflow),
    so signature agreement is an unbiased Jaccard estimator (32 perms,
    std-err ~0.065 at J=0.85 — sufficient for near-dup grouping)."""

    def __init__(self, num_perm=32, shingle=5, seed=1):
        rng = np.random.RandomState(seed)
        self.num_perm = num_perm
        self.shingle = shingle
        self.A = rng.randint(1, 2**31, size=num_perm).astype(np.uint64)
        self.B = rng.randint(0, 2**31, size=num_perm).astype(np.uint64)

    def sig(self, text):
        words = re.sub(r"\s+", " ", text).split()
        if len(words) < self.shingle:
            grams = [" ".join(words)] if words else [""]
        else:
            grams = [" ".join(words[i:i + self.shingle]) for i in range(len(words) - self.shingle + 1)]
        ids = np.fromiter((hash(g) & 0xFFFFFFFF for g in grams),
                          dtype=np.uint64, count=len(grams))
        h = self.A[:, None] * ids[None, :] + self.B[:, None]
        return h.min(axis=1)

    @staticmethod
    def est_jaccard(sig1, sig2):
        return float((sig1 == sig2).mean())


def near_dedup(rows, hasher, threshold=0.85, log_every=20000):
    """Union-find near-duplicate grouping via MinHash-LSH (band=4).
    Returns (parent_fn, dup_count, candidate_pairs)."""
    n = len(rows)
    sigs = []
    for i, r in enumerate(rows):
        sigs.append(hasher.sig(r["_nd_text"]))
        if (i + 1) % log_every == 0:
            print(f"  [minhash] {i + 1}/{n}", flush=True)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    bands = hasher.num_perm // 4
    buckets = defaultdict(list)
    for i in range(n):
        s = sigs[i]
        for b in range(bands):
            buckets[(b, hash(s[b * 4:(b + 1) * 4].tobytes()))].append(i)
    seen_pairs = set()
    cand = 0
    for key, members in buckets.items():
        if len(members) < 2 or len(members) > 60:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = members[x], members[y]
                if a > b:
                    a, b = b, a
                if (a, b) in seen_pairs:
                    continue
                seen_pairs.add((a, b))
                cand += 1
                if MinHasher.est_jaccard(sigs[a], sigs[b]) >= threshold:
                    union(a, b)
                if cand % 200000 == 0:
                    print(f"  [lsh] {cand} pairs verified", flush=True)
    dup_count = sum(1 for i in range(n) if find(i) != i)
    return find, dup_count, cand


# ---------------- quality filter ----------------
def quality_filter(msgs, meta, cfg):
    """Return 'ok' or a rejection reason. msgs already normalized."""
    roles = [m["role"] for m in msgs]
    if roles[0] != "user" or roles[-1] != "assistant":
        return "bad_role_structure"
    for i in range(1, len(roles)):
        if roles[i] == roles[i - 1]:
            return "bad_role_structure"
    for m in msgs:
        if not isinstance(m.get("content"), str):
            return "malformed"
        c = m["content"]
        if not c.strip():
            return "empty_content"
        if "\ufffd" in c:
            return "corrupted_unicode"
        if corruption_score(c) > 0.02:
            return "corrupted_unicode"

    user_txt = " ".join(m["content"] for m in msgs if m["role"] == "user")
    asst_txt = " ".join(m["content"] for m in msgs if m["role"] == "assistant")

    # conversation-level language control
    if conversation_lang(msgs) == "ar":
        return "arabic_language"

    if len(asst_txt) < cfg["min_answer_chars"]:
        return "answer_too_short"
    if len(user_txt) < cfg["min_prompt_chars"]:
        return "prompt_too_short"
    if len(asst_txt) > cfg["max_answer_chars"]:
        return "too_long"
    # token estimate: Persian ~2.2 chars/token with Qwen tokenizer, be conservative
    est_tokens = (len(user_txt) + len(asst_txt)) / 2.0
    if est_tokens > cfg["max_seq_len"] * 0.9:
        return "too_long"

    if repetition_score(asst_txt) > 0.45:
        return "repetitive"
    if max_gram_repeat(asst_txt, 12) >= 3:
        return "repetitive"
    uniq_ratio = len(set(asst_txt.split())) / max(1, len(asst_txt.split()))
    if len(asst_txt.split()) > 40 and uniq_ratio < 0.12:
        return "low_lexical_diversity"

    if PLACEHOLDER_RE.search(asst_txt):
        return "placeholder_artifact"
    # unfinished responses
    tail = asst_txt.rstrip()
    if tail.endswith(("،", ",", ":", "؛", " و", " یا", " که", " برای")):
        return "unfinished"
    if tail.count("```") % 2 == 1:
        return "unfinished_code_fence"

    for p in SECRET_PATTERNS:
        if p.search(asst_txt) or p.search(user_txt):
            return "secret_like"
    return "ok"


def quality_rank(msgs, meta):
    """Higher = better; used for quality-ranked subsampling."""
    asst = " ".join(m["content"] for m in msgs if m["role"] == "assistant")
    user = " ".join(m["content"] for m in msgs if m["role"] == "user")
    score = 0.0
    score += min(1.0, len(asst) / 900) * 2.0              # informative answers
    score += min(1.0, asst.count("\u200c") / 8) * 1.5     # natural Persian ZWNJ usage
    score += (1.0 - repetition_score(asst)) * 1.5
    uniq = len(set(asst.split())) / max(1, len(asst.split()))
    score += min(1.0, uniq) * 1.5
    if meta.get("turns", 2) > 2:
        score += 2.0                                       # multi-turn is precious
    score += min(1.0, len(user) / 120) * 0.8
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=f"{ROOT}/datasets/raw/persiansyntheticqa")
    ap.add_argument("--target", type=int, default=45000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--subset", type=int, default=0, help="debug: only first N rows")
    ap.add_argument("--max-seq-len", type=int, default=3072)
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--booster", default=(
        f"{ROOT}/datasets/generated/persian_seed_lang.jsonl,"
        f"{ROOT}/datasets/generated/persian_booster.jsonl"),
        help="comma-separated booster jsonl files")
    ap.add_argument("--keep-html-set", action="store_true", default=False,
                    help="also merge the existing HTML train/val sets (off by default)")
    args = ap.parse_args()
    cfg = dict(min_answer_chars=60, min_prompt_chars=10, max_answer_chars=4000,
               max_seq_len=args.max_seq_len)
    rng = random.Random(args.seed)

    import pyarrow.parquet as pq
    stats = Counter()
    by_domain = Counter()
    rows = []

    files = sorted(glob.glob(f"{args.raw_dir}/*/train-*.parquet"))
    print(f"[load] {len(files)} domain parquet files", flush=True)
    for f in files:
        domain = os.path.basename(os.path.dirname(f))
        t = pq.read_table(f, columns=["messages"])
        for msgs in t.column("messages").to_pylist():
            stats["source_count"] += 1
            if not msgs or len(msgs) < 2:
                stats["rejected_malformed"] += 1
                continue
            turns = len(msgs) // 2
            rows.append({"messages": normalize_conversation(msgs),
                         "domain": domain, "turns": turns, "source": "parsbench"})
            by_domain[domain] += 1
            if args.subset and stats["source_count"] >= args.subset:
                break
        if args.subset and stats["source_count"] >= args.subset:
            break

    # booster data (locally generated Persian conversations)
    def load_booster(path):
        n_b = 0
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
                msgs = ex["messages"]
            except Exception:
                stats["booster_malformed"] += 1
                continue
            if not isinstance(msgs, list) or len(msgs) < 2:
                stats["booster_malformed"] += 1
                continue
            stats["source_count"] += 1
            rows.append({"messages": normalize_conversation(msgs),
                         "domain": ex.get("domain", "booster"),
                         "turns": len(msgs) // 2, "source": ex.get("source", "booster")})
            by_domain[ex.get("domain", "booster")] += 1
            n_b += 1
        print(f"[booster] {os.path.basename(path)}: loaded {n_b} examples", flush=True)

    for booster_path in [bp for bp in args.booster.split(",") if bp.strip()]:
        if os.path.exists(booster_path):
            load_booster(booster_path)
        else:
            print(f"[booster] {booster_path}: not found (skipped)", flush=True)

    print(f"[load] total raw rows: {len(rows)}", flush=True)

    # ---- quality filter ----
    kept = []
    for r in rows:
        reason = quality_filter(r["messages"], r, cfg)
        if reason != "ok":
            stats[f"rejected_{reason}"] += 1
        else:
            kept.append(r)
    print(f"[filter] kept {len(kept)} / {len(rows)}", flush=True)

    # ---- exact duplicates ----
    seen, deduped = set(), []
    for r in kept:
        h = hash64(json.dumps(r["messages"], ensure_ascii=False, sort_keys=True))
        if h in seen:
            stats["duplicate_exact"] += 1
            continue
        seen.add(h)
        deduped.append(r)
    print(f"[exact-dedup] {len(deduped)} rows", flush=True)

    # ---- near duplicates (MinHash LSH) ----
    for r in deduped:
        r["_nd_text"] = " ".join(m["content"] for m in r["messages"])
    hasher = MinHasher(num_perm=32, shingle=5, seed=args.seed)
    find, dup_count, cand = near_dedup(deduped, hasher, threshold=0.85)
    stats["duplicate_near"] = dup_count
    rows2 = [r for i, r in enumerate(deduped) if find(i) == i]
    for r in rows2:
        r.pop("_nd_text", None)
    print(f"[near-dedup] removed {dup_count} near-dups ({cand} candidate pairs), "
          f"{len(rows2)} remain", flush=True)

    # ---- language audit (informational) ----
    lang_counter = Counter(conversation_lang(r["messages"]) for r in rows2)
    print(f"[lang] {dict(lang_counter)}", flush=True)

    # ---- quality-ranked, domain-balanced subsample ----
    for r in rows2:
        r["_score"] = quality_rank(r["messages"], r)
    target = args.target
    multi = [r for r in rows2 if r["turns"] > 1]
    single = [r for r in rows2 if r["turns"] == 1]
    picked = list(multi)  # keep ALL multi-turn
    per_domain_cap = max(1, (target - len(picked)) // max(1, len(by_domain)))
    dom_count = Counter()
    by_domain_scored = defaultdict(list)
    for r in single:
        by_domain_scored[r["domain"]].append(r)
    for d, lst in by_domain_scored.items():
        lst.sort(key=lambda r: (-r["_score"], hash64(r["_nd_text"] if "_nd_text" in r else d)))
        for r in lst[:per_domain_cap]:
            picked.append(r)
            dom_count[d] += 1
    if len(picked) > target:
        picked.sort(key=lambda r: -r["_score"])
        # never drop multi-turn in favor of single-turn
        kept_mt = [r for r in picked if r["turns"] > 1]
        kept_st = [r for r in picked if r["turns"] == 1][:max(0, target - len(kept_mt))]
        picked = kept_mt + kept_st
    stats["accepted_count"] = len(picked)
    stats["multi_turn_kept"] = len([r for r in picked if r["turns"] > 1])
    print(f"[subsample] picked {len(picked)} (target {target}, per-domain cap {per_domain_cap})",
          flush=True)

    # ---- deterministic 90/10 split with near-dup isolation ----
    # near-dup groups: rows2 indices collapsed; picked rows inherit group root
    idx_of = {id(r): i for i, r in enumerate(deduped)}
    def group_key(r):
        i = idx_of.get(id(r))
        return find(i) if i is not None else hash64(json.dumps(r["messages"], ensure_ascii=False))
    groups = defaultdict(list)
    for r in picked:
        g = group_key(r)
        groups[g].append(r)
    train, val = [], []
    for g, members in groups.items():
        h = int.from_bytes(hashlib.sha256(f"{args.seed}:{g}".encode()).digest()[:8], "little")
        if (h % 1000) < args.val_frac * 1000:
            val.extend(members)
        else:
            train.extend(members)
    rng.shuffle(train)
    rng.shuffle(val)
    print(f"[split] train={len(train)} val={len(val)}", flush=True)

    # ---- write ----
    os.makedirs(OUT, exist_ok=True)
    def dump(path, lst):
        with open(path, "w", encoding="utf-8") as f:
            for r in lst:
                f.write(json.dumps({
                    "messages": r["messages"],
                    "domain": r["domain"],
                    "source": r["source"],
                }, ensure_ascii=False) + "\n")
    dump(f"{OUT}/persian_train.jsonl", train)
    dump(f"{OUT}/persian_validation.jsonl", val)

    # ---- stats ----
    def lengths(lst):
        u = [sum(len(m["content"]) for m in r["messages"] if m["role"] == "user") for r in lst]
        a = [sum(len(m["content"]) for m in r["messages"] if m["role"] == "assistant") for r in lst]
        tot = sorted((x + y) for x, y in zip(u, a))
        n = len(tot) or 1
        return {
            "average_prompt_length_chars": round(sum(u) / n, 1),
            "average_response_length_chars": round(sum(a) / n, 1),
            "p50_total_chars": tot[n // 2],
            "p95_total_chars": tot[int(n * 0.95)],
            "max_total_chars": tot[-1],
            "est_p50_tokens": tot[n // 2] // 2,
            "est_p95_tokens": tot[int(n * 0.95)] // 2,
            "est_max_tokens": tot[-1] // 2,
        }

    stats_full = {
        "source_count": stats["source_count"],
        "accepted_count": stats["accepted_count"],
        "rejected_count": sum(v for k, v in stats.items() if k.startswith("rejected_")),
        "rejections": {k.replace("rejected_", ""): v for k, v in stats.items() if k.startswith("rejected_")},
        "duplicate_count": stats["duplicate_exact"] + stats["duplicate_near"],
        "duplicate_exact": stats["duplicate_exact"],
        "duplicate_near": stats["duplicate_near"],
        "train_count": len(train),
        "validation_count": len(val),
        "language_distribution": dict(lang_counter),
        "source_distribution": Counter(r["source"] for r in picked),
        "domain_distribution": dict(Counter(r["domain"] for r in picked)),
        "turn_distribution": dict(Counter(r["turns"] for r in picked)),
        "train": lengths(train),
        "validation": lengths(val),
        "seed": args.seed,
        "target": target,
        "max_seq_len_filter": args.max_seq_len,
        "near_dedup_threshold_jaccard": 0.85,
        "minhash_perms": 32,
    }
    json.dump(stats_full, open(f"{OUT}/persian_stats.json", "w"), ensure_ascii=False, indent=2)
    print("[done] wrote persian_train.jsonl / persian_validation.jsonl / persian_stats.json",
          flush=True)


if __name__ == "__main__":
    main()
