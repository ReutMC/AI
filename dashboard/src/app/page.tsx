"use client";

import { useEffect, useState } from "react";

type Metrics = {
  persian_response_rate?: number | null;
  language_compliance?: number | null;
  arabic_leakage_rate_on_fa?: number | null;
  empty_response_rate?: number | null;
  repetition_rate?: number | null;
  malformed_response_rate?: number | null;
  average_response_length_chars?: number | null;
  n_prompts?: number;
} | null;

type StatusData = {
  project: string;
  git: { commit: string; branch: string };
  dataset: {
    stats: Record<string, unknown> | null;
    trainCount: number | null;
    valCount: number | null;
  };
  booster: { generated: number; handAuthored: number };
  benchmark: { baseline: Metrics; arion: Metrics };
  training: { summary: Record<string, unknown> | null; seqBenchmark: Record<string, unknown> | null };
  artifactFiles: string[];
};

function pct(v: number | null | undefined): string {
  return typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—";
}

function MetricCard({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: "good" | "warn" | "neutral" }) {
  const toneCls =
    tone === "good" ? "border-emerald-500/40 bg-emerald-500/5" :
    tone === "warn" ? "border-amber-500/40 bg-amber-500/5" :
    "border-zinc-700/60 bg-zinc-900/60";
  return (
    <div className={`rounded-xl border p-4 ${toneCls}`}>
      <div className="text-xs uppercase tracking-wider text-zinc-400">{label}</div>
      <div className="mt-1 text-2xl font-bold text-zinc-100">{value}</div>
      {sub && <div className="mt-1 text-xs text-zinc-500">{sub}</div>}
    </div>
  );
}

function CompareTable({ baseline, arion }: { baseline: Metrics; arion: Metrics }) {
  const rows: { key: keyof NonNullable<Metrics>; label: string; higherBetter: boolean }[] = [
    { key: "persian_response_rate", label: "Persian response rate", higherBetter: true },
    { key: "language_compliance", label: "Requested-language compliance", higherBetter: true },
    { key: "arabic_leakage_rate_on_fa", label: "Arabic leakage (on fa prompts)", higherBetter: false },
    { key: "empty_response_rate", label: "Empty responses", higherBetter: false },
    { key: "repetition_rate", label: "Repetition rate", higherBetter: false },
    { key: "malformed_response_rate", label: "Malformed responses", higherBetter: false },
  ];
  return (
    <div className="overflow-x-auto rounded-xl border border-zinc-800">
      <table className="w-full text-sm">
        <thead className="bg-zinc-900 text-left text-xs uppercase tracking-wider text-zinc-400">
          <tr>
            <th className="px-4 py-3">Metric (239-prompt Persian benchmark)</th>
            <th className="px-4 py-3">BASE Qwen3-0.6B</th>
            <th className="px-4 py-3">ARION (GGUF)</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-zinc-800">
          {rows.map((r) => (
            <tr key={r.key} className="text-zinc-300">
              <td className="px-4 py-2.5">{r.label}</td>
              <td className="px-4 py-2.5 font-mono">{baseline ? pct(baseline[r.key]) : "pending"}</td>
              <td className="px-4 py-2.5 font-mono text-emerald-400">{arion ? pct(arion[r.key]) : "pending"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function Home() {
  const [data, setData] = useState<StatusData | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/status")
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setErr(String(e)));
    const t = setInterval(() => {
      fetch("/api/status").then((r) => r.json()).then(setData).catch(() => {});
    }, 30_000);
    return () => clearInterval(t);
  }, []);

  const stats = data?.dataset.stats as Record<string, unknown> | null;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <main className="mx-auto max-w-6xl px-4 py-8 sm:px-6">
        {/* header */}
        <header className="mb-8 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-2xl font-black tracking-tight sm:text-3xl">
              🦁 ARION ALPHA 1 <span className="text-emerald-400">— Persian phase</span>
            </h1>
            <p className="mt-1 text-sm text-zinc-400">
              Qwen3-0.6B + LoRA · Kaggle GPU pipeline · bilingual FA/EN web-dev assistant
            </p>
          </div>
          <div className="rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-xs text-zinc-400">
            git <span className="font-mono text-zinc-200">{data?.git.commit ?? "…"}</span> ·{" "}
            <span className="font-mono">{data?.git.branch ?? "main"}</span>
          </div>
        </header>

        {err && (
          <div className="mb-6 rounded-xl border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-300">
            status API error: {err}
          </div>
        )}

        {/* key numbers */}
        <section className="grid grid-cols-2 gap-4 lg:grid-cols-4" aria-label="dataset summary">
          <MetricCard
            label="Accepted examples"
            value={stats ? (stats.accepted_count as number).toLocaleString("en") : "…"}
            sub="quality-ranked, domain-balanced"
            tone="good"
          />
          <MetricCard
            label="Near-duplicates removed"
            value={stats ? (stats.duplicate_near as number).toLocaleString("en") : "…"}
            sub="MinHash-LSH, Jaccard ≥ 0.85"
          />
          <MetricCard
            label="Train / Validation"
            value={data?.dataset.trainCount != null ? `${data.dataset.trainCount.toLocaleString("en")} / ${(data.dataset.valCount ?? 0).toLocaleString("en")}` : "…"}
            sub="deterministic 90/10, leak-free"
          />
          <MetricCard
            label="Booster conversations"
            value={data ? `${data.booster.handAuthored} + ${data.booster.generated}` : "…"}
            sub="hand-authored + LLM-generated"
          />
        </section>

        {/* benchmark compare */}
        <section className="mt-8" aria-label="benchmark comparison">
          <h2 className="mb-3 text-lg font-bold">🇮🇷 Persian fluency — BASE vs ARION</h2>
          <CompareTable baseline={data?.benchmark.baseline ?? null} arion={data?.benchmark.arion ?? null} />
          <p className="mt-2 text-xs text-zinc-500">
            Measured with function-word language-ID (not character overlap). Baseline runs on the
            untrained base model; ARION on the fine-tuned GGUF. &ldquo;pending&rdquo; = run not
            finished yet (GPU requires a phone-verified Kaggle account; CPU fallback validated the
            full pipeline end-to-end).
          </p>
        </section>

        {/* pipeline status */}
        <section className="mt-8 grid gap-4 md:grid-cols-2" aria-label="pipeline status">
          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
            <h3 className="mb-2 font-semibold">Pipeline stages</h3>
            <ul className="space-y-1.5 text-sm text-zinc-300">
              {[
                ["Dataset pipeline (normalize → filter → dedup → split)", true],
                ["Persian benchmark (239 prompts, 12 categories)", true],
                ["Kaggle dataset bundle v7 (offline: base model + llama.cpp)", true],
                ["CPU smoke: train → merge → GGUF → llama-cli FA smoke", true],
                ["CPU baseline benchmark (239 prompts)", data?.benchmark.baseline != null],
                ["GPU seq-len benchmark (2048/3072/4096)", data?.training.seqBenchmark != null],
                ["GPU full training + GGUF + eval-on-GGUF", data?.benchmark.arion != null],
                ["Release v1.1 (Q4_K_M / Q8_0 / F16)", false],
              ].map(([label, done]) => (
                <li key={String(label)} className="flex items-center gap-2">
                  <span className={done ? "text-emerald-400" : "text-zinc-600"}>{done ? "✓" : "○"}</span>
                  <span className={done ? "" : "text-zinc-500"}>{String(label)}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
            <h3 className="mb-2 font-semibold">Artifacts on disk</h3>
            {data?.artifactFiles.length ? (
              <ul className="space-y-1 font-mono text-xs text-zinc-400">
                {data.artifactFiles.map((f) => (
                  <li key={f}>📄 {f}</li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-zinc-500">
                No Kaggle artifacts fetched yet. After each run:
                <code className="ml-1 rounded bg-zinc-800 px-1 py-0.5 text-xs">
                  python3 training/kaggle_pipeline.py fetch --kind baseline-cpu
                </code>
              </p>
            )}
            <div className="mt-3 rounded-lg border border-amber-500/30 bg-amber-500/5 p-2.5 text-xs text-amber-300">
              ⚠️ Kaggle GPU/internet needs phone verification on the Kaggle account
              (kaggle.com/settings). Until then the GPU kernels fail-fast by design and the CPU
              fallback keeps the pipeline validated.
            </div>
          </div>
        </section>
      </main>

      <footer className="mt-auto border-t border-zinc-800 bg-zinc-900/80 px-4 py-4 text-center text-xs text-zinc-500">
        ARION ALPHA 1 · Persian phase v1.1 · Apache-2.0 · data: ParsBench/PersianSyntheticQA + authored boosters
      </footer>
    </div>
  );
}
