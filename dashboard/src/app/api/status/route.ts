import { NextResponse } from "next/server";
import { readFileSync, existsSync, readdirSync } from "fs";
import { execSync } from "child_process";
import path from "path";

export const dynamic = "force-dynamic";

const REPO = "/home/z/my-project";

function safeJson(p: string): unknown | null {
  try {
    return JSON.parse(readFileSync(p, "utf-8"));
  } catch {
    return null;
  }
}

function countJsonl(p: string): number | null {
  try {
    const s = readFileSync(p, "utf-8");
    return s.split("\n").filter((l) => l.trim()).length;
  } catch {
    return null;
  }
}

export async function GET() {
  // git info
  let commit = "unknown";
  let branch = "main";
  try {
    commit = execSync("git -C /home/z/my-project rev-parse --short HEAD").toString().trim();
    branch = execSync("git -C /home/z/my-project rev-parse --abbrev-ref HEAD").toString().trim();
  } catch { /* noop */ }

  // dataset stats
  const stats = safeJson(path.join(REPO, "datasets/processed/persian_stats.json")) as Record<string, unknown> | null;

  // benchmark metrics (baseline / arion) — produced by Kaggle runs
  const artifactsDir = path.join(REPO, "artifacts/persian");
  const baselineMetrics = safeJson(path.join(artifactsDir, "baseline_metrics.json"));
  const arionMetrics = safeJson(path.join(artifactsDir, "arion_gguf_metrics.json"));
  const trainSummary = safeJson(path.join(artifactsDir, "train_summary.json"));
  const seqBench = safeJson(path.join(artifactsDir, "seq_benchmark.json"));
  let artifactFiles: string[] = [];
  if (existsSync(artifactsDir)) {
    artifactFiles = readdirSync(artifactsDir).filter((f) => !f.startsWith("."));
  }

  // booster data
  const boosterCount = countJsonl(path.join(REPO, "datasets/generated/persian_booster.jsonl"));
  const seedCount = countJsonl(path.join(REPO, "datasets/generated/persian_seed_lang.jsonl"));

  return NextResponse.json({
    ok: true,
    project: "ARION ALPHA 1 — Persian phase (v1.1)",
    git: { commit, branch },
    dataset: {
      stats,
      trainCount: stats ? (stats.train_count as number) : countJsonl(path.join(REPO, "datasets/processed/persian_train.jsonl")),
      valCount: stats ? (stats.validation_count as number) : countJsonl(path.join(REPO, "datasets/processed/persian_validation.jsonl")),
    },
    booster: { generated: boosterCount ?? 0, handAuthored: seedCount ?? 0 },
    benchmark: {
      baseline: baselineMetrics,
      arion: arionMetrics,
    },
    training: { summary: trainSummary, seqBenchmark: seqBench },
    artifactFiles,
    kaggle: {
      note: "Kaggle GPU requires phone-verified account; CPU fallback kernels validated the full path.",
    },
  });
}
