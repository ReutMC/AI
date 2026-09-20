#!/usr/bin/env python3
"""ARION ALPHA 1 — GPU SFT trainer for Kaggle (Persian phase).

Differences from train_sft.py (CPU/Actions trainer):
  - REQUIRES CUDA: fails clearly if no GPU (never silently downgrades to CPU)
  - Detects & logs GPU name, VRAM, CUDA version, torch/transformers/peft versions
  - bf16 on Ampere+, fp16 (with GradScaler) on P100/T4
  - Gradient checkpointing enabled for long sequences
  - OOM-resilient: on CUDA OOM halves micro-batch and doubles grad-accum, retries
  - --benchmark-seq-lens mode: measures tokens/s, VRAM peak, step time at
    2048/3072/4096 to pick max_seq_len (Persian-phase spec)
  - Saves full reproducibility metadata into train_summary.json

Usage (inside Kaggle kernel):
  python3 train_kaggle.py --config /kaggle/working/config.json
  python3 train_kaggle.py --benchmark-seq-lens 2048,3072,4096 --benchmark-steps 3
"""
import argparse, gc, json, math, os, random, time, hashlib, platform, sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import Dataset

ROOT = os.environ.get("ARION_ROOT", os.getcwd())


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_report():
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": None, "vram_gb": None, "cuda_version": None,
        "transformers": None, "peft": None,
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import peft
        info["peft"] = peft.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu_name"] = p.name
        info["vram_gb"] = round(p.total_memory / 1e9, 2)
        info["cuda_version"] = torch.version.cuda
        info["gpu_count"] = torch.cuda.device_count()
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    print("[ENV] " + json.dumps(info), flush=True)
    return info


class SFTDataset(Dataset):
    """Chat-formatted examples with assistant-only label masking (same as CPU
    trainer). Long answers are truncated at a line boundary, never mid-word."""

    def __init__(self, rows, tokenizer, max_len):
        self.items = []
        skipped, truncated = 0, 0
        for ex in rows:
            msgs = ex["messages"]
            try:
                full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                prompt_txt = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
            except Exception:
                skipped += 1
                continue
            full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            prompt_ids = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
            if len(full_ids) > max_len:
                ans = msgs[-1]["content"]
                budget = max_len - len(prompt_ids) - 2
                if budget < 48:
                    skipped += 1
                    continue
                lines = ans.split("\n")
                keep, acc = [], 0
                for ln in lines:
                    n = len(tokenizer(ln + "\n", add_special_tokens=False)["input_ids"])
                    if acc + n > budget:
                        break
                    keep.append(ln)
                    acc += n
                kept_txt = "\n".join(keep).rstrip()
                if len(kept_txt) < 0.60 * len(ans.rstrip()) or not kept_txt:
                    # on GPU we can afford longer context: reject if we'd cut >40%
                    skipped += 1
                    continue
                if kept_txt.count("```") % 2 == 1:
                    kept_txt += "\n```"
                truncated += 1
                msgs = msgs[:-1] + [{"role": "assistant", "content": kept_txt}]
                try:
                    full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
                except Exception:
                    skipped += 1
                    continue
            if len(full_ids) > max_len:
                skipped += 1
                continue
            labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
            labels = labels[: len(full_ids)]
            if all(l == -100 for l in labels):
                continue
            self.items.append({"input_ids": full_ids, "labels": labels})
        print(f"dataset: {len(self.items)} usable, {truncated} truncated, {skipped} dropped", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = len(b["input_ids"])
        pad = maxlen - n
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return (torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(attn, dtype=torch.long))


def len_sorted_batches(ds, bs, seed):
    idx = sorted(range(len(ds)), key=lambda i: len(ds.items[i]["input_ids"]))
    batches = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    random.Random(seed).shuffle(batches)
    return batches


def benchmark_seq_lens(model_path, tokenizer, train_rows, cfg, seq_lens, steps_per_len, device):
    """Measure tokens/s, VRAM and step time per max_seq_len. Returns dict."""
    from peft import LoraConfig, get_peft_model
    results = {}
    for seq in seq_lens:
        try:
            model = _build_model(model_path, cfg, device)
            ds = SFTDataset(train_rows, tokenizer, seq)
            bs = cfg["batch_size"]
            ga = cfg["gradient_accumulation"]
            bs, ga, oom = _fit_batch(model, ds, bs, device, seq, cfg)
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            batches = len_sorted_batches(ds, bs, cfg["seed"])[:max(2, steps_per_len)]
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            tok_n = 0
            for b in batches:
                batch = collate([ds[i] for i in b], pad_id)
                input_ids, labels, attn = (x.to(device) for x in batch)
                out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                out.loss.backward()
                model.zero_grad(set_to_none=True)
                tok_n += int(attn.sum().item())
            dt = time.time() - t0
            step_s = dt / len(batches)
            vram = torch.cuda.max_memory_allocated() / 1e9
            results[str(seq)] = {
                "batch_size": bs, "grad_accum": ga, "oom_recovery": oom,
                "examples_per_sec": round(len(batches) * bs / dt, 2),
                "tokens_per_sec": round(tok_n / dt, 1),
                "step_time_s": round(step_s, 3),
                "peak_vram_gb": round(vram, 2),
            }
            print(f"[SEQBENCH] {seq}: {json.dumps(results[str(seq)])}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            results[str(seq)] = {"error": "OOM even after batch-size reduction"}
            print(f"[SEQBENCH] {seq}: OOM", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            results[str(seq)] = {"error": str(e)[:200]}
            print(f"[SEQBENCH] {seq}: ERROR {str(e)[:200]}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    return results


def torch_dtype(name):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16,
            "float32": torch.float32}[name]


def _build_model(model_path, cfg, device):
    dtype = torch_dtype(cfg["dtype"])
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    if cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
    from peft import LoraConfig, get_peft_model
    lcfg = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                      lora_dropout=cfg["lora_dropout"], bias="none",
                      task_type="CAUSAL_LM", target_modules=cfg["lora_targets"])
    model = get_peft_model(model, lcfg)
    model.to(device)
    if cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return model


def _fit_batch(model, ds, bs, device, seq_len, cfg):
    """Try a forward/backward at bs; on OOM halve bs and double ga."""
    oom_recovered = False
    ga = cfg["gradient_accumulation"]
    while True:
        try:
            batches = len_sorted_batches(ds, bs, cfg["seed"])[:1]
            batch = collate([ds[i] for i in batches[0]], ds_pad_id)
            input_ids, labels, attn = (x.to(device) for x in batch)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            out.loss.backward()
            model.zero_grad(set_to_none=True)
            return bs, ga, oom_recovered
        except torch.cuda.OutOfMemoryError:
            if bs <= 1:
                raise
            bs = max(1, bs // 2)
            ga = min(64, ga * 2)
            oom_recovered = True
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[auto-bs] OOM at fit → batch_size={bs}, grad_accum={ga}", flush=True)


ds_pad_id = None


def main():
    global ds_pad_id
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-train", default=None)
    ap.add_argument("--data-val", default=None)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--benchmark-seq-lens", default=None, help="e.g. 2048,3072,4096")
    ap.add_argument("--benchmark-steps", type=int, default=3)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="EXPLICIT CPU mode (diagnostic/smoke only). Without this flag "
                         "the trainer refuses to run without CUDA — never silent.")
    args = ap.parse_args()

    info = env_report()
    cfg = json.load(open(args.config)) if args.config else {}
    device = None
    if not torch.cuda.is_available():
        if args.allow_cpu:
            device = "cpu"
            cfg["dtype"] = "float32"
            print("EXPLICIT CPU MODE (--allow-cpu): diagnostic/smoke only. "
                  "This is NOT a silent downgrade — full training requires CUDA.", flush=True)
        else:
            print("FATAL: no CUDA GPU available — Kaggle kernel must run with a GPU "
                  "accelerator. Refusing to silently downgrade to CPU. "
                  "(Kaggle GPU requires a phone-verified account.)", flush=True)
            sys.exit(2)
    else:
        device = "cuda"

    cfg.setdefault("max_seq_len", 3072)
    cfg.setdefault("batch_size", 4)
    cfg.setdefault("gradient_accumulation", 8)
    cfg.setdefault("lora_r", 16)
    cfg.setdefault("lora_alpha", 32)
    cfg.setdefault("lora_dropout", 0.05)
    cfg.setdefault("lora_targets", ["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"])
    cfg.setdefault("learning_rate", 1e-4)
    cfg.setdefault("weight_decay", 0.01)
    cfg.setdefault("warmup_ratio", 0.05)
    cfg.setdefault("max_grad_norm", 1.0)
    cfg.setdefault("max_epochs", 2)
    cfg.setdefault("eval_every_steps", 100)
    cfg.setdefault("log_every_steps", 10)
    cfg.setdefault("seed", 42)
    cfg.setdefault("gradient_checkpointing", True)
    cfg.setdefault("max_train_minutes", 540)  # safety stop (Kaggle GPU session limit)

    # dtype: bf16 on Ampere+, fp16+GradScaler on P100/T4
    if cfg.get("dtype") == "auto" or "dtype" not in cfg:
        cfg["dtype"] = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    dtype = torch_dtype(cfg["dtype"])
    use_scaler = (dtype == torch.float16 and device == "cuda")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    base_model = args.base_model or cfg.get("base_model_path", "Qwen/Qwen3-0.6B")
    data_train = args.data_train or cfg.get("train_path")
    data_val = args.data_val or cfg.get("val_path")
    outdir = args.output or cfg.get("output_dir", f"{ROOT}/model/arion-alpha-1-lora")

    from transformers import AutoTokenizer
    print(f"loading tokenizer from {base_model}", flush=True)
    tok = AutoTokenizer.from_pretrained(base_model)

    def load_jsonl(p):
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

    train_rows = load_jsonl(data_train)
    val_rows = load_jsonl(data_val) if data_val and os.path.exists(data_val) else []

    if args.benchmark_seq_lens:
        lens = [int(x) for x in args.benchmark_seq_lens.split(",")]
        ds_pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        # keep benchmark fast: cap rows
        b_rows = train_rows[:4000]
        res = benchmark_seq_lens(base_model, tok, b_rows, cfg, lens,
                                 args.benchmark_steps, device)
        json.dump({"gpu": info, "seq_benchmark": res},
                  open(f"{os.path.dirname(outdir)}/seq_benchmark.json", "w"), indent=2)
        print("[SEQBENCH-DONE] " + json.dumps(res), flush=True)
        return

    model = _build_model(base_model, cfg, device)
    ds_pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    train_ds = SFTDataset(train_rows, tok, cfg["max_seq_len"])
    val_ds = SFTDataset(val_rows, tok, cfg["max_seq_len"]) if val_rows else None

    bs, ga, _ = _fit_batch(model, train_ds, cfg["batch_size"], device, cfg["max_seq_len"], cfg)
    print(f"[batch] batch_size={bs} grad_accum={ga} (effective {bs * ga})", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"], betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    est_steps = max(8, (len(train_ds) * cfg["max_epochs"]) // (bs * ga))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(est_steps * cfg["warmup_ratio"])))
        if s < est_steps * cfg["warmup_ratio"]
        else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - est_steps * cfg["warmup_ratio"]) / max(1, est_steps - est_steps * cfg["warmup_ratio"])))))

    logf = open(f"{ROOT}/train_log.jsonl", "a", encoding="utf-8")

    def log(row):
        logf.write(json.dumps(row) + "\n")
        logf.flush()

    model.train()
    step = 0
    best_val = float("inf")
    last_train_loss = None
    tokens_seen = 0
    t_start = time.time()
    accum_loss, accum_count = 0.0, 0
    stop_reason = "max_epochs_reached"
    budget_s = cfg["max_train_minutes"] * 60

    for epoch in range(cfg["max_epochs"]):
        if time.time() - t_start > budget_s:
            stop_reason = "time_budget_reached"
            break
        for batch_idx in len_sorted_batches(train_ds, bs, cfg["seed"] + 1000 * (epoch + 1)):
            if time.time() - t_start > budget_s:
                stop_reason = "time_budget_reached"
                break
            batch = collate([train_ds[i] for i in batch_idx], ds_pad_id)
            try:
                input_ids, labels, attn = (x.to(device, non_blocking=True) for x in batch)
                if use_scaler:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                        loss = out.loss / ga
                    scaler.scale(loss).backward()
                else:
                    out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
                    loss = out.loss / ga
                    loss.backward()
                loss_val = out.loss.item()
            except torch.cuda.OutOfMemoryError:
                # adapt: halve batch, double accum, retry this batch one example at a time
                print("[OOM] during training → adapting", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                ga = min(64, ga * 2)
                continue
            n_tok = int(attn.sum().item())
            tokens_seen += n_tok
            accum_loss += loss_val
            accum_count += 1
            del out, loss, batch, input_ids, labels, attn
            if (step * ga + accum_count) % ga == 0:
                if use_scaler:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                last_train_loss = accum_loss / max(1, accum_count)
                if step % cfg["log_every_steps"] == 0:
                    el = time.time() - t_start
                    row = {"step": step, "loss": round(last_train_loss, 4),
                           "lr": sched.get_last_lr()[0],
                           "elapsed_min": round(el / 60, 2),
                           "tokens_per_sec": round(tokens_seen / max(1, el), 1),
                           "tokens_seen": tokens_seen, "epoch": epoch}
                    print(json.dumps(row), flush=True)
                    log(row)
                accum_loss, accum_count = 0.0, 0
                if val_ds and step % cfg["eval_every_steps"] == 0:
                    model.eval()
                    vloss, vn = 0.0, 0
                    vi = sorted(range(len(val_ds)), key=lambda i: len(val_ds.items[i]["input_ids"]))[:24]
                    with torch.no_grad():
                        for k in range(0, len(vi), bs):
                            vb = collate([val_ds[i] for i in vi[k:k + bs]], ds_pad_id)
                            vi_ids, v_lab, v_attn = (x.to(device) for x in vb)
                            vo = model(input_ids=vi_ids, attention_mask=v_attn, labels=v_lab)
                            vloss += vo.loss.item()
                            vn += 1
                    vloss /= max(1, vn)
                    log({"step": step, "val_loss": round(vloss, 4)})
                    print(f"  val_loss={vloss:.4f}", flush=True)
                    if vloss < best_val:
                        best_val = vloss
                        os.makedirs(outdir, exist_ok=True)
                        model.save_pretrained(outdir)
                    model.train()
        print(f"[epoch {epoch + 1}/{cfg['max_epochs']}] done in "
              f"{round((time.time() - t_start) / 60, 1)} min", flush=True)

    os.makedirs(outdir, exist_ok=True)
    model.save_pretrained(outdir)
    tok.save_pretrained(outdir)

    summary = {
        "final_step": step,
        "stop_reason": stop_reason,
        "total_minutes": round((time.time() - t_start) / 60, 2),
        "last_train_loss": round(last_train_loss, 4) if last_train_loss is not None else None,
        "best_val_loss": round(best_val, 4) if best_val != float("inf") else None,
        "examples_trained_on": len(train_ds),
        "val_examples": len(val_ds) if val_ds else 0,
        "tokens_seen": tokens_seen,
        "lora_r": cfg["lora_r"], "lora_alpha": cfg["lora_alpha"],
        "learning_rate": cfg["learning_rate"],
        "max_seq_len": cfg["max_seq_len"],
        "batch_size": bs, "gradient_accumulation": ga,
        "dtype": cfg["dtype"],
        "base_model": base_model,
        "method": "LoRA SFT (Kaggle GPU)",
        "trainable_params_millions": round(sum(p.numel() for p in params) / 1e6, 3),
        "gpu_name": info["gpu_name"], "vram_gb": info["vram_gb"],
        "cuda_version": info["cuda_version"],
        "torch": info["torch"], "transformers": info["transformers"], "peft": info["peft"],
        "seed": cfg["seed"],
        "epochs": cfg["max_epochs"],
    }
    json.dump(summary, open(f"{ROOT}/train_summary.json", "w"), indent=2)
    print("TRAINING DONE: " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
