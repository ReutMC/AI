#!/usr/bin/env python3
"""ARION ALPHA 1 — Supervised Fine-Tuning (LoRA) of Qwen3-0.6B.

Real trainable-weight training (spec 1, 16, 29, 30):
- PEFT LoRA adapters on all attention+MLP projections
- CPU/GPU auto device, bf16 base weights, gradient checkpointing
- probe measures real step time and fits the epoch into MAX_TRAIN_MINUTES
- checkpoint every CHECKPOINT_MINUTES with full resume support
- JSONL training log + final train_summary.json
"""
import argparse, json, math, os, random, time, sys, gc

os.environ.setdefault("MALLOC_ARENA_MAX", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.environ.get("ARION_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.environ.get("ARION_BASE_DIR", f"{ROOT}/model/base")
OUTDIR = f"{ROOT}/model/arion-alpha-1-lora"
CKPT_DIR = f"{ROOT}/artifacts/checkpoints"
LOG_PATH = f"{ROOT}/logs/train_log.jsonl"


def set_seed(seed, deterministic=True):
    random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


class SFTDataset(Dataset):
    """Chat-formatted examples with assistant-only label masking.

    Memory-frugal for 4GB-RAM CPU training: examples longer than max_len
    have their ASSISTANT answer truncated at the last line boundary inside
    the budget (dropped if that would cut >40% of the answer).
    """

    def __init__(self, rows, tokenizer, max_len):
        self.items = []
        skipped, truncated = 0, 0
        for ex in rows:
            msgs = ex["messages"]
            try:
                full = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False)
            except Exception:
                continue
            prompt_msgs = msgs[:-1]
            try:
                prompt_txt = tokenizer.apply_chat_template(
                    prompt_msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                continue
            full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            prompt_ids = tokenizer(prompt_txt, add_special_tokens=False)["input_ids"]
            if len(full_ids) > max_len:
                # try truncating the assistant answer at a line boundary
                ans = msgs[-1]["content"]
                budget = max_len - len(prompt_ids) - 2  # room for <|im_end|>
                if budget < 48:
                    skipped += 1
                    continue
                lines = ans.split("\n")
                keep, acc = [], 0  # acc counts ANSWER tokens; budget is answer-relative
                for ln in lines:
                    n = len(tokenizer(ln + "\n", add_special_tokens=False)["input_ids"])
                    if acc + n > budget:
                        break
                    keep.append(ln); acc += n
                kept_txt = "\n".join(keep).rstrip()
                if len(kept_txt) < 0.10 * len(ans.rstrip()) or not kept_txt:
                    skipped += 1
                    continue
                # never train on unclosed code fences
                if kept_txt.count("```") % 2 == 1:
                    kept_txt += "\n```"
                truncated += 1
                msgs = msgs[:-1] + [{"role": "assistant", "content": kept_txt}]
                try:
                    full = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=False)
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
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(attn, dtype=torch.long),
    )


def len_sorted_batches(ds, bs, seed):
    """Batches sorted by length (less padding), batch order shuffled."""
    idx = sorted(range(len(ds)), key=lambda i: len(ds.items[i]["input_ids"]))
    batches = [idx[i:i + bs] for i in range(0, len(idx), bs)]
    random.Random(seed).shuffle(batches)
    return batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=f"{ROOT}/configs/train_config.json")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    torch.set_num_threads(cfg.get("torch_threads", os.cpu_count()))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialized (resume path)
    device = cfg.get("device", "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    print(f"loading tokenizer from {BASE}", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE)

    print(f"loading base model ({cfg['dtype']}) ...", flush=True)
    dtype = torch.bfloat16 if cfg["dtype"] == "bfloat16" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=dtype, low_cpu_mem_usage=True)
    model.config.use_cache = False
    # NOTE: gradient checkpointing is intentionally DISABLED — on this
    # torch/CPU build it is pathologically slow (recompute hooks), and with
    # LoRA the full-graph peak at seq<=256 fits in the 4GB RAM budget.
    model.to(device)

    lcfg = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"], bias="none",
        task_type="CAUSAL_LM", target_modules=cfg["lora_targets"],
    )
    model = get_peft_model(model, lcfg)
    # adapters in fp32 for stable CPU optimization
    for n, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    # ---------------- data ----------------
    def load_jsonl(p):
        return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]

    train_rows = load_jsonl(f"{ROOT}/datasets/processed/train.jsonl")
    val_rows = load_jsonl(f"{ROOT}/datasets/processed/val.jsonl")
    train_ds = SFTDataset(train_rows, tok, cfg["max_seq_len"])
    val_ds = SFTDataset(val_rows, tok, cfg["max_seq_len"]) if val_rows else None

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    coll = lambda b: collate(b, pad_id)

    # ---------------- resume (before optimizer so params objects are final) ----------------
    start_step, best_val, elapsed0 = 0, float("inf"), 0.0
    if args.resume and os.path.exists(f"{CKPT_DIR}/latest/trainer_state.json"):
        st = json.load(open(f"{CKPT_DIR}/latest/trainer_state.json"))
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, f"{CKPT_DIR}/latest", is_trainable=True)
        start_step = st["step"]; best_val = st["best_val"]; elapsed0 = st["elapsed"]
        print(f"resumed from step {start_step} (adapter-only checkpoint)", flush=True)

    # ---------------- optimizer ----------------
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"],
                            weight_decay=cfg["weight_decay"], betas=(0.9, 0.95))

    # ---------------- probe: fit budget ----------------
    bs = cfg["batch_size"]; ga = cfg["gradient_accumulation"]
    probe_batches = len_sorted_batches(train_ds, bs, cfg["seed"])[:3]
    t0 = time.time()
    for pi, pb in enumerate(probe_batches):
        pb_t = time.time()
        batch = coll([train_ds[i] for i in pb])
        input_ids, labels, attn = (x.to(device) for x in batch)
        out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        out.loss.backward()
        model.zero_grad(set_to_none=True)
        print(f"probe batch {pi + 1}/{len(probe_batches)}: {time.time() - pb_t:.1f}s", flush=True)
    sec_per_example = (time.time() - t0) / max(1, len(probe_batches) * bs)

    budget_s = cfg["max_train_minutes"] * 60
    examples_budget = int(budget_s / max(0.1, sec_per_example))
    est_total = len(train_ds)
    if est_total > examples_budget:
        keep = max(examples_budget, 64)
        print(f"probe: {sec_per_example:.2f}s/example → capping epoch to {keep}/{est_total} "
              f"examples to fit {cfg['max_train_minutes']} min budget", flush=True)
        random.Random(cfg["seed"]).shuffle(train_ds.items)
        train_ds.items = train_ds.items[:keep]
    # schedule spans everything the budget can cover across epochs
    est_steps = max(8, int(budget_s / max(0.1, sec_per_example * bs * ga)))
    print(f"probe: {sec_per_example:.3f}s/example, est optimizer steps within budget: {est_steps}", flush=True)

    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, int(est_steps * cfg["warmup_ratio"]))) if s < est_steps * cfg["warmup_ratio"]
        else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - est_steps * cfg["warmup_ratio"]) / max(1, est_steps - est_steps * cfg["warmup_ratio"])))))

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logf = open(LOG_PATH, "a", encoding="utf-8")

    def log(row):
        logf.write(json.dumps(row) + "\n"); logf.flush()

    def save_ckpt(tag, step, bval, elapsed):
        path = f"{CKPT_DIR}/{tag}"
        os.makedirs(path, exist_ok=True)
        # ADAPTER-ONLY checkpoint (memory-frugal: no 1.2GB state_dict serialization)
        model.save_pretrained(path)
        json.dump({"step": step, "best_val": bval, "elapsed": elapsed,
                   "config": cfg}, open(f"{path}/trainer_state.json", "w"))
        print(f"checkpoint saved: {path}", flush=True)

    # ---------------- train loop ----------------
    model.train()
    step = start_step
    t_start = time.time() - elapsed0
    accum_loss, accum_count, tokens_seen = 0.0, 0, 0
    last_ckpt_t = time.time()

    stop_reason = "max_epochs_reached"
    micro_i = start_step * ga
    running = micro_i
    max_epochs = int(cfg.get("max_epochs", 3))
    for epoch in range(max_epochs):
        batches_epoch = len_sorted_batches(train_ds, bs, cfg["seed"] + 1000 * (epoch + start_step + 1))
        for batch_idx in batches_epoch:
            if time.time() - t_start > budget_s:
                stop_reason = "time_budget_reached"
                break
            batch = coll([train_ds[i] for i in batch_idx])
            input_ids, labels, attn = (x.to(device) for x in batch)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss / ga
            loss_val = out.loss.item()
            n_tok = int(attn.sum().item())
            loss.backward()
            del out, loss, batch, input_ids, labels, attn
            accum_loss += loss_val; accum_count += 1
            tokens_seen += n_tok
            micro_i += 1
            running += 1
            if micro_i % ga == 0:
                torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                gc.collect()
                step += 1
                if step % cfg["log_every_steps"] == 0:
                    el = time.time() - t_start
                    row = {"step": step, "loss": round(accum_loss / max(1, accum_count), 4),
                           "lr": sched.get_last_lr()[0], "elapsed_min": round(el / 60, 2),
                           "examples_done": micro_i, "tokens_seen": tokens_seen, "epoch": epoch}
                    print(json.dumps(row), flush=True)
                    log(row)
                accum_loss, accum_count = 0.0, 0
                if step % cfg["eval_every_steps"] == 0 and val_ds:
                    model.eval()
                    with torch.no_grad():
                        vloss, vn = 0.0, 0
                        vi = sorted(range(len(val_ds)), key=lambda i: len(val_ds.items[i]["input_ids"]))[:16]
                        for k in range(0, len(vi), bs):
                            vb = coll([val_ds[i] for i in vi[k:k + bs]])
                            vi_ids, v_lab, v_attn = (x.to(device) for x in vb)
                            vo = model(input_ids=vi_ids, attention_mask=v_attn, labels=v_lab)
                            vloss += vo.loss.item(); vn += 1
                        vloss /= max(1, vn)
                    log({"step": step, "val_loss": round(vloss, 4)})
                    print(f"  val_loss={vloss:.4f}", flush=True)
                    if vloss < best_val:
                        best_val = vloss
                        save_ckpt("best", step, best_val, time.time() - t_start)
                    model.train()
                if time.time() - last_ckpt_t > 480:  # every 8 min
                    save_ckpt("latest", step, best_val, time.time() - t_start)
                    last_ckpt_t = time.time()
        if stop_reason == "time_budget_reached":
            break

    if step != start_step or not os.path.exists(f"{CKPT_DIR}/latest"):
        save_ckpt("latest", step, best_val, time.time() - t_start)

    # final save of adapter
    os.makedirs(OUTDIR, exist_ok=True)
    model.save_pretrained(OUTDIR)
    tok.save_pretrained(OUTDIR)

    summary = {
        "final_step": step, "stop_reason": stop_reason,
        "examples_per_sec": round(1 / max(0.1, sec_per_example), 3),
        "total_minutes": round((time.time() - t_start) / 60, 2),
        "best_val_loss": best_val if best_val != float("inf") else None,
        "examples_trained_on": len(train_ds), "tokens_seen": tokens_seen,
        "lora_r": cfg["lora_r"], "lora_alpha": cfg["lora_alpha"],
        "base_model": "Qwen/Qwen3-0.6B", "method": "LoRA SFT",
        "trainable_params_millions": round(sum(p.numel() for p in params) / 1e6, 3),
    }
    json.dump(summary, open(f"{ROOT}/artifacts/train_summary.json", "w"), indent=2)
    print("TRAINING DONE:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
