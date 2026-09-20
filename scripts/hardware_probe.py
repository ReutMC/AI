#!/usr/bin/env python3
"""ARION ALPHA 1 — hardware inspection & adaptive training configuration.

Detects CPU/RAM/GPU/disk and derives safe training hyperparameters so the
run never crashes merely because defaults exceed the machine.
"""
import json, os, shutil

def inspect():
    info = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["cpu_cores"] = psutil.cpu_count(logical=True)
        info["cpu_physical"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = round(vm.total / 1e9, 2)
        info["ram_available_gb"] = round(vm.available / 1e9, 2)
    except ImportError:
        info["cpu_cores"] = os.cpu_count()
        with open("/proc/meminfo") as f:
            total = [l for l in f if l.startswith("MemTotal")][0]
        info["ram_total_gb"] = round(int(total.split()[1]) / 1e6, 2)
        info["ram_available_gb"] = info["ram_total_gb"]
    du = shutil.disk_usage("/")
    info["disk_total_gb"] = round(du.total / 1e9, 2)
    info["disk_free_gb"] = round(du.free / 1e9, 2)
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if torch.cuda.is_available() else 0
        info["bf16_support"] = torch.backends.cpu.get_cpu_capability() if hasattr(torch.backends.cpu, "get_cpu_capability") else "unknown"
    except Exception as e:
        info["torch"] = f"unavailable: {e}"
        info["cuda_available"] = False
    return info

def derive_config(hw, overrides=None):
    """Choose training hyperparameters that fit the machine."""
    ram = hw.get("ram_available_gb", 4)
    cores = max(1, hw.get("cpu_cores", 2))
    gpu = hw.get("cuda_available", False)

    cfg = {
        "device": "cuda" if gpu else "cpu",
        "torch_threads": cores,
        # RAM-adapted context: 4GB machines must stay at/below 256 tokens
        # (verified empirically: seq 256 ≈ 3.25GB peak with LoRA; 448 OOMs)
        "max_seq_len": 768 if (gpu or ram >= 12) else (384 if ram >= 8 else 256),
        "batch_size": 2 if (gpu or ram >= 8) else 1,
        "gradient_accumulation": 4 if (gpu or ram >= 8) else 8,
        "dtype": "bfloat16",
        "use_gradient_checkpointing": False,
        # LoRA
        "lora_r": 16 if (gpu or ram >= 8) else 12,
        "lora_alpha": 32 if (gpu or ram >= 8) else 24,
        "lora_dropout": 0.05,
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # optimization
        "learning_rate": 1.5e-4 if (gpu or ram >= 8) else 1.2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "lr_scheduler": "cosine",
        "max_grad_norm": 1.0,
        # time budget (minutes) — the key CPU adaptation knob
        "max_train_minutes": int(os.environ.get("MAX_TRAIN_MINUTES", "150")),
        "max_epochs": 3,
        "eval_every_steps": 40,
        "log_every_steps": 5,
        "seed": 42,
        "deterministic": True,
    }
    if overrides:
        cfg.update(overrides)
    return cfg

if __name__ == "__main__":
    hw = inspect()
    cfg = derive_config(hw)
    out = {
        "hardware": hw,
        "derived_training_config": cfg,
        "notes": "Config auto-derived so the run fits this machine (spec section 29: Hardware Adaptation).",
    }
    os.makedirs("/home/z/my-project/arion-alpha-1/artifacts", exist_ok=True)
    with open("/home/z/my-project/arion-alpha-1/artifacts/hardware_report.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
