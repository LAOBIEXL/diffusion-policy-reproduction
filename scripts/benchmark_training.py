"""Measure optimizer-step time and GPU memory for the published policy."""

import argparse
import copy
import json
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from training_utils import backward_batch, update_cpu_ema

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--microbatch-size", type=int, default=64)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--ema-device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_num_threads(8)

    cfg = OmegaConf.load(ROOT / "references/official/config.yaml")
    cfg.task.dataset.zarr_path = str(ROOT / cfg.task.dataset.zarr_path)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size)))
    # The original visual policy only reads the first two observation frames.
    batch["obs"] = {key: value[:, :cfg.n_obs_steps] for key, value in batch["obs"].items()}
    model = hydra.utils.instantiate(cfg.policy)
    model.set_normalizer(dataset.get_normalizer())
    ema_model = copy.deepcopy(model).to(args.ema_device)
    ema = hydra.utils.instantiate(cfg.ema, model=ema_model)
    model.to("cuda:0").train()
    # Disable the newer foreach backend so optimizer temporaries do not duplicate all parameters.
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters(), foreach=False)
    batch = dict_apply(batch, lambda x: x.to("cuda:0"))
    torch.cuda.reset_peak_memory_stats()
    times = []
    losses = []
    for step in range(args.steps):
        started = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total_loss = backward_batch(model, batch, args.microbatch_size, args.precision)
        optimizer.step()
        if args.ema_device == "cpu":
            update_cpu_ema(ema, model)
        else:
            ema.step(model)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        times.append(elapsed)
        losses.append(total_loss)
        print(f"step={step} loss={total_loss:.6f} seconds={elapsed:.3f}", flush=True)

    report = {
        "kind": "training_resource_benchmark",
        "settings": {**vars(args), "output": str(args.output)},
        "architecture": OmegaConf.to_container(cfg.policy),
        "optimizer": {**OmegaConf.to_container(cfg.optimizer), "foreach": False},
        "train_windows": len(dataset),
        "optimizer_steps_per_epoch": (len(dataset) + args.batch_size - 1) // args.batch_size,
        "optimizer_step_seconds": times,
        "losses": losses,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "ema_device": args.ema_device,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ["architecture", "optimizer"]}, indent=2))


if __name__ == "__main__":
    main()
