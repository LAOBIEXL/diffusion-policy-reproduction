"""Exercise the real dataset, simulator and original visual policy on one batch."""

import argparse
import json
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import hydra
import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/smoke")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.set_num_threads(8)

    cfg = OmegaConf.load(ROOT / "references/official/config.yaml")
    cfg.task.dataset.zarr_path = str(ROOT / cfg.task.dataset.zarr_path)
    OmegaConf.save(cfg, args.output / "source_config.yaml")
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    validation = dataset.get_validation_dataset()
    batch = next(iter(DataLoader(dataset, batch_size=args.batch_size)))
    print(f"Dataset: {dataset.replay_buffer.n_episodes} episodes, {len(dataset)} train windows", flush=True)

    env = PushTImageEnv(legacy=cfg.task.env_runner.legacy_test)
    env.seed(cfg.task.env_runner.test_start_seed)
    observation = env.reset()
    observation, reward, done, info = env.step(np.array([256.0, 256.0]))
    imageio.imwrite(args.output / "environment.png", env.render(mode="rgb_array"))
    env.close()
    print(f"Simulator reset/step/render passed; reward={reward:.6f}", flush=True)

    device = torch.device("cuda:0")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.set_normalizer(dataset.get_normalizer())
    policy.to(device)
    batch = dict_apply(batch, lambda x: x.to(device))
    policy.train()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    loss = policy.compute_loss(batch)
    loss.backward()
    torch.cuda.synchronize()
    backward_seconds = time.monotonic() - started
    gradients_finite = all(torch.isfinite(p.grad).all().item() for p in policy.parameters() if p.grad is not None)
    assert torch.isfinite(loss).item() and gradients_finite
    backward_peak = torch.cuda.max_memory_allocated()
    policy.zero_grad(set_to_none=True)
    policy.eval()
    started = time.monotonic()
    with torch.no_grad():
        actions = policy.predict_action({k: v[:1] for k, v in batch["obs"].items()})
    torch.cuda.synchronize()
    inference_seconds = time.monotonic() - started
    assert actions["action"].shape == (1, 8, 2)
    assert torch.isfinite(actions["action"]).all().item()

    report = {
        "kind": "model_and_simulator_check",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "upstream_commit": subprocess.check_output(["git", "-C", str(ROOT / "third_party/diffusion_policy"), "rev-parse", "HEAD"], text=True).strip(),
        "dataset_episodes": dataset.replay_buffer.n_episodes,
        "train_episodes": int(dataset.train_mask.sum()),
        "validation_episodes": int(validation.train_mask.sum()),
        "train_windows": len(dataset),
        "validation_windows": len(validation),
        "batch_size": args.batch_size,
        "batch_shapes": {"action": list(batch["action"].shape), **{k: list(v.shape) for k, v in batch["obs"].items()}},
        "diffusion_parameters": sum(p.numel() for p in policy.model.parameters()),
        "vision_parameters": sum(p.numel() for p in policy.obs_encoder.parameters()),
        "loss": loss.item(),
        "gradients_finite": gradients_finite,
        "forward_backward_seconds": backward_seconds,
        "forward_backward_peak_gpu_mib": backward_peak / 2**20,
        "inference_steps": policy.num_inference_steps,
        "inference_seconds": inference_seconds,
        "action_shape": list(actions["action"].shape),
        "simulator_step_reward": float(reward),
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
