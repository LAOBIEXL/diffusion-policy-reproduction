"""Evaluate an official EMA policy and preserve every episode's score."""

import argparse
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import dill
import hydra
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-envs", type=int, help="Override parallelism; keep all configured evaluation seeds")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    # This command reads the author-published checkpoint, including its Hydra config.
    payload = torch.load(args.checkpoint, map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)
    cfg = payload["cfg"]
    OmegaConf.save(cfg, args.output / "checkpoint_config.yaml")
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    random.seed(cfg.training.seed)
    torch.set_num_threads(8)
    policy = hydra.utils.instantiate(cfg.policy)
    state_key = "ema_model" if cfg.training.use_ema else "model"
    policy.load_state_dict(payload["state_dicts"][state_key])
    checkpoint_epoch = dill.loads(payload["pickles"]["epoch"])
    del payload
    policy.eval().to("cuda:0")

    runner_cfg = OmegaConf.create(OmegaConf.to_container(cfg.task.env_runner))
    if args.n_envs is not None:
        runner_cfg.n_envs = args.n_envs
    OmegaConf.save(runner_cfg, args.output / "evaluation_config.yaml")
    runner = hydra.utils.instantiate(runner_cfg, output_dir=str(args.output))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    try:
        metrics = runner.run(policy)
        torch.cuda.synchronize()
    finally:
        runner.env.close()

    result = {key: value._path if isinstance(value, wandb.Video) else float(value) for key, value in metrics.items()}
    result["checkpoint_epoch"] = int(checkpoint_epoch)
    result["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["checkpoint_path"] = str(args.checkpoint)
    result["random_seed"] = int(cfg.training.seed)
    result["parallel_environments"] = len(runner.env_fns)
    result["inference_steps"] = policy.num_inference_steps
    result["evaluation_seconds"] = time.monotonic() - started
    result["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    result["environment"] = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()}
    result["policy_state"] = state_key
    result["test_episode_count"] = sum(key.startswith("test/sim_max_reward_") for key in metrics)
    result["metric_definition"] = "mean across episodes of max_t clip(target_coverage_t / 0.95, 0, 1)"
    (args.output / "eval_log.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if "sim_" not in key}, indent=2))


if __name__ == "__main__":
    main()
