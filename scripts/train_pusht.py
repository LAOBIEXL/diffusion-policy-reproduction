"""Run the author's training workspace with explicitly recorded local overrides."""

import argparse
import json
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="Run three optimizer steps and two short rollouts")
    args = parser.parse_args()
    args.output = args.output.resolve()
    # Create a separate output directory for each run.
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)

    cfg = OmegaConf.load(ROOT / "references/official/config.yaml")
    cfg.task.dataset.zarr_path = str(ROOT / cfg.task.dataset.zarr_path)
    cfg.training.seed = args.seed
    cfg.training.resume = False
    cfg.optimizer.foreach = False
    cfg.dataloader.num_workers = args.workers
    cfg.val_dataloader.num_workers = args.workers
    cfg.logging.mode = "disabled"
    # Retain latest and best checkpoints, instead of five full 4 GB checkpoints.
    cfg.checkpoint.topk.k = 1
    if args.smoke:
        cfg.training.num_epochs = 1
        cfg.training.max_train_steps = 3
        cfg.training.max_val_steps = 2
        cfg.task.env_runner.n_train = 0
        cfg.task.env_runner.n_train_vis = 0
        cfg.task.env_runner.n_test = 2
        cfg.task.env_runner.n_test_vis = 1
        cfg.task.env_runner.n_envs = 2
        cfg.task.env_runner.max_steps = 16
        cfg.checkpoint.topk.k = 0

    OmegaConf.save(cfg, args.output / "config.yaml")
    manifest = {
        "kind": "training_pipeline_smoke_test" if args.smoke else "from_scratch_training",
        "training_seed": args.seed,
        "precision": "fp32",
        "gradient_accumulation": 1,
        "ema_device": "cuda",
        "optimizer_foreach": False,
        "dataloader_workers": args.workers,
        "initialization": "random",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    workspace_type = hydra.utils.get_class(cfg._target_)
    workspace = workspace_type(cfg, output_dir=str(args.output))
    workspace.run()
    if workspace._saving_thread is not None:
        workspace._saving_thread.join()
    print(f"Finished {manifest['kind']}: {args.output}", flush=True)


if __name__ == "__main__":
    main()
