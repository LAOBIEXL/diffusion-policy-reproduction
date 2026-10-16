"""Export the official EMA checkpoint and its matching evaluation record."""

import argparse
import hashlib
import json
from pathlib import Path

import dill
import torch

from compare_results import reference_at_epoch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SHA256 = "bac7221f7e34cd51162dc1972e1a39ffcddc87de1dc1780c44ffa61b88c4ff76"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "data/checkpoints/official_train_0_epoch0500.ckpt")
    parser.add_argument("--reference", type=Path, default=ROOT / "data/reference/train_0.logs.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/docker")
    args = parser.parse_args()
    source_hash = sha256(args.checkpoint)
    if source_hash != OFFICIAL_SHA256:
        raise ValueError("Checkpoint SHA256 does not match the official epoch=0500 download")
    source = torch.load(args.checkpoint, map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)
    compact = {
        "cfg": source["cfg"],
        "state_dicts": {"ema_model": source["state_dicts"]["ema_model"]},
        "pickles": {"epoch": source["pickles"]["epoch"]},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output / "ema.ckpt"
    torch.save(compact, checkpoint, pickle_module=dill)
    restored = torch.load(checkpoint, map_location="cpu", pickle_module=dill, weights_only=False, mmap=True)
    original_state = compact["state_dicts"]["ema_model"]
    exported_state = restored["state_dicts"]["ema_model"]
    assert original_state.keys() == exported_state.keys()
    for name, value in original_state.items():
        exported = exported_state[name]
        assert value.dtype == exported.dtype and torch.equal(value, exported), name
    epoch = int(dill.loads(source["pickles"]["epoch"]))
    reference = reference_at_epoch(args.reference, epoch)
    reference_path = args.output / "reference.jsonl"
    reference_path.write_text(json.dumps(reference) + "\n", encoding="utf-8")
    manifest = {
        "source_checkpoint_sha256": source_hash,
        "checkpoint_epoch": epoch,
        "policy_state": "ema_model",
        "exported_checkpoint_sha256": sha256(checkpoint),
        "exported_checkpoint_bytes": checkpoint.stat().st_size,
        "state_dict_entries": len(original_state),
        "tensor_elements": sum(value.numel() for value in original_state.values()),
        "all_tensors_equal": True,
        "reference_sha256": sha256(reference_path),
        "official_mean_score": reference["test/mean_score"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
