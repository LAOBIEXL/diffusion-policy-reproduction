"""Check gradient accumulation and CPU EMA against their reference operations."""

import copy
import json
from pathlib import Path

import torch
from torch import nn

from diffusion_policy.model.diffusion.ema_model import EMAModel
from training_utils import backward_batch, update_cpu_ema

ROOT = Path(__file__).resolve().parents[1]


class RegressionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(4, 16), nn.GroupNorm(4, 16), nn.Linear(16, 2))

    def compute_loss(self, batch):
        return nn.functional.mse_loss(self.network(batch["obs"]), batch["action"])


def main():
    torch.manual_seed(42)
    model = RegressionPolicy().cuda()
    accumulated = copy.deepcopy(model)
    batch = {"obs": torch.randn(5, 4, device="cuda"), "action": torch.randn(5, 2, device="cuda")}
    loss = model.compute_loss(batch)
    loss.backward()
    accumulated_loss = backward_batch(accumulated, batch, microbatch_size=2, precision="fp32")
    gradient_error = max((a.grad - b.grad).abs().max().item() for a, b in zip(model.parameters(), accumulated.parameters()))
    for actual, expected in zip(accumulated.parameters(), model.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-6, rtol=1e-5)

    # Include frozen parameters to check that they are copied, not averaged.
    model.network[-1].bias.requires_grad_(False)
    cpu_ema = EMAModel(copy.deepcopy(model).cpu(), power=0.75)
    reference = EMAModel(copy.deepcopy(model), power=0.75)
    ema_error = 0.0
    for step in range(8):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.01 * (step + 1))
        update_cpu_ema(cpu_ema, model)
        reference.step(model)
        for actual, expected in zip(cpu_ema.averaged_model.parameters(), reference.averaged_model.parameters()):
            torch.testing.assert_close(actual, expected.cpu(), atol=1e-6, rtol=1e-5)
            ema_error = max(ema_error, (actual - expected.cpu()).abs().max().item())
    assert cpu_ema.optimization_step == reference.optimization_step
    assert cpu_ema.decay == reference.decay
    result = {
        "gradient_accumulation": "passed, including a partial final microbatch",
        "full_batch_loss": loss.item(),
        "accumulated_loss": accumulated_loss,
        "gradient_max_absolute_error": gradient_error,
        "cpu_ema": "passed against upstream GPU EMA for 8 updates",
        "ema_max_absolute_error": ema_error,
        "ema_optimization_step": cpu_ema.optimization_step,
        "ema_decay": cpu_ema.decay,
    }
    output = ROOT / "records/validation/training_math.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
