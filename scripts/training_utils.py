"""Memory adaptations for the published Push-T policy; no model changes."""

import torch

from diffusion_policy.common.pytorch_util import dict_apply


@torch.no_grad()
def update_cpu_ema(ema, model):
    """Apply the upstream EMA equation to FP32 weights kept in host memory.

    This visual policy replaces all BatchNorm modules with GroupNorm. Its
    frozen parameters (including normalization statistics) are copied directly.
    """
    ema.decay = ema.get_decay(ema.optimization_step)
    for parameter, averaged in zip(model.parameters(), ema.averaged_model.parameters()):
        source = parameter.detach().to(device="cpu", dtype=averaged.dtype)
        if parameter.requires_grad:
            averaged.mul_(ema.decay).add_(source, alpha=1 - ema.decay)
        else:
            averaged.copy_(source)
    ema.optimization_step += 1


def backward_batch(model, batch, microbatch_size, precision):
    """Accumulate the sample-weighted mean loss for one optimizer batch."""
    batch_size = len(batch["action"])
    total_loss = 0.0
    for start in range(0, batch_size, microbatch_size):
        microbatch = dict_apply(batch, lambda x: x[start:start + microbatch_size])
        fraction = len(microbatch["action"]) / batch_size
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            loss = model.compute_loss(microbatch)
        (loss * fraction).backward()
        total_loss += loss.item() * fraction
    return total_loss
