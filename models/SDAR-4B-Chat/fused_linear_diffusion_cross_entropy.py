import torch.nn as nn


class FusedLinearDiffusionCrossEntropyLoss(nn.Module):
    """Stub — fused kernel not available; SFT trainer uses its own loss."""
    def forward(self, *args, **kwargs):
        raise NotImplementedError("fused_linear_diffusion_cross_entropy not installed")
