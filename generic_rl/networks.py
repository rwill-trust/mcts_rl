from __future__ import annotations
from typing import Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover - torch optional
    torch = None
    nn = None
    F = None


if torch is not None:
    class ResidualBlock(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.b1 = nn.BatchNorm2d(ch)
            self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.b2 = nn.BatchNorm2d(ch)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = F.relu(self.b1(self.c1(x)))
            y = self.b2(self.c2(y))
            return F.relu(x + y)


    class AlphaZeroNet(nn.Module):
        """Simple residual network used by the generic trainer."""

        def __init__(self, input_shape: Tuple[int, int, int], action_size: int,
                     ch: int = 128, blocks: int = 10):
            super().__init__()
            c, h, w = input_shape
            self.stem = nn.Sequential(
                nn.Conv2d(c, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch), nn.ReLU(),
            )
            self.res = nn.Sequential(*[ResidualBlock(ch) for _ in range(blocks)])

            self.policy_conv = nn.Conv2d(ch, 2, kernel_size=1)
            self.policy_bn = nn.BatchNorm2d(2)
            self.policy_flatten = nn.Flatten()
            self.policy_linear = nn.Linear(2 * h * w, action_size)

            self.value_conv = nn.Conv2d(ch, 1, kernel_size=1)
            self.value_bn = nn.BatchNorm2d(1)
            self.value_flatten = nn.Flatten()
            self.value_linear1 = nn.Linear(1 * h * w, 64)
            self.value_linear2 = nn.Linear(64, 1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.stem(x)
            x = self.res(x)
            p = F.relu(self.policy_bn(self.policy_conv(x)))
            logits = self.policy_linear(self.policy_flatten(p))
            v = F.relu(self.value_bn(self.value_conv(x)))
            v = F.relu(self.value_linear1(self.value_flatten(v)))
            value = torch.tanh(self.value_linear2(v))
            return logits, value.squeeze(1)
else:  # pragma: no cover - torch missing
    class ResidualBlock:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available")

    class AlphaZeroNet:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch not available")
