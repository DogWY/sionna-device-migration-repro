"""User-facing repro modules built around Sionna components."""

from __future__ import annotations

import torch
from torch import nn
from sionna.phy.channel import AWGN
from sionna.phy.utils import ebnodb2no


def power_normalize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    power = torch.mean(x**2, dim=(1, 2), keepdim=True)
    return x / torch.sqrt(power), power


class AWGNChannel(nn.Module):
    """Wrapper that reproduces stale Sionna AWGN device state after `.to(device)`.

    This mirrors the user-side model shape: real-valued pairs are converted to
    complex values, passed through Sionna's `AWGN`, then converted back.
    """

    def __init__(self, snr: float | None = None, device: str | None = None):
        super().__init__()
        self.snr = snr
        self.awgn = AWGN(device=device)

    def forward(self, inputs: torch.Tensor, snr: float | None = None) -> torch.Tensor:
        if snr is None and self.snr is None:
            raise ValueError(
                "SNR must be provided either during initialization or as an argument to the forward method."
            )
        if snr is not None:
            self.snr = snr

        batch, length, dim = inputs.shape
        if dim % 2 != 0:
            raise ValueError("The last dimension of inputs must be even.")

        x, pwr = power_normalize(inputs)
        x = x.reshape(batch, length, -1, 2)
        x_complex = x[..., 0] + 1j * x[..., 1]
        noise_std = ebnodb2no(self.snr, 1, 1)
        y_complex = self.awgn(x_complex, noise_std)
        y = torch.stack((y_complex.real, y_complex.imag), dim=-1)
        return y.reshape(batch, length, dim) * torch.sqrt(pwr)
