"""Shared 3D building blocks.

Transcribed from panels (b) and (c) of
``docs/flowchart/phase1_encoder_decoder.drawio``, with the spatial sizes adapted
to this project's 64^3 volumes. Both stages use them: Stage A's U-Net and Stage
B's relational encoder/decoder are built from the same primitives, which is why
the activation is a parameter (Stage A uses ReLU, Stage B leaky ReLU).

Conventions kept from the reference figure:

* ``InstanceNorm3d(affine=False)`` everywhere (``IN``); the only affine
  normalisation is the ``LayerNorm`` inside the prompt decoder;
* every convolution is ``k=3, s=1, p=1, bias=False`` unless it is a stride-2
  transition;
* upsampling is trilinear with ``align_corners=True`` - no transposed
  convolutions;
* skip concatenation is the only path by which encoder features re-enter the
  decoder.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def conv3d(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> nn.Conv3d:
    """The project's standard 3D convolution: no bias, padding preserves size."""
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        bias=False,
    )


def norm3d(channels: int) -> nn.InstanceNorm3d:
    """``InstanceNorm3d(affine=False)`` - no learned gamma/beta, as in the figure."""
    return nn.InstanceNorm3d(channels, affine=False)


def prior_logit(foreground_fraction: float) -> float:
    """``log(p / (1 - p))`` - the logit that makes sigmoid output ``p`` at init.

    Both stages' mask heads start here rather than at zero. A zero-initialised
    head predicts p = 0.5 for every voxel, but a single structure covers about
    0.16% of a 64^3 volume, so training then begins by pushing a quarter of a
    million background logits down before either Dice term carries usable
    gradient.
    """
    if not 0.0 < foreground_fraction < 1.0:
        raise ValueError(f"foreground fraction must be in (0, 1), got {foreground_fraction}")
    return math.log(foreground_fraction / (1.0 - foreground_fraction))


def make_activation(name: str = "relu") -> nn.Module:
    """The two activations the configs allow: ``relu`` (Stage A) or ``leaky_relu``."""
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    raise ValueError(f"activation must be 'relu' or 'leaky_relu', got {name!r}")


class ConvINReLU(nn.Module):
    """``Conv3d -> InstanceNorm3d -> activation`` (ReLU unless told otherwise)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.conv = conv3d(in_channels, out_channels, kernel_size, stride)
        self.norm = norm3d(out_channels)
        self.act = make_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    """Post-activation residual block with an identity shortcut (panel b).

    ``Conv-IN-ReLU-Conv-IN`` plus the input, then ReLU. Spatial size and channel
    count are unchanged, so the shortcut needs no projection.
    """

    def __init__(self, channels: int, kernel_size: int = 3, activation: str = "relu") -> None:
        super().__init__()
        self.conv1 = conv3d(channels, channels, kernel_size)
        self.norm1 = norm3d(channels)
        self.conv2 = conv3d(channels, channels, kernel_size)
        self.norm2 = norm3d(channels)
        self.act = make_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class EncoderStage(nn.Module):
    """``Conv3d(stride) -> IN -> ReLU -> ResidualBlock`` (panel c).

    The stem uses ``stride=1``; every later stage uses ``stride=2`` and halves
    each spatial dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_blocks: int = 1,
        kernel_size: int = 3,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.transition = ConvINReLU(in_channels, out_channels, kernel_size, stride, activation)
        self.blocks = nn.Sequential(
            *(ResidualBlock(out_channels, kernel_size, activation) for _ in range(num_blocks))
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(self.transition(x))


class UpBlock(nn.Module):
    """Trilinear upsample, skip concatenation, then ``Conv-IN-ReLU`` (panel c).

    ``Decode i`` maps ``in_channels + skip_channels -> out_channels``. There is
    no transposed convolution, so there are no checkerboard artefacts to fight.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        align_corners: bool = True,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.fuse = ConvINReLU(
            in_channels + skip_channels, out_channels, kernel_size, activation=activation
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        upsampled = F.interpolate(
            x, size=skip.shape[2:], mode="trilinear", align_corners=self.align_corners
        )
        return self.fuse(torch.cat([upsampled, skip], dim=1))


class DecomposedPositionalEncoding3D(nn.Module):
    """Learned positional encoding factorised over the three spatial axes.

    Holds ``pos_z [1, D, 1, 1, C]``, ``pos_y [1, 1, H, 1, C]`` and
    ``pos_x [1, 1, 1, W, C]`` and returns their sum, flattened to
    ``[1, D*H*W, C]``. Parameter count is ``(D + H + W) * C`` instead of
    ``D*H*W*C``.

    The grid is fixed at construction: the runtime bottleneck must match it,
    which the forward pass asserts rather than silently interpolating.
    """

    def __init__(self, grid_shape: Sequence[int], channels: int) -> None:
        super().__init__()
        depth, height, width = (int(v) for v in grid_shape)
        self.grid_shape = (depth, height, width)
        self.channels = int(channels)
        self.pos_z = nn.Parameter(torch.zeros(1, depth, 1, 1, channels))
        self.pos_y = nn.Parameter(torch.zeros(1, 1, height, 1, channels))
        self.pos_x = nn.Parameter(torch.zeros(1, 1, 1, width, channels))
        for parameter in (self.pos_z, self.pos_y, self.pos_x):
            nn.init.trunc_normal_(parameter, std=0.02)

    def forward(self, grid_shape: Sequence[int] | None = None) -> Tensor:
        """Return ``[1, D*H*W, C]`` positional codes for the configured grid."""
        if grid_shape is not None and tuple(int(v) for v in grid_shape) != self.grid_shape:
            raise ValueError(
                f"positional encoding is built for grid {self.grid_shape}, got "
                f"{tuple(grid_shape)}; rebuild the model for this resolution"
            )
        codes = self.pos_z + self.pos_y + self.pos_x  # [1, D, H, W, C]
        return codes.reshape(1, -1, self.channels)


def flatten_spatial(x: Tensor) -> Tensor:
    """``[B, C, D, H, W] -> [B, D*H*W, C]`` (tokens last-but-one, as MHA wants)."""
    batch, channels = x.shape[0], x.shape[1]
    return x.reshape(batch, channels, -1).transpose(1, 2)


def normalized_world_grid(
    grid_shape: Sequence[int],
    full_shape: Sequence[int],
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Continuous normalised world coordinates ``(x, y, z)`` on a (possibly coarse) grid.

    Returns ``[1, 3, D', H', W']``, channel order ``(x, y, z)``, values in
    ``[-1, 1]`` at the corners of the *full* volume.

    The coordinates are world coordinates, not tensor indices: a cell of a grid
    that is ``f`` times coarser than the full volume covers input indices
    ``f*i .. f*i + f - 1``, so its world position is ``(f*i + (f-1)/2) *
    spacing``. Feeding local indices instead would make the same anatomical
    position take different values at different scales, which is exactly the
    failure CLAUDE.md calls out ("Recompute them correctly after resizing or
    cropping; local tensor indices are insufficient").

    Args:
        grid_shape: ``(D', H', W')`` of the feature map being annotated.
        full_shape: ``(D, H, W)`` of the input volume the world frame is tied to.
        spacing: world units per voxel, ordered ``(x, y, z)``.
    """
    grid = tuple(int(v) for v in grid_shape)
    full = tuple(int(v) for v in full_shape)
    if len(grid) != 3 or len(full) != 3:
        raise ValueError(f"grid_shape and full_shape must be (D, H, W), got {grid} and {full}")
    if any(v <= 0 for v in grid + full):
        raise ValueError(f"shapes must be positive, got {grid} and {full}")
    spacing_xyz = tuple(float(v) for v in spacing)
    if len(spacing_xyz) != 3 or any(v <= 0 for v in spacing_xyz):
        raise ValueError(f"spacing must be three positive values (x, y, z), got {spacing}")

    axes: list[Tensor] = []  # in (z, y, x) order to match the array layout
    for axis, (size, full_size) in enumerate(zip(grid, full)):
        # spacing is (x, y, z); array axes are (z, y, x).
        step = spacing_xyz[2 - axis]
        factor = full_size / size
        indices = torch.arange(size, device=device, dtype=dtype)
        world = (indices * factor + (factor - 1.0) / 2.0) * step
        half_extent = ((full_size - 1) / 2.0) * step
        centre = half_extent
        axes.append((world - centre) / max(half_extent, 1e-8))

    depth, height, width = grid
    z = axes[0].view(1, 1, depth, 1, 1).expand(1, 1, depth, height, width)
    y = axes[1].view(1, 1, 1, height, 1).expand(1, 1, depth, height, width)
    x = axes[2].view(1, 1, 1, 1, width).expand(1, 1, depth, height, width)
    return torch.cat([x, y, z], dim=1).contiguous()


def downsample_masks(masks: Tensor, size: Sequence[int], *, mode: str = "max") -> Tensor:
    """Resize binary mask channels to a coarser grid.

    ``mode='max'`` preserves presence (a torus is one voxel thick at 16^3 and
    disappears under averaging); ``mode='avg'`` gives the occupancy fraction,
    which is what masked pooling weights want.

    The fixed-kernel pooling path is not an optimisation, it is a portability
    requirement: ``aten::_adaptive_avg_pool3d`` has no MPS kernel, so the
    adaptive variants raise ``NotImplementedError`` on Apple Silicon - one of
    the two devices this project must run on. Every ratio here is a power of
    two (64 -> 8), so an exact integer kernel always exists; the interpolation
    fallback only covers a hypothetical non-integer ratio.
    """
    target = tuple(int(v) for v in size)
    source = tuple(int(v) for v in masks.shape[2:])
    if source == target:
        return masks
    factors = [s // t for s, t in zip(source, target)]
    exact = all(f >= 1 for f in factors) and [
        t * f for t, f in zip(target, factors)
    ] == list(source)
    if mode == "max":
        if exact:
            return F.max_pool3d(masks, kernel_size=factors, stride=factors)
        return F.interpolate(masks, size=target, mode="nearest")
    if mode == "avg":
        if exact:
            return F.avg_pool3d(masks, kernel_size=factors, stride=factors)
        return F.interpolate(masks, size=target, mode="area")
    raise ValueError(f"mode must be 'max' or 'avg', got {mode!r}")
