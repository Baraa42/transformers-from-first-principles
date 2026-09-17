"""Rotary positional embeddings implemented as adjacent two-dimensional rotations."""

import torch


def apply_rope(x: torch.Tensor, base: float = 10_000.0) -> torch.Tensor:
    """Apply RoPE to vectors shaped ``(B, H, n, d_head)``.

    This efficiently applies the known position-dependent rotation ``R_p`` as
    ``x_p_rope = R_p x_p``. Rotated query/key dot products depend on relative
    position without explicitly constructing rotation matrices.
    """
    if x.ndim != 4:
        raise ValueError(f"RoPE expects (B, H, n, d_head), got {tuple(x.shape)}")
    _, _, n, d_head = x.shape
    if d_head % 2:
        raise ValueError(f"RoPE requires even d_head, got {d_head}")
    # One frequency per adjacent pair: (x0, x1), (x2, x3), ...
    freq = 1.0 / (base ** (torch.arange(0, d_head, 2, device=x.device).float() / d_head))
    positions = torch.arange(n, device=x.device, dtype=freq.dtype)
    angles = positions[:, None] * freq[None, :]
    cos = torch.cos(angles)[None, None, :, :, None]
    sin = torch.sin(angles)[None, None, :, :, None]
    # (B, H, n, d_head) -> (B, H, n, d_head / 2, 2), then rotate and flatten.
    pairs = x.reshape(*x.shape[:-1], d_head // 2, 2)
    x_even, x_odd = pairs.unbind(dim=-1)
    rotated = torch.stack(
        (x_even * cos[..., 0] - x_odd * sin[..., 0], x_even * sin[..., 0] + x_odd * cos[..., 0]),
        dim=-1,
    )
    return rotated.flatten(start_dim=-2)
