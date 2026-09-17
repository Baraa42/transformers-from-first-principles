import torch

from transformers_from_scratch.rope import apply_rope


def test_rope_preserves_shape_and_position_zero() -> None:
    x = torch.randn(2, 4, 5, 4)
    rotated = apply_rope(x)

    assert rotated.shape == (2, 4, 5, 4)
    # Position zero has zero rotation angle, so it is unchanged.
    assert torch.allclose(rotated[:, :, 0], x[:, :, 0])
