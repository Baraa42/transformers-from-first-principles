import torch

from transformers_from_scratch.rope import apply_rope


def test_rope_preserves_shape_and_position_zero() -> None:
    x = torch.randn(2, 4, 5, 4)
    rotated = apply_rope(x)

    assert rotated.shape == (2, 4, 5, 4)
    # Position zero has zero rotation angle, so it is unchanged.
    assert torch.allclose(rotated[:, :, 0], x[:, :, 0])


def test_rope_single_token_offset_matches_full_sequence_position() -> None:
    prefix_length = 5
    x = torch.randn(2, 3, prefix_length + 1, 8)

    full = apply_rope(x)
    new_only = apply_rope(x[:, :, -1:, :], offset=prefix_length)

    assert torch.allclose(full[:, :, -1:, :], new_only)


def test_rope_chunk_offset_matches_full_sequence_positions() -> None:
    prefix_length = 5
    chunk_length = 3
    x = torch.randn(2, 3, prefix_length + chunk_length, 8)

    full = apply_rope(x)
    chunk = apply_rope(x[:, :, prefix_length:, :], offset=prefix_length)

    assert torch.allclose(full[:, :, prefix_length:, :], chunk)


def test_rope_default_offset_matches_explicit_zero() -> None:
    x = torch.randn(2, 3, 5, 8)

    assert torch.equal(apply_rope(x), apply_rope(x, offset=0))


def test_rope_rejects_invalid_offsets() -> None:
    x = torch.randn(1, 1, 2, 4)

    for offset in (-1, 1.5, True):
        try:
            apply_rope(x, offset=offset)
        except ValueError as error:
            assert str(error) == "RoPE offset must be a non-negative integer"
        else:
            raise AssertionError(f"Expected invalid offset {offset!r} to raise ValueError")
