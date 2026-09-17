import torch

from transformers_from_scratch.layers import DecoderBlock
from transformers_from_scratch.model import TinyDecoderLM


def test_tiny_decoder_default_shape() -> None:
    B, n, V = 2, 5, 100
    token_ids = torch.randint(0, V, (B, n))
    model = TinyDecoderLM(vocab_size=V, d_model=16, n_heads=4, d_ff=64, n_layers=3)

    logits = model(token_ids)

    assert token_ids.shape == (2, 5)
    assert logits.shape == (2, 5, 100)


def test_decoder_block_preserves_shape() -> None:
    block = DecoderBlock(d_model=16, n_heads=4, d_ff=64)
    assert block(torch.randn(2, 5, 16)).shape == (2, 5, 16)
