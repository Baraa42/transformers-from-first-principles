import torch

from transformers_from_scratch.generation import generate, sample_next_token
from transformers_from_scratch.model import TinyDecoderLM


def test_generate_appends_requested_number_of_tokens_and_restores_mode() -> None:
    model = TinyDecoderLM(vocab_size=10, d_model=8, n_heads=2, d_ff=16, n_layers=1)
    model.train()
    output = generate(
        model, torch.tensor([[1, 2, 3]], dtype=torch.long), max_new_tokens=4, context_length=3
    )
    assert output.shape == (1, 7)
    assert model.training


def test_top_k_one_always_selects_the_largest_logit() -> None:
    logits = torch.tensor([[0.0, 1.0, 3.0, 2.0]])
    assert sample_next_token(logits, top_k=1).item() == 2
