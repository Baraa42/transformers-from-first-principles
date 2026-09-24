import torch

from transformers_from_scratch.generation import (
    generate,
    generate_greedy,
    greedy_next_token,
    sample_next_token,
)
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


def test_greedy_next_token_selects_argmax() -> None:
    logits = torch.tensor(
        [
            [0.0, 1.0, 3.0, 2.0],
            [4.0, 0.0, 1.0, 2.0],
        ]
    )

    tokens = greedy_next_token(logits)

    assert torch.equal(tokens, torch.tensor([[2], [0]], dtype=torch.long))
    assert tokens.shape == (2, 1)
    assert tokens.dtype == torch.long


def test_generate_greedy_appends_tokens_and_restores_mode() -> None:
    model = TinyDecoderLM(vocab_size=10, d_model=8, n_heads=2, d_ff=16, n_layers=1)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)

    model.train()
    output = generate_greedy(model, input_ids, max_new_tokens=4, context_length=3)
    assert output.shape == (2, 7)
    assert model.training

    model.eval()
    generate_greedy(model, input_ids, max_new_tokens=1, context_length=3)
    assert not model.training


def test_generate_greedy_is_deterministic_without_resetting_seed() -> None:
    model = TinyDecoderLM(vocab_size=10, d_model=8, n_heads=2, d_ff=16, n_layers=1)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    output_1 = generate_greedy(model, input_ids, max_new_tokens=4, context_length=3)
    output_2 = generate_greedy(model, input_ids, max_new_tokens=4, context_length=3)

    assert torch.equal(output_1, output_2)


def test_generate_greedy_does_not_use_multinomial(monkeypatch) -> None:
    def fail_multinomial(*args, **kwargs):
        raise AssertionError("greedy generation must not call torch.multinomial")

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)
    model = TinyDecoderLM(vocab_size=10, d_model=8, n_heads=2, d_ff=16, n_layers=1)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    output = generate_greedy(model, input_ids, max_new_tokens=2, context_length=3)

    assert output.shape == (1, 5)
