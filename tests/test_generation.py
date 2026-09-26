import pytest
import torch

from transformers_from_scratch.generation import (
    generate,
    generate_greedy,
    generate_greedy_cached,
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


def test_cached_greedy_matches_uncached_for_multiple_steps_and_batch() -> None:
    torch.manual_seed(31)
    model = TinyDecoderLM(vocab_size=19, d_model=16, n_heads=4, d_ff=32, n_layers=2)
    input_ids = torch.randint(0, 19, (2, 5))
    original_input = input_ids.clone()

    uncached = generate_greedy(model, input_ids, max_new_tokens=6, context_length=11)
    cached = generate_greedy_cached(model, input_ids, max_new_tokens=6, context_length=11)

    assert torch.equal(cached, uncached)
    assert cached.shape == (2, 11)
    assert torch.equal(input_ids, original_input)


def test_cached_greedy_single_token_matches_uncached_prefill() -> None:
    torch.manual_seed(32)
    model = TinyDecoderLM(vocab_size=13, d_model=16, n_heads=4, d_ff=32, n_layers=2)
    input_ids = torch.randint(0, 13, (2, 5))

    uncached = generate_greedy(model, input_ids, max_new_tokens=1, context_length=6)
    cached = generate_greedy_cached(model, input_ids, max_new_tokens=1, context_length=6)

    assert torch.equal(cached, uncached)


def test_cached_greedy_prefills_once_then_forwards_one_token_with_cache() -> None:
    class CacheSpy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = TinyDecoderLM(
                vocab_size=17,
                d_model=16,
                n_heads=4,
                d_ff=32,
                n_layers=2,
            )
            self.call_lengths: list[int] = []
            self.cache_was_none: list[bool] = []

        def forward(self, input_ids: torch.Tensor, *, use_cache=False, kv_cache=None):
            self.call_lengths.append(input_ids.shape[1])
            self.cache_was_none.append(kv_cache is None)
            return self.model(input_ids, use_cache=use_cache, kv_cache=kv_cache)

    model = CacheSpy()
    input_ids = torch.randint(0, 17, (2, 5))

    output = generate_greedy_cached(
        model,
        input_ids,
        max_new_tokens=5,
        context_length=10,
    )

    assert output.shape == (2, 10)
    assert model.call_lengths == [5, 1, 1, 1, 1]
    assert model.cache_was_none == [True, False, False, False, False]


@pytest.mark.parametrize("starts_in_training_mode", [True, False])
def test_cached_greedy_restores_model_mode(starts_in_training_mode: bool) -> None:
    model = TinyDecoderLM(vocab_size=10, d_model=8, n_heads=2, d_ff=16, n_layers=1)
    model.train(starts_in_training_mode)

    generate_greedy_cached(
        model,
        torch.tensor([[1, 2, 3]], dtype=torch.long),
        max_new_tokens=2,
        context_length=5,
    )

    assert model.training is starts_in_training_mode


def test_cached_greedy_allows_zero_tokens_without_mutating_input() -> None:
    model = TinyDecoderLM(vocab_size=10, d_model=8, n_heads=2, d_ff=16, n_layers=1)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    output = generate_greedy_cached(
        model,
        input_ids,
        max_new_tokens=0,
        context_length=3,
    )

    assert torch.equal(output, input_ids)
    assert output.data_ptr() != input_ids.data_ptr()


@pytest.mark.parametrize(
    ("max_new_tokens", "context_length", "message"),
    [
        (-1, 3, "max_new_tokens"),
        (1, 3, "context_length"),
    ],
)
def test_cached_greedy_rejects_invalid_workloads(
    max_new_tokens: int,
    context_length: int,
    message: str,
) -> None:
    model = TinyDecoderLM(vocab_size=10, d_model=8, n_heads=2, d_ff=16, n_layers=1)
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with pytest.raises(ValueError, match=message):
        generate_greedy_cached(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            context_length=context_length,
        )
