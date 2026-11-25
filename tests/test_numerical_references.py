"""CPU FP32 mathematical oracles independent of production forward implementations."""

import math

import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformers_from_scratch.attention import SelfAttention
from transformers_from_scratch.generation import generate_greedy_cached
from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.rope import apply_rope


def rotation_matrix_reference(x, *, offset=0, base=10_000.0):
    """Construct full block-diagonal matrices using scalar trigonometry, then multiply."""
    width = x.shape[-1]
    matrices = []
    for position in range(offset, offset + x.shape[2]):
        blocks = []
        for pair in range(width // 2):
            angle = position * base ** (-2 * pair / width)
            cosine, sine = math.cos(angle), math.sin(angle)
            blocks.append(torch.tensor([[cosine, -sine], [sine, cosine]], dtype=torch.float32))
        matrices.append(torch.block_diag(*blocks))
    return torch.einsum("tij,bhtj->bhti", torch.stack(matrices), x)


def attention_reference(x, weights, *, heads):
    """Independent projections + matrix RoPE + PyTorch's math SDPA implementation."""
    batch, length, width = x.shape
    projected = [
        F.linear(x, weights[name]).reshape(batch, length, heads, width // heads).transpose(1, 2)
        for name in ("Wq.weight", "Wk.weight", "Wv.weight")
    ]
    queries, keys = (rotation_matrix_reference(value) for value in projected[:2])
    values = projected[2]
    with sdpa_kernel(SDPBackend.MATH):
        attended = F.scaled_dot_product_attention(
            queries, keys, values, dropout_p=0.0, is_causal=True
        )
    merged = attended.transpose(1, 2).reshape(batch, length, width)
    return F.linear(merged, weights["Wo.weight"]), (keys, values)


def decoder_reference(ids, weights, *, layers=3, heads=2):
    """Functional full-prefix decoder; never calls a production module's forward."""
    x = F.embedding(ids, weights["token_embedding.weight"])
    caches = []
    for layer in range(layers):
        prefix = f"blocks.{layer}."

        def norm(value, name):
            return F.layer_norm(
                value,
                (value.shape[-1],),
                weights[prefix + name + ".weight"],
                weights[prefix + name + ".bias"],
                eps=1e-5,
            )

        attention_weights = {
            name: weights[prefix + "attn." + name]
            for name in ("Wq.weight", "Wk.weight", "Wv.weight", "Wo.weight")
        }
        output, cache = attention_reference(norm(x, "norm1"), attention_weights, heads=heads)
        caches.append(cache)
        x = x + output
        hidden = F.linear(
            norm(x, "norm2"), weights[prefix + "mlp.up.weight"], weights[prefix + "mlp.up.bias"]
        )
        x = x + F.linear(
            F.silu(hidden), weights[prefix + "mlp.down.weight"], weights[prefix + "mlp.down.bias"]
        )
    x = F.layer_norm(
        x, (x.shape[-1],), weights["final_norm.weight"], weights["final_norm.bias"], eps=1e-5
    )
    return F.linear(x, weights["lm_head.weight"]), tuple(caches)


def check_close(actual, expected, errors, label, *, atol=2e-6, rtol=1e-5):
    assert actual.dtype == expected.dtype == torch.float32
    assert actual.device.type == expected.device.type == "cpu"
    errors[label] = max(errors.get(label, 0.0), (actual - expected).abs().max().item())
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("length", [1, 5, 9])
def test_attention_outputs_and_gradients_against_matrix_rope_sdpa(length):
    torch.manual_seed(101)
    attention = SelfAttention(d_model=16, n_heads=2).float()
    x = torch.randn(2, length, 16, dtype=torch.float32, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    weights = {
        name: parameter.detach().clone().requires_grad_()
        for name, parameter in attention.named_parameters()
    }
    expected, _ = attention_reference(reference_x, weights, heads=2)
    actual = attention(x)
    cotangent = torch.randn_like(actual)
    actual_gradients = torch.autograd.grad((actual * cotangent).sum(), (x, *attention.parameters()))
    expected_gradients = torch.autograd.grad(
        (expected * cotangent).sum(), (reference_x, *weights.values())
    )
    errors = {}
    check_close(actual, expected, errors, "output")
    check_close(actual_gradients[0], expected_gradients[0], errors, "input_gradient")
    for actual_gradient, expected_gradient in zip(
        actual_gradients[1:], expected_gradients[1:], strict=True
    ):
        check_close(actual_gradient, expected_gradient, errors, "projection_gradient")
    print(f"attention length={length} max_absolute_errors={errors}")


def test_attention_future_perturbations_and_gradients_cannot_affect_past():
    torch.manual_seed(102)
    attention = SelfAttention(d_model=16, n_heads=2).float()
    x = torch.randn(2, 7, 16, dtype=torch.float32, requires_grad=True)
    output = attention(x)
    changed = x.detach().clone()
    changed[:, 4:] += 20 * torch.randn_like(changed[:, 4:])
    perturbed_output = attention(changed)
    # Exact equality: identical earlier queries/keys/values cannot see the changed suffix.
    assert torch.equal(output[:, :4], perturbed_output[:, :4])
    assert not torch.allclose(output[:, 4:], perturbed_output[:, 4:])
    gradient = torch.autograd.grad(output[:, :4].square().sum(), x)[0]
    assert torch.count_nonzero(gradient[:, 4:]) == 0
    assert torch.count_nonzero(gradient[:, :4]) > 0
    print("causal prefix_output_max_error=0 future_input_gradient_max=0")


@pytest.mark.parametrize("offset,base", [(0, 10_000.0), (7, 10_000.0), (31, 100.0)])
def test_rope_against_explicit_rotation_matrices_and_norms(offset, base):
    torch.manual_seed(103)
    x = torch.randn(2, 3, 6, 8, dtype=torch.float32, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = apply_rope(x, offset=offset, base=base)
    expected = rotation_matrix_reference(reference_x, offset=offset, base=base)
    errors = {}
    check_close(actual, expected, errors, "rotation", atol=3e-6, rtol=3e-6)
    check_close(x.norm(dim=-1), actual.norm(dim=-1), errors, "norm")
    cotangent = torch.randn_like(actual)
    actual_gradient = torch.autograd.grad((actual * cotangent).sum(), x)[0]
    expected_gradient = torch.autograd.grad((expected * cotangent).sum(), reference_x)[0]
    check_close(actual_gradient, expected_gradient, errors, "gradient", atol=3e-6, rtol=3e-6)
    print(f"rope offset={offset} base={base:g} max_absolute_errors={errors}")


@pytest.mark.parametrize("prompt_length,context_length", [(1, 2), (3, 8), (7, 16)])
def test_cached_logits_and_layer_positions_against_independent_decoder(
    prompt_length, context_length
):
    torch.manual_seed(104)
    model = TinyDecoderLM(vocab_size=23, d_model=16, n_heads=2, d_ff=32, n_layers=3).float()
    model.eval()
    weights = model.state_dict()
    ids = torch.randint(0, 23, (2, context_length))
    errors = {}
    with torch.inference_mode():
        for length in range(prompt_length, context_length + 1):
            expected, expected_cache = decoder_reference(ids[:, :length], weights)
            check_close(model(ids[:, :length]), expected, errors, "full_logits")
            if length == prompt_length:
                actual, cache = model(ids[:, :length], use_cache=True)
                check_close(actual, expected, errors, "prefill_logits")
            else:
                old_cache = tuple((keys.clone(), values.clone()) for keys, values in cache)
                actual, cache = model(ids[:, length - 1 : length], use_cache=True, kv_cache=cache)
                check_close(actual, expected[:, -1:], errors, "incremental_logits")
                for (old_k, old_v), (keys, values) in zip(old_cache, cache, strict=True):
                    assert torch.equal(old_k, keys[:, :, :-1])
                    assert torch.equal(old_v, values[:, :, :-1])
            assert len(cache) == 3
            for (keys, values), (expected_k, expected_v) in zip(cache, expected_cache, strict=True):
                assert keys.shape == values.shape == (2, 2, length, 8)
                check_close(keys, expected_k, errors, "layer_position_keys")
                check_close(values, expected_v, errors, "layer_position_values")

        # The last sampled token is not forwarded: P + N - 1 == context_length.
        new_tokens = context_length - prompt_length + 1
        prompt = ids[:, :prompt_length]
        expected_ids = prompt.clone()
        for _ in range(new_tokens):
            logits, _ = decoder_reference(expected_ids, weights)
            expected_ids = torch.cat((expected_ids, logits[:, -1].argmax(-1, keepdim=True)), 1)
        actual_ids = generate_greedy_cached(
            model, prompt, max_new_tokens=new_tokens, context_length=context_length
        )
        assert torch.equal(actual_ids, expected_ids)
        with pytest.raises(ValueError, match="largest cached prefix"):
            generate_greedy_cached(
                model, prompt, max_new_tokens=new_tokens, context_length=context_length - 1
            )
    print(
        f"cache prompt={prompt_length} context={context_length} "
        f"max_absolute_errors={errors} greedy_token_mismatches=0"
    )
