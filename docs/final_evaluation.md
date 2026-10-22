# Final checkpoint evaluation

## Result

The existing `checkpoints/tinystories-2k.pt` checkpoint was evaluated once over the complete configured validation split. No training was performed and no model weights were changed.

| Metric | Result |
|---|---:|
| Validation cross-entropy | 3.232746124267578 |
| Validation perplexity | 25.34917345721539 |
| Completed validation batches | 4,386 |
| Completed validation windows | 70,162 |
| Completed target tokens | 8,980,736 |
| Wall-clock evaluation time | 22.193376415991224 seconds |
| Actual device | Apple MPS (`mps`) |
| Precision | FP32 |

The wall time was measured around the complete `evaluate()` call, with an MPS synchronization immediately before starting and immediately after completion.

## Evaluation methodology

- Evaluation used the repository's existing `evaluate()` implementation with `max_batches=None`.
- `evaluate()` placed the model in evaluation mode, ran under `torch.inference_mode()`, computed a target-token-weighted mean cross-entropy, and returned perplexity as `exp(mean_token_cross_entropy)`.
- The validation loader was sequential (`shuffle=False`) and retained its final incomplete batch (`drop_last=False`). Its batch size was 16; the final batch contained 2 windows.
- Every window contained 128 target tokens. Therefore, 70,162 windows contributed exactly 8,980,736 target tokens.
- The run used FP32. No autocast or reduced-precision inference was enabled.
- Model state loading was strict and succeeded with all keys matched.
- The checkpoint was loaded safely on CPU before its model state was copied to the model on MPS.

## Data and tokenizer provenance

| Field | Value |
|---|---|
| Dataset | `karpathy/tinystories-gpt4-clean` |
| Dataset split used by the loader | `train` |
| Deterministic validation story range | `[10,000, 11,000)` |
| Validation stories | 1,000 |
| Tokenizer repository | `fracagnetta/tinystories.BPE8192` |
| Tokenizer vocabulary size | 8,192 |
| Context length | 128 |
| Validation windows | 70,162 |
| Validation DataLoader | batch size 16, `shuffle=False`, `drop_last=False`, `num_workers=0`, `pin_memory=False` |

The dataset and tokenizer identifiers are repository/config identifiers; this run did not record immutable upstream Hugging Face revision hashes.

## Checkpoint provenance

| Field | Value |
|---|---|
| Path | `checkpoints/tinystories-2k.pt` |
| SHA-256 | `7a46afad1ef5d1a7159e9461e8f8bfb93ad09f269c673b87f837bc5f141c8e21` |
| Saved optimizer step | 2,000 |
| Model configuration | `d_model=128`, `n_heads=4`, `d_ff=512`, `n_layers=4` |
| Checkpoint vocabulary size | 8,192 |
| Strict model-state load | All keys matched successfully |

## Execution provenance

| Field | Value |
|---|---|
| Git HEAD | `4e1ef661dbee5a6e7827f2592c0b739c23939a1d` |
| Python | 3.12.13 |
| PyTorch | 2.14.0 |
| Device selected from `device: auto` | `mps` |
| Evaluation precision | `fp32` |
| `max_batches` | `None` |

The working tree already contained uncommitted changes to `poetry.lock`, `pyproject.toml`, `src/transformers_from_scratch/attention.py`, `src/transformers_from_scratch/training.py`, and `train.py` when this evaluation was run. Those changes were preserved. This result therefore records both the exact checkpoint digest and the code state used for evaluation.

## Interpretation boundary

This is a full evaluation over the repository's configured 1,000-story validation subset, not a test-set result. It establishes the checkpoint's loss and perplexity under the current evaluation implementation. No separate CPU notebook training trajectory is attributed to this checkpoint.
