# Inference API

The repository exposes the trained `TinyDecoderLM` through a deliberately small FastAPI
service. It is an educational, testable serving baseline—not a claim of production readiness.

## Start the service

Install dependencies and launch Uvicorn from the repository root:

```bash
poetry install
poetry run uvicorn serve:app --host 127.0.0.1 --port 8000
```

The default checkpoint is `checkpoints/tinystories-2k.pt`, and the device is selected in the
order CUDA, MPS, then CPU. Both can be overridden:

```bash
TRANSFORMER_CHECKPOINT=checkpoints/tinystories-2k.pt \
TRANSFORMER_DEVICE=cpu \
poetry run uvicorn serve:app --host 127.0.0.1 --port 8000
```

## Model lifecycle

FastAPI's lifespan hook runs once per application process. At startup it:

1. resolves the requested device;
2. safely deserializes the legacy checkpoint onto CPU with `weights_only=True`;
3. reads the model shape, vocabulary, tokenizer repository, and context length from checkpoint
   metadata;
4. loads the tokenizer once;
5. constructs the manual `TinyDecoderLM`, strictly loads its weights, moves it to the selected
   device, and selects evaluation mode;
6. stores the resulting runtime on application state.

The service does not use the training-resume loader because inference needs neither optimizer
nor scaler/RNG state. `/health` reports the actual loaded device and checkpoint filename rather
than hardcoded values.

## Request lifecycle

For `POST /generate`, the service:

1. validates and trims one prompt;
2. tokenizes it and checks the dynamic vocabulary/context constraints;
3. acquires the process-local inference lock;
4. performs one full-prompt prefill that creates each layer's KV cache;
5. repeatedly forwards only the newest token, reusing the cache and correct RoPE offset;
6. selects tokens with argmax for greedy decoding or with the existing temperature/top-k/top-p
   sampler;
7. stops early when the single generated sequence emits the tokenizer's EOS token;
8. synchronizes the accelerator, decodes the complete prompt plus completion, and returns the
   actual token counts and request latency.

The shared cached loop runs under `torch.inference_mode()`. This disables autograd bookkeeping,
which reduces inference memory and overhead; it does not change the trained parameters.

Set `do_sample` to `false` for deterministic greedy generation. When it is `true` (the default),
`temperature`, `top_k`, and `top_p` control sampling.

## Endpoints

Check lifecycle state:

```bash
curl http://127.0.0.1:8000/health
```

Generate a sampled completion:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Once upon a time",
    "max_new_tokens": 50,
    "do_sample": true,
    "temperature": 0.8,
    "top_k": 40,
    "top_p": 0.9
  }'
```

For greedy decoding, send `"do_sample": false`. `generated_text` contains the decoded prompt and
completion; `generated_tokens` is the actual completion length and may be below the requested
maximum after EOS.

## Validation and context limits

- Prompts must remain non-empty after trimming and must encode to at least one token.
- `max_new_tokens` is limited to 1–256 at the HTTP boundary.
- Sampling temperature must be positive, `top_k` must be positive and no greater than the model
  vocabulary, and `top_p` must be in `(0, 1]`.
- With prompt length `P` and generation limit `N`, cached execution must process at most
  `P + N - 1` positions. That value cannot exceed the checkpoint's training context length.
- There is no cache eviction or sliding window. Oversized requests receive HTTP 422 instead of
  silently falling back to uncached generation.

## Concurrency policy

HTTP requests may arrive concurrently, but a process-local lock permits only one request at a
time to execute model inference. Other generation requests wait, and their reported/request
latency includes this queueing time. This avoids uncontrolled concurrent work on one accelerator
without introducing a scheduler. Each Uvicorn worker would load its own model and own lock, so
this baseline should run with one worker.

## Deliberate non-goals

This block does not implement continuous/dynamic batching, streaming responses, batched EOS
tracking, cache eviction, sliding-window decoding, distributed workers, authentication, rate
limits, request cancellation, HTTP benchmarking, containers, or an external job queue. Those
require separate correctness and systems work rather than being hidden behind this minimal API.
