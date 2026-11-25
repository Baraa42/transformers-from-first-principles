# Final documentation and reproducibility review

Review date: 2026-11-25. Source baseline: `443838a` (completed cache comparison).
This pass changes documentation and ignore rules only. Production source, tests,
benchmark runners, dependencies, configuration and retained measurements are unchanged.
No standalone training, benchmark, profiler or checkpoint-evaluation run was launched.

## Deliverables

- A reorganized [README](../README.md) with setup, training/resume, sampling, checks,
  benchmark entry points, evidence links, and bounded performance claims.
- An [interview summary](project_summary.md) with short and technical explanations,
  lessons, trade-offs, numerical results, and links to code/evidence.
- Completed [architecture notes](../notes/architecture.md) replacing placeholders.
- Small corrections to cache and historical performance documentation: completed-run
  status, tracked raw evidence, stale line references, fixed-cache versus growing-request
  scope, and the limits of logged-loss and utilization interpretations.
- Ignore rules for local `.env` files, while allowing `.env.example` and `.env.template`.
  Existing benchmark ignore rules and all explicitly tracked evidence are retained.

## Validation results

| Check | Result |
| --- | --- |
| Full pytest suite | **255 passed, 10 skipped**, one existing warning |
| Repository-wide Ruff lint | Passed |
| Repository-wide Ruff formatting | Passed, including Python examples in Markdown |
| Package/lockfile metadata (`poetry check --lock`) | Passed; no dependency changes |
| CLI audit | Eleven argparse entry points accepted `--help`; legacy precision/scaling runners were inspected without execution |
| Local documentation links | All README, report and notes targets and referenced Markdown anchors resolve |
| Parameter count and input hashes | Configured model has 2,888,448 parameters; both cache runs' weight/prompt hashes and both controlled training runs' weight/batch hashes match |
| Training integrity manifest | All 12 archived files and 17 original artifact records match size and SHA-256 |
| Compressed training correctness reports | Decompressed SHA-256 matches original traces and navigation summaries; both gates passed |
| Cache integrity manifest | All eight Phase 4A/4B files match SHA-256 |
| Artifact preservation | All 61 pre-pass evidence/checkpoint files retain their original checksums |
| Tracked-file hygiene | No tracked credential files, system metadata, caches or temporary files; no matches in the private-key/token/credential-pattern scan |
| Production and dependency diff | No changes |

The test runtime used Python 3.12.15 / PyTorch 2.14.0. MPS was unavailable in this
runtime, accounting for the ten skipped tests. The completed MPS experiments are
historical evidence, not measurements rerun by this pass. Existing tests include small
CPU instrumentation fixtures; those are correctness tests, not new research experiments.
Ruff was run with `--no-cache` because the source checkout is read-only to this session.
The checks cover source, tests, scripts, root entry points and applicable documentation
examples; intentionally ignored notebooks are outside this lint/format scope.

Reproduction/check commands from the repository root:

```bash
poetry run pytest -q
poetry run ruff check --no-cache .
poetry run ruff format --check --no-cache .
poetry check --lock
```

The first whole-repository formatting check identified one pre-existing Python example
in `preallocated_cache.md`; the example was formatted and the final check passed. No
production file needed formatting changes.

### Existing warning

Pytest emitted `StarletteDeprecationWarning`: using `httpx` with
`starlette.testclient` is deprecated in favor of `httpx2`. No dependency was installed
or upgraded to suppress it. There were no test failures or remaining lint/format failures.

## Numerical and provenance checks

The audit recomputed controlled training worker statistics directly from both archived
CSVs, including warmup flags/counts, medians, p95, spread, derived throughput, and A→D
ratios. Run-003 contains 2,240 measured updates; its primary latencies are 17.88173975 ms
for A and 15.9333125 ms for D, giving **10.8961839%** lower latency and **1.1222864x**
speedup. Run-002 gives **11.8652719%**. The [training report](training_optimizations.md)
correctly describes a median of worker p50s, not a pooled step median. Its hardware,
software and environment descriptions agree with run metadata. Both archived source
snapshots match their recorded hashes.

The cache audit checked workloads, repetitions, alternating pair order, timing denominators,
CSV-derived paired ratios, summary medians, throughput, logical/reserved shapes/bytes,
and theoretical old-prefix accounting. It reproduces **1.0123231x** median paired decode
speedup, **1.0031289x** median workload speedup and **60/108** preallocation wins.
The 18 workload gates report a maximum absolute logit difference of zero. All Phase 4B
source hashes match current source. Phase 4A's model/generation hashes match its recorded
Git revision; its other recorded source hashes match current files. See the
[cache integrity record](benchmarks/cache-comparison-run-001-integrity.json).

The historical evaluation checkpoint matches its recorded SHA-256. Evaluation window,
batch and target-token counts are arithmetically consistent, as is perplexity from loss.
The checkpoint was not reevaluated; that historical result remains bounded by the
[data/revision limitations in its report](final_evaluation.md).

Historical precision/scaling tables match the retained local logs. The older 19.1%
training result and early inference tables do not have complete retained raw timing
series. Their status is explicitly historical; they are not used as the controlled
headline or as substitutes for the paired cache evidence. No missing traces, upstream
revision hashes, hardware defaults or effective environment values were invented.

## Evidence organization and remaining boundaries

Controlled training evidence and source snapshots are committed under
`docs/benchmarks/training-optimizations/`; the original local runs, failed gates and
legacy precision/scaling logs remain untouched. Eight selected cache run files are
explicitly tracked under `benchmarks/`, although the directory otherwise ignores new
outputs. Existing manifests are unchanged. Checksums establish byte integrity against
those records, not independent historical authenticity.

Checkpoints and datasets are local artifacts, not distributed assets. Dataset/tokenizer
identifiers do not pin immutable upstream revisions. Resume does not recover the exact
DataLoader position. The basic HTTP API is not a production scheduler. Synthetic MPS
latency results neither establish real-data/model-quality improvements nor generalize
to other devices, larger models or production workloads.

## Review scope and proposed commit

Changed/new files in the prepared documentation pass:

- `.gitignore`
- `README.md`
- `docs/project_summary.md`
- `docs/final_review.md`
- `docs/cache_growth.md`
- `docs/preallocated_cache.md`
- `docs/inference.md`
- `docs/performance.md`
- `notes/architecture.md`

Proposed single commit message:

```text
docs: finalize Transformer research portfolio and reproducibility guide
```

Nothing was staged, committed or pushed. The source checkout was clean at review start
and remains untouched; the documentation changes are supplied as an applyable patch
because this session cannot write to that checkout.
