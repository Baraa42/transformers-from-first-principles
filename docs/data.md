# TinyStories Data Pipeline

## Dataset

TinyStories provides short, syntactically simple stories that make language-model training
and generation practical for this small educational Transformer. The primary
`configs/tiny.yaml` run uses 20,000 training stories and 1,000 validation stories from the
upstream TinyStories `train` split.

`load_tinystories_text_splits` selects deterministic story-level ranges before tokenization:

- validation: indices `[10_000, 10_000 + val_stories)`;
- training: indices `[20_000, 20_000 + train_stories)`.

It rejects overlapping or unavailable configured ranges. These are two ranges from the
upstream training split; this repository does not use TinyStories' upstream validation split.

Each selected story is tokenized independently with the configured published BPE tokenizer.
If that tokenizer exposes `<|endoftext|>`, `<eos>`, or `</s>`, the first matching token in
that order is appended to every story. Otherwise no EOS token is added.

## Causal language-model examples

For context length `T = 4`, one five-token chunk

```text
[10, 11, 12, 13, 14]
```

becomes:

```text
input:  [10, 11, 12, 13]
target: [11, 12, 13, 14]
```

Thus every example consumes `T + 1` tokens and trains each position to predict the next
token. The dataset enumerates every valid stride-one window inside each tokenized story.
Stories with at most `T` tokens produce no windows.

## Split before windowing

Stories are selected into training and validation ranges before tokenization and before
causal windows are constructed. `CausalWindowDataset` records `(story_index, start)` for
each window and slices only that one story. A window can therefore never cross a story
boundary, and windows derived from one selected story cannot be assigned to both loaders.

The training loader uses shuffled sampling and drops its incomplete final batch. The
validation loader preserves dataset order and retains its incomplete final batch.

## What this repository currently guarantees

- Configured training and validation index ranges are deterministic and disjoint.
- Tokenization and optional EOS insertion happen independently per story.
- Fixed-length windows stay within individual story boundaries.
- Inputs and targets are exact one-token shifts of the same `T + 1` token chunk.
- `scripts/check_dataset_integrity.py` hashes normalized story text, reports exact duplicate
  counts within and across the configured splits, and exits unsuccessfully if any normalized
  story hash appears in both splits.

The integrity normalization applies Unicode NFKC, replaces every whitespace run with one
ASCII space, removes leading/trailing whitespace, and preserves case. A duplicate count
within one split counts redundant records beyond the first occurrence. The cross-split count
counts matching train/validation record pairs; the report also gives the number of distinct
normalized story texts shared by both splits.

Run the audit against the same configuration used by training:

```bash
poetry run python scripts/check_dataset_integrity.py --config configs/tiny.yaml
```

## What this does not guarantee

SHA-256 equality after the stated normalization detects exact normalized duplicates only.
It does not detect paraphrases, semantic duplicates, lightly edited stories, or other
near-duplicate contamination. Disjoint source indices alone also do not prove that upstream
content is unique; that is why the explicit content-hash audit exists. The configured
dataset and tokenizer repository names do not pin immutable upstream revisions, so audit
counts describe the artifacts resolved when the command is run.

## What I would add for a production-scale dataset

A production pipeline could add stronger canonicalization, near-duplicate detection with
methods such as MinHash or LSH, deduplication before splitting, train-only fitted
preprocessing or tokenization where relevant, and a final untouched test set. Those are
deliberately not claims about the current repository and are not implemented here.
