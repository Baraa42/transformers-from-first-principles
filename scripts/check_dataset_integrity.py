"""Audit the configured TinyStories splits, duplicates, tokens, and windows."""

import argparse
import hashlib
import unicodedata
from collections import Counter
from pathlib import Path

from transformers_from_scratch.data import (
    CausalWindowDataset,
    load_tinystories_text_splits,
    load_tokenizer,
    tokenize_stories,
)
from transformers_from_scratch.training import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "tiny.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Training configuration whose data section should be audited.",
    )
    return parser.parse_args()


def normalize_story(text: str) -> str:
    """Apply Unicode NFKC, collapse whitespace, trim, and preserve case."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def story_hash(text: str) -> str:
    """Return the SHA-256 digest of one normalized story."""
    return hashlib.sha256(normalize_story(text).encode("utf-8")).hexdigest()


def duplicate_record_count(counts: Counter[str]) -> int:
    """Count redundant records beyond the first occurrence of each digest."""
    return sum(count - 1 for count in counts.values())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    context_length = data_config["context_length"]

    train_text, val_text = load_tinystories_text_splits(
        data_config["dataset_name"],
        data_config["train_stories"],
        data_config["val_stories"],
    )
    train_counts = Counter(story_hash(story) for story in train_text)
    val_counts = Counter(story_hash(story) for story in val_text)
    overlapping_hashes = train_counts.keys() & val_counts.keys()
    cross_split_duplicate_pairs = sum(
        train_counts[digest] * val_counts[digest] for digest in overlapping_hashes
    )

    tokenizer = load_tokenizer(data_config["tokenizer_repo"])
    train_tokens = tokenize_stories(tokenizer, train_text)
    val_tokens = tokenize_stories(tokenizer, val_text)
    train_windows = CausalWindowDataset(train_tokens, context_length)
    val_windows = CausalWindowDataset(val_tokens, context_length)

    print(f"config={args.config.resolve()}")
    print("normalization=Unicode NFKC, collapse all whitespace, trim, preserve case")
    print("hash=SHA-256 of normalized UTF-8 text")
    print(f"train_stories={len(train_text)}")
    print(f"validation_stories={len(val_text)}")
    print(f"exact_overlap_unique_normalized_stories={len(overlapping_hashes)}")
    print(f"exact_duplicates_within_train={duplicate_record_count(train_counts)}")
    print(f"exact_duplicates_within_validation={duplicate_record_count(val_counts)}")
    print(f"exact_duplicates_across_train_validation={cross_split_duplicate_pairs}")
    print(f"train_tokens={sum(len(story) for story in train_tokens)}")
    print(f"validation_tokens={sum(len(story) for story in val_tokens)}")
    print(f"train_causal_windows={len(train_windows)}")
    print(f"validation_causal_windows={len(val_windows)}")
    print(f"context_length={context_length}")

    if overlapping_hashes:
        print("integrity_status=FAIL: exact normalized story overlap detected")
        raise SystemExit(1)
    print("integrity_status=PASS: no exact normalized story overlap detected")


if __name__ == "__main__":
    main()
