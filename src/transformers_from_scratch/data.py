"""TinyStories loading, one-time tokenization, and causal-LM windows."""

from collections.abc import Sequence

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset


def load_tokenizer(
    repo_id: str,
    filename: str = "tinystories.BPE8192.tokenizer.json",
) -> Tokenizer:
    """Load a tokenizer file from the published TinyStories dataset repository."""
    path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    return Tokenizer.from_file(path)


def load_tinystories_text_splits(
    dataset_name: str,
    train_stories: int = 20_000,
    val_stories: int = 1_000,
    *,
    validation_start: int = 10_000,
    train_start: int = 20_000,
) -> tuple[list[str], list[str]]:
    """Load deterministic, disjoint text ranges from TinyStories' train split."""
    dataset = load_dataset(dataset_name, split="train")
    val_end, train_end = validation_start + val_stories, train_start + train_stories
    if train_start < val_end or len(dataset) < max(val_end, train_end):
        raise ValueError("Requested TinyStories ranges are overlapping or unavailable")
    validation = dataset.select(range(validation_start, val_end))["text"]
    training = dataset.select(range(train_start, train_end))["text"]
    return list(training), list(validation)


def tokenize_stories(tokenizer: Tokenizer, stories: Sequence[str]) -> list[list[int]]:
    """Tokenize every story once and append EOS when this tokenizer exposes one."""
    vocab = tokenizer.get_vocab()
    eos_id = next(
        (vocab[token] for token in ("<|endoftext|>", "<eos>", "</s>") if token in vocab), None
    )
    tokens: list[list[int]] = []
    for story in stories:
        ids = tokenizer.encode(story).ids
        if eos_id is not None:
            ids.append(eos_id)
        tokens.append(ids)
    return tokens


class CausalWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """All valid fixed-size causal windows, without crossing story boundaries."""

    def __init__(self, stories: Sequence[Sequence[int]], context_length: int) -> None:
        if context_length < 1:
            raise ValueError("context_length must be positive")
        self.stories = [list(story) for story in stories]
        self.context_length = context_length
        self.windows = [
            (story_index, start)
            for story_index, story in enumerate(self.stories)
            for start in range(max(0, len(story) - context_length))
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        story_index, start = self.windows[index]
        chunk = self.stories[story_index][start : start + self.context_length + 1]
        values = torch.tensor(chunk, dtype=torch.long)
        return values[:-1], values[1:]


def create_dataloaders(
    train_stories: Sequence[Sequence[int]],
    val_stories: Sequence[Sequence[int]],
    *,
    context_length: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create shuffled training and ordered validation loaders."""
    options = {"batch_size": batch_size, "num_workers": num_workers, "pin_memory": pin_memory}
    return (
        DataLoader(CausalWindowDataset(train_stories, context_length), shuffle=True, **options),
        DataLoader(CausalWindowDataset(val_stories, context_length), shuffle=False, **options),
    )
