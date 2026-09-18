import torch

import transformers_from_scratch.data as data
from transformers_from_scratch.data import CausalWindowDataset, create_dataloaders


def test_load_tokenizer_uses_dataset_repository(monkeypatch) -> None:
    calls = {}

    class FakeTokenizer:
        @staticmethod
        def from_file(path: str) -> str:
            return path

    def fake_download(**kwargs: str) -> str:
        calls.update(kwargs)
        return "local-tokenizer.json"

    monkeypatch.setattr(data, "Tokenizer", FakeTokenizer)
    monkeypatch.setattr(data, "hf_hub_download", fake_download)
    assert data.load_tokenizer("fracagnetta/tinystories.BPE8192") == "local-tokenizer.json"
    assert calls == {
        "repo_id": "fracagnetta/tinystories.BPE8192",
        "filename": "tinystories.BPE8192.tokenizer.json",
        "repo_type": "dataset",
    }


def test_causal_window_shift_shape_and_dtype() -> None:
    dataset = CausalWindowDataset([[10, 11, 12, 13, 14]], context_length=4)
    input_ids, targets = dataset[0]

    assert input_ids.tolist() == [10, 11, 12, 13]
    assert targets.tolist() == [11, 12, 13, 14]
    assert input_ids.shape == targets.shape == (4,)
    assert input_ids.dtype == targets.dtype == torch.long
    assert torch.equal(targets[:-1], input_ids[1:])


def test_short_stories_do_not_create_windows() -> None:
    dataset = CausalWindowDataset([[1, 2, 3, 4]], context_length=4)
    assert len(dataset) == 0


def test_dataloader_batches_are_two_dimensional() -> None:
    train_loader, val_loader = create_dataloaders(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
        [[11, 12, 13, 14, 15]],
        context_length=4,
        batch_size=2,
    )
    inputs, targets = next(iter(train_loader))
    val_inputs, val_targets = next(iter(val_loader))
    assert inputs.shape == targets.shape == (2, 4)
    assert val_inputs.shape == val_targets.shape == (1, 4)


def test_training_loader_drops_an_incomplete_batch() -> None:
    train_loader, _ = create_dataloaders(
        [[0, 1, 2, 3, 4, 5]],
        [[0, 1, 2, 3, 4, 5]],
        context_length=1,
        batch_size=2,
    )

    assert [inputs.shape[0] for inputs, _ in train_loader] == [2, 2]


def test_validation_loader_keeps_an_incomplete_batch() -> None:
    _, val_loader = create_dataloaders(
        [[0, 1, 2, 3, 4, 5]],
        [[0, 1, 2, 3, 4, 5]],
        context_length=1,
        batch_size=2,
    )

    assert [inputs.shape[0] for inputs, _ in val_loader] == [2, 2, 1]
