"""Train the tiny decoder from a plain YAML configuration file."""

import argparse
import time
from itertools import cycle
from pathlib import Path

import torch

from transformers_from_scratch.checkpointing import load_checkpoint, save_checkpoint
from transformers_from_scratch.data import (
    create_dataloaders,
    load_tinystories_text_splits,
    load_tokenizer,
    tokenize_stories,
)
from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.training import (
    causal_lm_loss,
    evaluate,
    global_grad_norm,
    load_config,
    resolve_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser.parse_args()


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config, model_config, training_config = (
        config["data"],
        config["model"],
        config["training"],
    )
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    print(f"device={device}")

    tokenizer = load_tokenizer(data_config["tokenizer_repo"])
    train_text, val_text = load_tinystories_text_splits(
        data_config["dataset_name"], data_config["train_stories"], data_config["val_stories"]
    )
    train_loader, val_loader = create_dataloaders(
        tokenize_stories(tokenizer, train_text),
        tokenize_stories(tokenizer, val_text),
        context_length=data_config["context_length"],
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        pin_memory=data_config["pin_memory"],
    )
    if not len(train_loader) or not len(val_loader):
        raise ValueError(
            "No full causal windows were created; reduce context length or load more data"
        )

    # Build fresh objects first. A resume below replaces their saved state.
    model = TinyDecoderLM(vocab_size=tokenizer.get_vocab_size(), **model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    # Fresh runs start at global step 0. Resumed runs continue at N + 1.
    start_step = 0
    if args.resume is not None:
        checkpoint = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            device=device,
            model_config=model_config,
            vocab_size=tokenizer.get_vocab_size(),
            tokenizer_repo=data_config["tokenizer_repo"],
        )
        start_step = checkpoint["step"]
        print(f"resuming_from={args.resume}")
        print(f"start_step={start_step}")
    # ``steps`` is the desired final global step, not an additional-step count.
    target_step = training_config["steps"]
    if start_step >= target_step:
        raise ValueError("Checkpoint step is already at or beyond training.steps")

    checkpoint_dir = Path(training_config["checkpoint_dir"])
    checkpoint_interval = training_config["checkpoint_interval"]
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")

    def write_checkpoint(step: int) -> Path:
        path = checkpoint_dir / f"step-{step:06d}.pt"
        return save_checkpoint(
            path,
            step=step,
            model=model,
            optimizer=optimizer,
            model_config=model_config,
            vocab_size=tokenizer.get_vocab_size(),
            tokenizer_repo=data_config["tokenizer_repo"],
            config=config,
        )

    # Keep requesting batches until the configured global target is reached.
    # A newly created iterator after resume does not restore its old shuffle position.
    batches = cycle(train_loader)
    log_start = time.perf_counter()
    log_tokens = 0

    for step in range(start_step + 1, target_step + 1):
        # 1. Fetch one causal-LM batch: inputs and next-token targets are (B, T).
        model.train()
        input_ids, targets = next(batches)

        # 2. Put the token IDs on the same device as the model parameters.
        input_ids, targets = input_ids.to(device), targets.to(device)

        # 3. Forward pass: logits are (B, T, V), then CE reduces them to one loss.
        logits = model(input_ids)
        loss = causal_lm_loss(logits, targets)

        # 4. Clear previous gradients, backpropagate this loss, and inspect their norm.
        optimizer.zero_grad()
        loss.backward()
        grad_norm = global_grad_norm(model)

        # 5. AdamW reads parameter gradients and updates model weights in place.
        optimizer.step()

        # Count actual input tokens for stable interval-level throughput logging.
        log_tokens += input_ids.numel()
        if step % training_config["log_interval"] == 0:
            synchronize_if_cuda(device)
            elapsed = time.perf_counter() - log_start
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={step} train_loss={loss.item():.4f} lr={lr:.2e} "
                f"grad_norm={grad_norm:.4f} tokens_per_sec={log_tokens / elapsed:.0f}"
            )
            log_start, log_tokens = time.perf_counter(), 0
        # Evaluation temporarily switches to eval mode and restores train mode afterward.
        if step % training_config["eval_interval"] == 0:
            metrics = evaluate(model, val_loader, device, training_config["eval_batches"])
            print(
                f"step={step} val_loss={metrics['loss']:.4f} "
                f"val_perplexity={metrics['perplexity']:.2f}"
            )
        # Persist full training state at periodic global steps for future resume.
        if step % checkpoint_interval == 0:
            print(f"checkpoint={write_checkpoint(step)}")

    # If the last periodic save did not land on the target, save the final state once.
    if target_step % checkpoint_interval != 0:
        print(f"checkpoint={write_checkpoint(target_step)}")


if __name__ == "__main__":
    main()
