"""Train the tiny decoder from a plain YAML configuration file."""

import argparse
import time
from itertools import cycle
from pathlib import Path

import torch

from transformers_from_scratch.checkpointing import load_checkpoint, save_training_checkpoint
from transformers_from_scratch.data import (
    create_dataloaders,
    load_tinystories_text_splits,
    load_tokenizer,
    tokenize_stories,
)
from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.training import (
    causal_lm_loss,
    clip_or_measure_grad_norm,
    evaluate,
    load_config,
    resolve_device,
    set_seed,
    validate_grad_accum_steps,
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
    # 1. Parse the CLI so we know which config and optional checkpoint to use.
    args = parse_args()

    # 2. Load the plain YAML configuration and name its three major sections.
    config = load_config(args.config)
    data_config, model_config, training_config = (
        config["data"],
        config["model"],
        config["training"],
    )
    # 3. Seed global randomness and select CPU, MPS, or CUDA.
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    print(f"device={device}")

    # 4. Load the fixed tokenizer so its vocabulary can be checked on resume.
    tokenizer = load_tokenizer(data_config["tokenizer_repo"])

    # 5. Build fresh model/optimizer objects. A resume below replaces their saved state.
    model = TinyDecoderLM(vocab_size=tokenizer.get_vocab_size(), **model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
    )
    # 6. Optionally restore model, AdamW, global step, and RNG from a checkpoint.
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
    # 7. Confirm there is work left: ``steps`` is a final global step, not extra steps.
    target_step = training_config["steps"]
    if start_step >= target_step:
        raise ValueError("Checkpoint step is already at or beyond training.steps")

    # Validate clipping early; null leaves existing gradient behavior unchanged.
    max_grad_norm = training_config.get("max_grad_norm")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be strictly positive or null")

    grad_accum_steps = validate_grad_accum_steps(training_config.get("grad_accum_steps", 1))
    # 8. Load/tokenize data only after resume compatibility and RNG restoration succeed.
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

    # 9. Configure periodic checkpoint naming and its explicit save arguments.
    checkpoint_dir = Path(training_config["checkpoint_dir"])
    checkpoint_interval = training_config["checkpoint_interval"]
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_interval must be positive")

    checkpoint_args = {
        "checkpoint_dir": checkpoint_dir,
        "model": model,
        "optimizer": optimizer,
        "model_config": model_config,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_repo": data_config["tokenizer_repo"],
        "config": config,
    }

    # 10. Create the training-batch iterator and counters used for interval metrics.
    # A new iterator after resume does not restore its old shuffle position.
    batches = cycle(train_loader)
    log_start = time.perf_counter()
    log_tokens = 0

    # 11. Run steps start_step + 1 through target_step, inclusive.
    for step in range(start_step + 1, target_step + 1):
        # 11.1 Each outer-loop iteration is one optimizer/global step.
        model.train()
        optimizer.zero_grad()

        # 11.2 Accumulate scaled gradients from N microbatches without zeroing between them.
        step_loss = 0.0
        for _ in range(grad_accum_steps):
            input_ids, targets = next(batches)
            input_ids, targets = input_ids.to(device), targets.to(device)

            # 11.3 Forward pass: logits are (B, T, V), then CE reduces them to one loss.
            loss = causal_lm_loss(model(input_ids), targets)
            step_loss += loss.item()
            (loss / grad_accum_steps).backward()
            log_tokens += input_ids.numel()

        # 11.4 Measure/clip the final accumulated gradient once, before the update.
        grad_norm = clip_or_measure_grad_norm(model, max_grad_norm)
        grad_clipped = max_grad_norm is not None and grad_norm > max_grad_norm

        # 11.5 AdamW reads the (possibly clipped) gradients and updates weights in place.
        optimizer.step()

        # 11.6 Log mean unscaled microbatch loss and all processed input tokens.
        if step % training_config["log_interval"] == 0:
            synchronize_if_cuda(device)
            elapsed = time.perf_counter() - log_start
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"step={step} train_loss={step_loss / grad_accum_steps:.4f} lr={lr:.2e} "
                f"grad_norm={grad_norm:.4f} grad_clipped={grad_clipped} "
                f"tokens_per_sec={log_tokens / elapsed:.0f}"
            )
            log_start, log_tokens = time.perf_counter(), 0
        # 11.7 Evaluate periodically; evaluate() restores training mode afterward.
        if step % training_config["eval_interval"] == 0:
            metrics = evaluate(model, val_loader, device, training_config["eval_batches"])
            print(
                f"step={step} val_loss={metrics['loss']:.4f} "
                f"val_perplexity={metrics['perplexity']:.2f}"
            )
        # 11.8 Persist full state at periodic global steps for future resume.
        if step % checkpoint_interval == 0:
            print(f"checkpoint={save_training_checkpoint(step=step, **checkpoint_args)}")

    # 12. If no periodic save landed on the target, save the final state once.
    if target_step % checkpoint_interval != 0:
        print(f"checkpoint={save_training_checkpoint(step=target_step, **checkpoint_args)}")


if __name__ == "__main__":
    main()
