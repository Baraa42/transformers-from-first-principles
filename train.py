"""Train the tiny decoder from a plain YAML configuration file."""

import argparse
import time
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
    autocast_context,
    causal_lm_loss,
    clip_or_measure_grad_norm,
    create_grad_scaler,
    evaluate,
    infinite_batches,
    load_config,
    resolve_device,
    scaler_step_was_skipped,
    set_seed,
    validate_grad_accum_steps,
    validate_precision,
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
    precision = validate_precision(training_config.get("precision", "fp32"))
    scaler = create_grad_scaler(device, precision)

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
            scaler=scaler,
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
        "scaler": scaler,
    }

    # 10. Create the training-batch iterator and counters used for interval metrics.
    # A new iterator after resume does not restore its old shuffle position.
    batches = infinite_batches(train_loader)
    log_start = time.perf_counter()
    log_tokens = 0

    # 11. Each completed outer iteration is one successful optimizer/global step.
    step = start_step
    while step < target_step:
        model.train()
        optimizer.zero_grad()

        # 11.1 Accumulate N scaled or unscaled microbatch gradients.
        step_loss = 0.0
        for _ in range(grad_accum_steps):
            input_ids, targets = next(batches)
            input_ids, targets = input_ids.to(device), targets.to(device)
            with autocast_context(device, precision):
                loss = causal_lm_loss(model(input_ids), targets)
            step_loss += loss.item()
            scaled_loss = loss / grad_accum_steps
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()
            log_tokens += input_ids.numel()

        # 11.2 Unscale once, then measure/clip the real accumulated gradient once.
        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = clip_or_measure_grad_norm(model, max_grad_norm)
        grad_clipped = max_grad_norm is not None and grad_norm > max_grad_norm

        # 11.3 FP16 overflow is detected by public scale backoff after step/update.
        if scaler is None:
            optimizer.step()
        else:
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler_step_was_skipped(old_scale, scaler.get_scale()):
                print(f"amp_overflow=true scale={scaler.get_scale():.0f}")
                continue

        step += 1

        # 11.4 Log mean unscaled loss and all consumed microbatch tokens.
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
        if step % training_config["eval_interval"] == 0:
            metrics = evaluate(
                model, val_loader, device, training_config["eval_batches"], precision
            )
            print(
                f"step={step} val_loss={metrics['loss']:.4f} "
                f"val_perplexity={metrics['perplexity']:.2f}"
            )
        if step % checkpoint_interval == 0:
            print(f"checkpoint={save_training_checkpoint(step=step, **checkpoint_args)}")

    # 12. If no periodic save landed on the target, save the final state once.
    if target_step % checkpoint_interval != 0:
        print(f"checkpoint={save_training_checkpoint(step=target_step, **checkpoint_args)}")


if __name__ == "__main__":
    main()
