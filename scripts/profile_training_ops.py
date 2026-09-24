"""Profile a short baseline training run at the PyTorch operator level."""

import argparse
from pathlib import Path

import torch

from transformers_from_scratch.data import (
    create_dataloaders,
    load_tinystories_text_splits,
    load_tokenizer,
    tokenize_stories,
)
from transformers_from_scratch.model import TinyDecoderLM
from transformers_from_scratch.profiler_tools import (
    profiler_activities,
    trace_output_path,
    validate_profiler_steps,
)
from transformers_from_scratch.training import (
    autocast_context,
    causal_lm_loss,
    create_adamw_optimizer,
    infinite_batches,
    load_config,
    optimizer_step,
    prepare_gradients,
    resolve_device,
    set_seed,
    synchronize_device,
    validate_precision,
    validate_precision_support,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "benchmarks" / "profile-fp32.yaml"
DEFAULT_TRACE_PATH = Path("benchmarks/profiler/trace.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--profile-steps", type=int, default=15)
    parser.add_argument("--top-ops", type=int, default=20)
    parser.add_argument(
        "--export-trace",
        action="store_true",
        help="Export a Chrome trace under benchmarks/profiler/",
    )
    parser.add_argument("--trace-path", default=str(DEFAULT_TRACE_PATH))
    args = parser.parse_args()
    try:
        validate_profiler_steps(args.warmup_steps, args.profile_steps)
    except ValueError as error:
        parser.error(str(error))
    if args.top_ops < 1:
        parser.error("top_ops must be >= 1")
    return args


def run_training_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batches,
    device: torch.device,
    precision: str,
    max_grad_norm: float | None,
) -> float:
    """Run one real FP32 optimizer step and return its scalar loss."""
    model.train()
    optimizer.zero_grad()
    input_ids, targets = next(batches)
    input_ids, targets = input_ids.to(device), targets.to(device)
    with autocast_context(device, precision):
        loss = causal_lm_loss(model(input_ids), targets)
    loss_value = loss.item()
    loss.backward()
    prepare_gradients(
        model=model,
        optimizer=optimizer,
        precision=precision,
        scaler=None,
        max_grad_norm=max_grad_norm,
    )
    did_step = optimizer_step(precision=precision, optimizer=optimizer, scaler=None)
    if not did_step:
        raise RuntimeError("FP32 optimizer step was unexpectedly skipped")
    return loss_value


def main() -> None:
    args = parse_args()
    config = load_config(CONFIG_PATH)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]

    device = resolve_device(config["device"])
    precision = validate_precision(training_config["precision"])
    if precision != "fp32":
        raise ValueError("Operator profiler baseline must use precision=fp32")
    if training_config.get("grad_accum_steps", 1) != 1:
        raise ValueError("Operator profiler baseline requires grad_accum_steps=1")
    validate_precision_support(device, precision)
    set_seed(config["seed"])

    tokenizer = load_tokenizer(data_config["tokenizer_repo"])
    train_text, val_text = load_tinystories_text_splits(
        data_config["dataset_name"],
        data_config["train_stories"],
        data_config["val_stories"],
    )
    train_loader, _ = create_dataloaders(
        tokenize_stories(tokenizer, train_text),
        tokenize_stories(tokenizer, val_text),
        context_length=data_config["context_length"],
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        pin_memory=data_config["pin_memory"],
    )
    if not len(train_loader):
        raise ValueError("Training loader produced no full batches")
    batches = infinite_batches(train_loader)

    model = TinyDecoderLM(vocab_size=tokenizer.get_vocab_size(), **model_config).to(device)
    optimizer = create_adamw_optimizer(
        model.parameters(),
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        device=device,
    )
    step_args = {
        "model": model,
        "optimizer": optimizer,
        "batches": batches,
        "device": device,
        "precision": precision,
        "max_grad_norm": training_config.get("max_grad_norm"),
    }

    print(f"device={device.type}")
    print(f"warmup_steps={args.warmup_steps}")
    print(f"profile_steps={args.profile_steps}")
    print(f"context_length={data_config['context_length']} batch_size={data_config['batch_size']}")

    for _ in range(args.warmup_steps):
        run_training_step(**step_args)
    synchronize_device(device)

    activities = profiler_activities(device)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        final_loss = 0.0
        for _ in range(args.profile_steps):
            final_loss = run_training_step(**step_args)
            profiler.step()
    synchronize_device(device)

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    activity_names = ",".join(str(activity).rsplit(".", 1)[-1].lower() for activity in activities)
    print(f"profiler_activities={activity_names}")
    if device.type == "mps":
        print("mps_device_activity=unavailable; table reports CPU-side operator activity")
    print(f"sort_by={sort_key}")
    print(f"final_profiled_loss={final_loss:.4f}")
    print()
    print(
        profiler.key_averages(group_by_input_shape=True).table(
            sort_by=sort_key,
            row_limit=args.top_ops,
        )
    )

    if args.export_trace:
        trace_path = trace_output_path(PROJECT_ROOT, args.trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(trace_path))
        print(f"trace={trace_path}")

    print(
        "diagnostic_goal=validate compute, backward, gradient/optimizer, "
        "and synchronization patterns from Stage 5.1"
    )
    print(
        "inspect_operator_families=mm/matmul/bmm/addmm, softmax/masking, "
        "normalization, loss/backward, AdamW, norm/foreach, local_scalar/copy/wait"
    )


if __name__ == "__main__":
    main()
