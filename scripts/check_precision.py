"""Report real local autocast and GradScaler capabilities."""

import argparse
from collections.abc import Callable

import torch

from transformers_from_scratch.training import (
    resolve_device,
    validate_autocast_support,
    validate_grad_scaler_support,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help="Backend to inspect (default: auto)",
    )
    return parser.parse_args()


def report(name: str, check: Callable[[], None]) -> None:
    try:
        check()
    except ValueError as error:
        print(f"{name}=unsupported: {error}")
    else:
        print(f"{name}=supported")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"torch_version={torch.__version__}")
    print(f"device={device.type}")
    print()
    report("fp32", lambda: validate_autocast_support(device, "fp32"))
    report("fp16_autocast", lambda: validate_autocast_support(device, "fp16"))
    report("fp16_grad_scaler", lambda: validate_grad_scaler_support(device))
    report("bf16_autocast", lambda: validate_autocast_support(device, "bf16"))


if __name__ == "__main__":
    main()
