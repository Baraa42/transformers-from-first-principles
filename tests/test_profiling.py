import pytest
import torch

import transformers_from_scratch.profiling as profiling
from transformers_from_scratch.profiling import (
    TRAIN_COMPONENTS,
    TimingProfile,
    component_statistics,
    timed_section,
    validate_profile_config,
)


def component_times(value: float) -> dict[str, float]:
    return {name: value for name in TRAIN_COMPONENTS}


def test_timed_section_disabled_executes_without_sync_or_sample(monkeypatch) -> None:
    sync_calls: list[torch.device] = []
    monkeypatch.setattr(profiling, "synchronize_device", sync_calls.append)
    timings: dict[str, float] = {}
    executed = False

    with timed_section("forward_loss", timings, torch.device("cpu"), enabled=False):
        executed = True

    assert executed
    assert sync_calls == []
    assert timings == {}


def test_timed_section_enabled_synchronizes_and_records_elapsed(monkeypatch) -> None:
    sync_calls: list[torch.device] = []
    times = iter((10.0, 10.025))
    monkeypatch.setattr(profiling, "synchronize_device", sync_calls.append)
    monkeypatch.setattr(profiling.time, "perf_counter", lambda: next(times))
    timings: dict[str, float] = {}
    device = torch.device("mps")

    with timed_section("backward", timings, device, enabled=True):
        pass

    assert sync_calls == [device, device]
    assert timings["backward"] == pytest.approx(0.025)


def test_component_statistics_calculates_median_p95_and_share() -> None:
    profile = TimingProfile(enabled=True, warmup_steps=0)
    for value in (0.001, 0.002, 0.003, 0.004):
        times = component_times(value)
        times["forward_loss"] = value * 2
        profile.record_successful_step(times, step_time=value * 9)

    summary = component_statistics(profile)

    assert summary["zero_grad"].median_ms == pytest.approx(2.5)
    assert summary["zero_grad"].p95_ms == pytest.approx(3.85)
    assert summary["forward_loss"].median_ms == pytest.approx(5.0)
    assert sum(item.pct_measured_step for item in summary.values()) == pytest.approx(100.0)


def test_warmup_excludes_first_successful_steps() -> None:
    profile = TimingProfile(enabled=True, warmup_steps=2)

    profile.record_successful_step(component_times(1.0), step_time=8.0)
    profile.record_successful_step(component_times(2.0), step_time=16.0)
    profile.record_successful_step(component_times(3.0), step_time=24.0)

    assert profile.successful_steps == 3
    assert profile.step_samples == [24.0]
    assert all(samples == [3.0] for samples in profile.component_samples.values())


@pytest.mark.parametrize(
    ("enabled", "warmup"),
    [(1, 10), ("true", 10), (True, -1), (True, 1.5), (True, False)],
)
def test_profile_config_validation_rejects_invalid_values(enabled: object, warmup: object) -> None:
    with pytest.raises(ValueError):
        validate_profile_config(enabled, warmup)
