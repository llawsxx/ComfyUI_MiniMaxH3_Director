"""Editable first-pass sigma schedules for MiniMax H3."""

from __future__ import annotations

import math
import re
from typing import Iterable


def _coerce_steps(raw) -> int:
    try:
        steps = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sigma steps must be an integer.") from exc
    if not 1 <= steps <= 200:
        raise ValueError("Sigma steps must be between 1 and 200.")
    return steps


def _coerce_shift(raw, name: str) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not math.isfinite(value) or not 0.01 <= value <= 100.0:
        raise ValueError(f"{name} must be between 0.01 and 100.")
    return value


def validate_sigma_values(
    values: Iterable[float],
    *,
    expected_steps: int | None = None,
    expected_count: int | None = None,
) -> tuple[float, ...]:
    """Validate an H3 flow schedule without silently changing user input."""
    vals = tuple(float(value) for value in values)
    if len(vals) < 2:
        raise ValueError("Sigma 表至少需要 2 个数（起点和结尾 0）。")
    wanted = int(expected_count) if expected_count is not None else (
        int(expected_steps) + 1 if expected_steps is not None else None
    )
    if wanted is not None and len(vals) != wanted:
        configured = int(expected_steps) if expected_steps is not None else wanted - 1
        raise ValueError(
            f"当前步数/调度器的原生 Sigma 表有 {wanted} 个数"
            f"（设置步数 {configured}）；现在有 {len(vals)} 个。"
        )

    for index, value in enumerate(vals):
        if not math.isfinite(value):
            raise ValueError(f"Sigma[{index}] 不是有限数值。")
        if value < 0.0 or value > 1.0 + 1e-7:
            raise ValueError(f"Sigma[{index}]={value:g} 超出 H3 有效范围 0..1。")
        if index and value > vals[index - 1] + 1e-7:
            raise ValueError(
                f"Sigma 必须从大到小：Sigma[{index - 1}]={vals[index - 1]:g}，"
                f"Sigma[{index}]={value:g}。"
            )

    if vals[0] <= 0.0:
        raise ValueError("Sigma[0] 必须大于 0。")
    if abs(vals[-1]) > 1e-7:
        raise ValueError(f"最后一个 Sigma 必须为 0；现在是 {vals[-1]:g}。")
    if any(value <= 1e-8 for value in vals[:-1]):
        raise ValueError("只有 Sigma 表的最后一个数可以为 0。")
    return tuple(0.0 if abs(value) <= 1e-8 else value for value in vals)


def parse_sigma_values(
    raw,
    *,
    expected_steps: int | None = None,
    expected_count: int | None = None,
) -> tuple[float, ...]:
    """Parse comma/space/newline separated values from the Director editor."""
    text = str(raw or "").strip()
    if not text:
        raise ValueError("Sigma 编辑框为空。")
    tokens = [token for token in re.split(r"[\s,;]+", text) if token]
    values: list[float] = []
    for index, token in enumerate(tokens):
        try:
            values.append(float(token))
        except ValueError as exc:
            raise ValueError(f"Sigma 编辑框第 {index + 1} 项不是数值：{token!r}。") from exc
    return validate_sigma_values(
        values,
        expected_steps=expected_steps,
        expected_count=expected_count,
    )


def audio_sigma_values(
    video_sigmas: Iterable[float], shift_video: float, shift_audio: float
) -> tuple[float, ...]:
    """Map the shared video schedule to H3's internal audio schedule."""
    shift_v = _coerce_shift(shift_video, "shift_video")
    shift_a = _coerce_shift(shift_audio, "shift_audio")
    out: list[float] = []
    for raw in video_sigmas:
        sigma = float(raw)
        base = sigma / (shift_v + sigma * (1.0 - shift_v))
        out.append(shift_a * base / (1.0 + (shift_a - 1.0) * base))
    return tuple(out)


def calculate_h3_sigma_schedule(
    steps: int, scheduler: str, shift_video: float, shift_audio: float
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Run ComfyUI's exact BasicScheduler math without loading a MODEL."""
    import comfy.model_sampling
    import comfy.samplers

    step_count = _coerce_steps(steps)
    shift_v = _coerce_shift(shift_video, "shift_video")
    shift_a = _coerce_shift(shift_audio, "shift_audio")
    scheduler_name = str(scheduler or "simple")
    if scheduler_name not in comfy.samplers.SCHEDULER_NAMES:
        raise ValueError(f"未知 scheduler：{scheduler_name}。")

    class H3ModelSampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
        pass

    model_sampling = H3ModelSampling(None)
    model_sampling.set_parameters(shift=shift_v, audio_shift=shift_a)
    tensor = comfy.samplers.calculate_sigmas(model_sampling, scheduler_name, step_count)
    video = validate_sigma_values(
        float(value) for value in tensor.detach().float().cpu().reshape(-1).tolist()
    )
    return video, audio_sigma_values(video, shift_v, shift_a)


def format_sigma_values(values: Iterable[float]) -> str:
    """One value per line keeps each denoise boundary easy to edit."""
    return "\n".join(f"{float(value):.9g}" for value in values)


def repeated_sigma_indices(values: Iterable[float]) -> list[int]:
    vals = tuple(float(value) for value in values)
    return [index for index in range(1, len(vals)) if abs(vals[index] - vals[index - 1]) <= 1e-8]
