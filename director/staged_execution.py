"""Execution policies and CPU handoffs for isolated Director model stages."""

from __future__ import annotations

from contextvars import ContextVar

import torch

current_stage = ContextVar("director_execution_stage", default=None)
EXECUTION_MODES = frozenset({"standard", "staged_segment", "staged_batch"})


def execution_mode(plan) -> str:
    output = (plan.raw or {}).get("output") or {}
    mode = str(output.get("executionMode", output.get("execution_mode", "standard")) or "standard")
    if mode not in EXECUTION_MODES:
        raise ValueError(f"Unknown Director execution mode: {mode}")
    return mode


def validate_staged_execution(plan, *, model=None, vae=None, audio_vae=None, clip=None) -> None:
    if plan.refine or plan.face_refine:
        raise ValueError(
            "分阶段执行暂不兼容 Refine / FaceRefine：它们需要穿插采样和解码。"
            "请断开这两个输入，或切回原逐段流程。"
        )
    if plan.selflift and float(plan.selflift.get("rho") or 0) > 1e-8:
        raise ValueError("分阶段执行要求 SelfLift.rho=0；像素锚会在采样中调用 VAE 解码。")
    for name, component in (("MODEL", model), ("video_vae", vae), ("audio_vae", audio_vae), ("CLIP", clip)):
        patcher = getattr(component, "patcher", component)
        offload_device = getattr(patcher, "offload_device", None)
        if offload_device is not None and torch.device(offload_device).type != "cpu":
            raise ValueError(
                f"分阶段执行需要将 {name} 卸载到 CPU，目前卸载设备为 {offload_device}。"
                "请关闭 --highvram / --gpu-only 等强制驻留显存的设置，或将该模型的卸载设备设为 CPU。"
            )


def cpu_handoff(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if getattr(value, "is_nested", False) and hasattr(value, "unbind"):
        return type(value)([cpu_handoff(item) for item in value.unbind()])
    if isinstance(value, dict):
        return {key: cpu_handoff(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_handoff(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_handoff(item) for item in value)
    return value


def advance_segment(generator, *, stage=None, expected=None):
    token = current_stage.set(stage)
    try:
        try:
            boundary = next(generator)
        except StopIteration as complete:
            if expected is not None:
                raise RuntimeError(f"Segment finished before stage boundary {expected!r}.") from complete
            return complete.value
        if expected is None or boundary != expected:
            generator.close()
            raise RuntimeError(f"Unexpected Director stage boundary: {boundary!r}")
    finally:
        current_stage.reset(token)
