"""Lock the target H3 audio stream to an independently selected soundtrack."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import torch


def drive_audio_settings(plan):
    output = (plan.raw or {}).get("output") or {}
    media = output.get("driveAudio", output.get("drive_audio")) or {}
    offset = float(output.get("driveAudioOffsetSec", output.get("drive_audio_offset_sec", 0)) or 0)
    if not math.isfinite(offset) or offset < 0:
        raise ValueError("驱动音频起点必须是非负秒数。")
    return media, offset


def drive_audio_identity(plan):
    from .plan import _reference_audio_file

    output = (plan.raw or {}).get("output") or {}
    if str(output.get("audioMode", output.get("audio_mode", ""))).strip().lower() not in {"drive", "audio_drive"}:
        return None
    media, offset = drive_audio_settings(plan)
    relative, filename = _reference_audio_file(media)
    try:
        stat = Path(filename).stat()
        identity = [relative, stat.st_size, stat.st_mtime_ns]
    except OSError:
        identity = [relative, "missing"]
    return {"version": 1, "audio": identity, "offset_sec": offset}


def _wave_window(waveform, start, length):
    result = waveform.new_zeros((*waveform.shape[:-1], max(0, int(length))))
    source_start = max(0, int(start))
    destination_start = max(0, -int(start))
    count = min(waveform.shape[-1] - source_start, result.shape[-1] - destination_start)
    if count > 0:
        result[..., destination_start:destination_start + count] = waveform[..., source_start:source_start + count]
    return result


class AudioDriveSession:
    def __init__(self, plan, audio_vae):
        from .plan import load_reference_audio_item

        try:
            from comfy.ldm.minimax.model import MiniMaxH3Model
            from comfy.model_base import MiniMaxH3
        except ImportError as exc:
            raise ValueError("当前 ComfyUI 缺少 H3 音频驱动支持，请升级 ComfyUI。") from exc
        if audio_vae is None:
            raise ValueError("音频驱动需要连接 MiniMax H3 audio_vae。")
        if not math.isclose(float(plan.frame_rate), 24.0, abs_tol=1e-6):
            raise ValueError("音频驱动要求 24 fps（H3 原生音视频时间轴），请将输出帧率设为 24。")
        if ("audio_denoise_mask" not in inspect.signature(MiniMaxH3Model.forward).parameters
                or not callable(getattr(MiniMaxH3, "_denoise_mask_values", None))):
            raise ValueError("当前 ComfyUI 缺少 H3 目标音频遮罩支持，请升级 ComfyUI 后使用音频驱动。")
        if plan.refine or plan.face_refine or plan.selflift:
            raise ValueError("音频驱动暂不兼容 Refine / FaceRefine / SelfLift，请断开这些节点后运行。")
        media, self.offset = drive_audio_settings(plan)
        if not isinstance(media, dict) or not (media.get("audioFile") or media.get("fileName")):
            raise ValueError("已选择音频驱动，请单独选择一份驱动音频。")
        audio = load_reference_audio_item(media)
        if not isinstance(audio, dict) or not torch.is_tensor(audio.get("waveform")) or audio["waveform"].numel() == 0:
            raise ValueError("驱动音频不存在或无法解码，请重新选择音频文件。")
        waveform = audio["waveform"].detach().cpu()
        self.sample_rate = int(audio["sample_rate"])
        if waveform.ndim != 3 or waveform.shape[0] != 1 or self.sample_rate <= 0:
            raise ValueError("驱动音频必须是有效的单批次 ComfyUI AUDIO。")
        if self.offset * self.sample_rate >= waveform.shape[-1]:
            raise ValueError("驱动音频起点已超过音频长度。")
        self.waveform = waveform
        self.plan = plan
        self.encoded = None
        self.starts = {}
        self.leading_seconds = max(3.0, float(plan.continuity_overlap_frames or 0) / 24.0 + 1.0)
        self.latent_rate = 40.0

    def prepare(self, audio_vae):
        if self.encoded is not None:
            return
        import torchaudio

        rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        compression = int(getattr(audio_vae, "downscale_ratio", 800))
        if rate != 32000 or compression != 800:
            raise ValueError("音频驱动需要 MiniMax H3 Audio VAE（32000 Hz / 800 倍压缩）。")
        selected = [seg for seg in self.plan.segments
                    if self.plan.run_indices is None or seg.index in self.plan.run_indices]
        end_frame = max((seg.end_frame for seg in selected), default=0)
        extra = len(self.plan.segments) * (int(self.plan.continuity_overlap_frames or 0) + 17)
        seconds = (end_frame + extra) / float(self.plan.frame_rate) + 3.0
        offset_samples = round(self.offset * self.sample_rate)
        waveform = self.waveform[..., offset_samples:offset_samples + math.ceil(seconds * self.sample_rate)]
        waveform = waveform.float()
        if waveform.shape[1] == 1:
            waveform = waveform.repeat(1, 2, 1)
        elif waveform.shape[1] != 2:
            raise ValueError("驱动音频仅支持单声道或立体声。")
        if self.sample_rate != rate:
            waveform = torchaudio.functional.resample(waveform, self.sample_rate, rate)
        leading = round(self.leading_seconds * rate)
        trailing = max(0, math.ceil(seconds * rate) - waveform.shape[-1]) + round(3.0 * rate)
        waveform = torch.nn.functional.pad(waveform, (leading, trailing))
        encoded = audio_vae.encode(waveform.movedim(1, -1))
        if not torch.is_tensor(encoded) or encoded.ndim != 4 or encoded.shape[1:3] != (32, 2):
            raise ValueError("驱动音频编码未返回 H3 [B,32,2,T] latent。")
        self.encoded = encoded.detach().cpu()
        self.latent_rate = rate / compression

    def start_segment(self, seg, previous_export_frames=None, previous_start=None):
        previous_start = self.starts.get(seg.index - 1, previous_start)
        start = float(seg.start_frame)
        if previous_start is not None and previous_export_frames is not None:
            start = float(previous_start) + int(previous_export_frames)
        self.starts[seg.index] = start
        return start

    def lock(self, latent, *, start_frame, trim_frames):
        from comfy.nested_tensor import NestedTensor

        if self.encoded is None:
            raise RuntimeError("驱动音频尚未完成编码。")
        samples = latent.get("samples")
        if not getattr(samples, "is_nested", False):
            raise ValueError("音频驱动需要 H3 联合 AV latent。")
        video, audio = samples.unbind()
        seconds = (float(start_frame) - int(trim_frames)) / float(self.plan.frame_rate)
        start = round((seconds + self.leading_seconds) * self.latent_rate)
        driven = _wave_window(self.encoded, start, audio.shape[-1]).to(audio)
        if driven.shape != audio.shape:
            raise ValueError("驱动音频 latent 与目标音频形状不匹配。")
        mask = latent.get("noise_mask")
        if mask is None:
            video_mask = torch.ones_like(video)
        elif getattr(mask, "is_nested", False):
            video_mask = mask.unbind()[0].clone()
        else:
            raise ValueError("音频驱动遇到不兼容的 AV noise_mask。")
        output = dict(latent)
        output["samples"] = NestedTensor((video, driven))
        output["noise_mask"] = NestedTensor((video_mask, torch.zeros_like(audio)))
        return output

    def export_audio(self, seg, frame_count):
        start = self.starts.get(seg.index, float(seg.start_frame))
        first = round((self.offset + start / float(self.plan.frame_rate)) * self.sample_rate)
        count = round(int(frame_count) / float(self.plan.frame_rate) * self.sample_rate)
        return {"waveform": _wave_window(self.waveform, first, count), "sample_rate": self.sample_rate}

    def clear(self):
        self.encoded = None
        self.waveform = None
