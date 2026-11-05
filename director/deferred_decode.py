"""Persistent sampled AV latents and independent, sampling-free video decode."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from .audio_export import (
    AUDIO_MODE_GENERATE,
    AUDIO_MODE_SOURCE,
    _merge_generated_segment_audios,
    _pad_or_trim_audio_to_frames,
    empty_audio_dict,
    prepare_segment_audio_for_file_export,
    resolve_audio_mode,
)
from .h3_motion_context import _repack_av_streams, _streams_from_latent, av_pixel_size

FORMAT = "minimax-h3-director-latents"
VERSION = 1
log = logging.getLogger("ComfyUI-MiniMaxH3-Director.deferred_decode")


def deferred_decode_enabled(plan) -> bool:
    output = (plan.raw or {}).get("output") or {}
    return output.get("deferVaeDecode", output.get("defer_vae_decode", False)) in (
        True, 1, "1", "true", "True", "on",
    )


def save_sampled_latents_enabled(plan) -> bool:
    output = (plan.raw or {}).get("output") or {}
    return output.get("saveSampledLatents", output.get("save_sampled_latents", False)) in (
        True, 1, "1", "true", "True", "on",
    )


def _new_run_directory(parent: Path) -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root = parent / stamp
    root.mkdir(exist_ok=False)
    return root


def _atomic_write(path: Path, writer) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, data: dict) -> None:
    _atomic_write(path, lambda target: target.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
    ))


def _child_path(root: Path, name: str) -> Path:
    path = (root / str(name)).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError(f"Latent manifest path escapes its task folder: {name}")
    return path


def _save_tensors(path: Path, tensors: dict) -> None:
    cpu = {key: value.detach().cpu().contiguous().clone() for key, value in tensors.items()}
    _atomic_write(path, lambda target: save_file(cpu, str(target)))


def load_av_latent(path: Path) -> dict:
    tensors = load_file(str(path), device="cpu")
    video, audio = tensors.get("video"), tensors.get("audio")
    if video is None or video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
        raise ValueError(f"Not a single-batch H3 video latent: {path}")
    if audio is None or audio.ndim not in (3, 4):
        raise ValueError(f"Missing or invalid H3 audio latent: {path}")
    return {"samples": _repack_av_streams([video, audio])}


def _decode_video(samples: dict, vae, *, tiled: bool, tile_size: int, temporal_size: int):
    from nodes import VAEDecode, VAEDecodeTiled

    if tiled:
        images, = VAEDecodeTiled().decode(
            vae, samples, tile_size=int(tile_size), temporal_size=int(temporal_size),
        )
    else:
        images, = VAEDecode().decode(vae, samples)
    return images.detach().cpu().float()


def _decode_audio(samples: dict, audio_vae) -> dict:
    from .executor_core import _unpack_node_output

    try:
        from comfy_extras.nodes_audio import VAEDecodeAudio
    except ImportError:
        from comfy_extras.nodes_lt import VAEDecodeAudio
    audio = _unpack_node_output(VAEDecodeAudio.execute(audio_vae, samples))[0]
    return {"waveform": audio["waveform"].detach().cpu(), "sample_rate": int(audio["sample_rate"])}


def _crop_export(frames, audio, record: dict, fps: float):
    from .executor_core import _trim_decoded_to_export

    trim_frames, export_frames = int(record["trim_frames"]), int(record["export_frames"])
    if trim_frames < 0 or export_frames < 1 or frames.shape[0] < trim_frames + export_frames:
        raise ValueError(
            f"Decoded {frames.shape[0]} frames, but segment needs "
            f"{trim_frames} context + {export_frames} export frames. Check the video VAE."
        )
    return _trim_decoded_to_export(
        frames, audio, trim_frames=trim_frames, export_len=export_frames,
        plan=SimpleNamespace(frame_rate=fps),
    )


class LatentSaveRun:
    def __init__(self, plan, *, root: Path | None = None):
        import folder_paths
        output = (plan.raw or {}).get("output") or {}
        requested = str(output.get("latentOutputDir", output.get("latent_output_dir", "")) or "").strip()
        base = Path(folder_paths.get_output_directory())
        parent = Path(requested).expanduser() if requested else Path("minimax_seg_export")
        if not parent.is_absolute():
            parent = base / parent
        self.root = Path(root).resolve() if root is not None else _new_run_directory(parent.resolve())
        self.plan = plan
        self.manifest = {
            "format": FORMAT, "version": VERSION, "status": "sampling",
            "frame_rate": float(plan.frame_rate), "width": int(plan.width), "height": int(plan.height),
            "audio_mode": resolve_audio_mode(plan),
            "continuity_enabled": bool(plan.continuity_enabled),
            "continuity_grade_align": bool(plan.continuity_grade_align),
            "preview_decoder": None,
            "latent_stage": "first_pass",
            "selected_indices": [int(seg.timeline_index) for seg in plan.segments
                                 if plan.run_indices is None or seg.index in plan.run_indices],
            "segments": [],
        }
        self._publish()
        log.info("AV latent task folder: %s", self.root)

    def _publish(self):
        _write_json(self.root / "manifest.json", self.manifest)

    def validate_context(self, target: dict, context: dict | None):
        if context is None or av_pixel_size(target) != av_pixel_size(context):
            raise ValueError(
                "延后解码的段间引导需要同分辨率的上一段原始 AV latent，"
                "不会使用 TAE 预览图重新编码。请一起运行相邻段，或关闭该段的引用上段。"
            )

    def save_sample(self, seg, samples: dict, *, trim_frames: int, export_frames: int, sample_frames: int):
        video, audio = _streams_from_latent(samples)
        if video.ndim != 5 or video.shape[0] != 1 or video.shape[1] != 24:
            raise ValueError("保存 latent 需要单批次 H3 24 通道视频 latent。")
        name = f"seg_{seg.timeline_index + 1:04d}"
        latent_name = name + ".safetensors"
        _save_tensors(self.root / latent_name, {"video": video, "audio": audio})
        timeline = self.plan.raw or {}
        raw_segments = (timeline.get("shots") if timeline.get("timelineMode") == "fl2v" else timeline.get("segments")) or []
        raw = next((item for index, item in enumerate(raw_segments)
                    if index == seg.timeline_index and isinstance(item, dict)), {})
        record = {
            "index": int(seg.index), "timeline_index": int(seg.timeline_index),
            "title": str(raw.get("title") or raw.get("name") or name),
            "prompt": str(seg.prompt), "task_key": str(seg.task_key),
            "seed": int(self.plan.sample_seed), "latent": latent_name,
            "trim_frames": int(trim_frames), "export_frames": int(export_frames),
            "sample_frames": int(sample_frames),
            "continuity_from_prev": bool(seg.continuity_from_prev),
            "preview": None,
        }
        self.manifest["segments"].append(record)
        self._publish()
        log.info("Segment #%d raw AV latent saved: %s", seg.timeline_index + 1, self.root / latent_name)
        if self.manifest["audio_mode"] == AUDIO_MODE_SOURCE:
            source = prepare_segment_audio_for_file_export(
                self.plan, seg, audio_dict=None, frame_count=export_frames,
            )
            if source is not None:
                record["source_audio"] = name + "_source_audio.safetensors"
                record["source_sample_rate"] = int(source["sample_rate"])
                _save_tensors(self.root / record["source_audio"], {"waveform": source["waveform"]})
        self._publish()
        return record

    def update_export_length(self, index: int, frame_count: int):
        record = next((item for item in self.manifest["segments"] if item["index"] == index), None)
        if record is not None:
            record["export_frames"] = int(frame_count)
            self._publish()

    def trim_previous(self, index: int, frames):
        self.update_export_length(index, int(frames.shape[0]))

    def finish(self):
        self.manifest["status"] = "complete"
        self._publish()


class DeferredDecodeRun(LatentSaveRun):
    def __init__(self, plan):
        if plan.refine or plan.face_refine:
            raise ValueError("延后 VAE 解码暂不兼容导演台上的 Refine / FaceRefine，请断开它们后再运行。")
        if plan.selflift and float(plan.selflift.get("rho") or 0) > 1e-8:
            raise ValueError("延后解码要求 SelfLift.rho=0；像素锚混合会在采样中调用正式 VAE 解码。")
        super().__init__(plan)


def _available_output_path(root: Path, name: str) -> Path:
    path = root / name
    if not path.exists():
        return path
    for suffix in range(1, 1000000):
        candidate = path.with_name(f"{path.stem}_{suffix:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise ValueError(f"Too many decoded output files: {path}")


def read_manifest(folder: str) -> tuple[Path, dict]:
    import folder_paths

    root = Path(str(folder).strip()).expanduser()
    if not root.is_absolute():
        root = Path(folder_paths.get_output_directory()) / root
    manifest_path = root if root.name == "manifest.json" else root / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent.resolve()
    if data.get("format") != FORMAT or data.get("version") != VERSION:
        raise ValueError("不是受支持的 MiniMax H3 Director latent 任务文件夹。")
    fps = float(data.get("frame_rate", 0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid frame rate in latent manifest.")
    records = data.get("segments")
    if not isinstance(records, list) or not records:
        raise ValueError("任务中尚没有已保存的 latent。")
    seen = set()
    for record in records:
        index = int(record["timeline_index"])
        if index < 0 or index in seen:
            raise ValueError("Invalid or duplicate segment index in latent manifest.")
        seen.add(index)
        if int(record["trim_frames"]) < 0 or int(record["export_frames"]) < 1:
            raise ValueError("Invalid context/export frame count in latent manifest.")
        for key in ("latent", "source_audio"):
            if record.get(key):
                _child_path(root, record[key])
    return root, data


def selected_records(manifest: dict, selection: str) -> list[dict]:
    records = sorted(manifest["segments"], key=lambda record: int(record["timeline_index"]))
    selection = str(selection or "").strip()
    if not selection or selection.lower() == "all":
        return records
    wanted = set()
    for token in re.split(r"[,，\s]+", selection):
        match = re.fullmatch(r"([1-9]\d*)(?:-([1-9]\d*))?", token)
        if not match:
            raise ValueError("分段选择格式：留空 / all，或 1,3-5（从 1 开始）。")
        start, end = int(match[1]), int(match[2] or match[1])
        if end < start or end - start > 100000:
            raise ValueError("Invalid segment selection range.")
        wanted.update(range(start - 1, end))
    available = {int(record["timeline_index"]) for record in records}
    if wanted - available:
        raise ValueError(f"任务中没有这些分段的 latent：{sorted(index + 1 for index in wanted - available)}")
    return [record for record in records if int(record["timeline_index"]) in wanted]


def decode_latent_run(
    folder: str, *, video_vae, audio_vae=None, selection: str = "", output_directory: str = "",
    tiled: bool = False, tile_size: int = 512, temporal_size: int = 64,
    merge_video: bool = False, return_images: bool = False,
):
    import folder_paths
    import comfy.model_management
    import comfy.utils
    from ..lib.video_export import write_frames_to_mp4
    from .segment_continuity import match_export_opening_grade, concat_continuous_chunks

    root, manifest = read_manifest(folder)
    records = selected_records(manifest, selection)
    mode, fps = manifest["audio_mode"], float(manifest["frame_rate"])
    if mode == AUDIO_MODE_GENERATE and audio_vae is None:
        raise ValueError("该任务包含生成音频，请连接 audio_vae；不会静默丢弃音频。")
    destination = root
    if output_directory.strip():
        base = Path(output_directory.strip()).expanduser()
        if not base.is_absolute():
            base = Path(folder_paths.get_output_directory()) / base
        destination = _new_run_directory(base.resolve())
    decode_manifest = _available_output_path(destination, "decode.json")
    report = [f"Latent task: {root}", f"Decoded output: {destination}"]
    if manifest.get("status") != "complete":
        report.append("采样任务未全部完成；仅解码已经落盘的分段。")
    frames_out, audios_out, merge_chunks, merge_audios, counts = [], [], [], [], []
    previous_frames, previous_index = None, None
    outputs = []
    progress = comfy.utils.ProgressBar(len(records))
    for position, record in enumerate(records):
        comfy.model_management.throw_exception_if_processing_interrupted()
        index = int(record["timeline_index"])
        samples = load_av_latent(_child_path(root, record["latent"]))
        frames = _decode_video(samples, video_vae, tiled=tiled, tile_size=tile_size, temporal_size=temporal_size)
        audio = _decode_audio(samples, audio_vae) if mode == AUDIO_MODE_GENERATE else None
        frames, audio = _crop_export(frames, audio, record, fps)
        if mode == AUDIO_MODE_SOURCE and record.get("source_audio"):
            tensors = load_file(str(_child_path(root, record["source_audio"])), device="cpu")
            audio = {"waveform": tensors["waveform"], "sample_rate": int(record["source_sample_rate"])}
        elif mode == AUDIO_MODE_SOURCE:
            report.append(f"Segment #{index + 1}: 没有已保存的源音频，此段输出无声视频。")
        if audio is not None:
            audio = _pad_or_trim_audio_to_frames(
                audio, frame_count=int(frames.shape[0]), fps=fps, sample_rate=int(audio["sample_rate"]),
            )
        if (manifest.get("continuity_enabled") and manifest.get("continuity_grade_align")
                and record.get("continuity_from_prev", True) and int(record["trim_frames"]) > 0
                and previous_frames is not None and previous_index == index - 1):
            frames = match_export_opening_grade(frames, previous_frames)
        elif manifest.get("continuity_grade_align") and int(record["trim_frames"]) > 0:
            report.append(f"Segment #{index + 1}: 本轮未解码相邻前段，跳过段首光色对齐。")
        path = write_frames_to_mp4(_available_output_path(destination, f"seg_{index + 1:04d}.mp4"), frames, fps=fps, audio=audio)
        outputs.append({"timeline_index": index, "video": str(path), "frames": int(frames.shape[0])})
        _write_json(decode_manifest, {"source": str(root), "outputs": outputs, "status": "decoding"})
        report.append(f"Segment #{index + 1}: {frames.shape[0]} frames → {path}")
        counts.append(int(frames.shape[0]))
        if return_images:
            frames_out.append(frames)
            audios_out.append(audio or empty_audio_dict())
        if merge_video:
            merge_chunks.append(frames)
            merge_audios.append(audio or {})
        previous_frames, previous_index = frames[-1:].clone(), index
        del samples, frames, audio
        progress.update_absolute(position + 1)
    if merge_video:
        groups, current = [], []
        for position, record in enumerate(records):
            if current and (not record.get("continuity_from_prev", True)
                            or int(record["timeline_index"]) != int(records[position - 1]["timeline_index"]) + 1):
                groups.append(concat_continuous_chunks(current, [], SimpleNamespace(continuity_enabled=bool(manifest.get("continuity_enabled")))))
                current = []
            current.append(merge_chunks[position])
        if current:
            groups.append(concat_continuous_chunks(current, [], SimpleNamespace(continuity_enabled=bool(manifest.get("continuity_enabled")))))
        merged = concat_continuous_chunks(groups, [], SimpleNamespace(continuity_enabled=False))
        audio = _merge_generated_segment_audios(
            SimpleNamespace(segments=[]), merge_audios, total_frames=sum(counts), fps=fps, frame_counts=counts,
        ) if mode != "mute" else None
        path = write_frames_to_mp4(_available_output_path(destination, "merged.mp4"), merged, fps=fps, audio=audio)
        report.append(f"Merged: {path}")
    _write_json(decode_manifest, {"source": str(root), "outputs": outputs, "status": "complete"})
    return frames_out, audios_out, fps, sum(counts), "\n".join(report), str(destination)
