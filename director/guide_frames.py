"""Timed image conditions shared by reference and first/last-frame generation."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass

import torch

GUIDE_FRAME_KEY = "_director_timed_guide"


@dataclass
class GuideFrame:
    position: str
    frame_index: int
    image: torch.Tensor
    image_identity: str


def load_guide_frames(raw, common=None) -> list[GuideFrame]:
    from .plan import load_reference_tensor
    import folder_paths

    out = []
    for entries in (common, raw):
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ValueError("Guide frames must be a list.")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Each guide frame must be an object.")
            if not (entry.get("imageFile") or entry.get("imageB64")):
                continue
            position = str(entry.get("position") or "frame")
            if position not in {"first", "middle", "last", "frame"}:
                raise ValueError(f"Unknown guide frame position: {position}")
            frame = 0
            if position == "frame":
                try:
                    value = float(entry.get("frameIndex", 0))
                except (TypeError, ValueError) as exc:
                    raise ValueError("Guide frame index must be a non-negative integer.") from exc
                if not math.isfinite(value) or value < 0 or not value.is_integer():
                    raise ValueError("Guide frame index must be a non-negative integer.")
                frame = int(value)
            image = load_reference_tensor(entry)
            if image is None:
                raise ValueError(f"Guide image could not be loaded: {entry.get('imageFile') or '(embedded image)'}")
            name = str(entry.get("imageFile") or "").replace("\\", "/")
            path = os.path.join(folder_paths.get_input_directory(), name)
            if name and os.path.isfile(path):
                stat = os.stat(path)
                identity = f"{name}:{stat.st_mtime_ns}:{stat.st_size}"
            else:
                identity = hashlib.sha256(str(entry.get("imageB64") or "").encode()).hexdigest()
            out.append(GuideFrame(position, frame, image[:1], identity))
    return out


def resolve_guide_positions(guides, *, visible_frames: int, prefix_frames: int = 0):
    visible = int(visible_frames)
    prefix = max(0, int(prefix_frames))
    if visible < 1:
        raise ValueError("Guide frames require a non-empty output segment.")
    resolved = {}
    for guide in guides:
        frame = {
            "first": 0,
            "middle": (visible - 1) // 2,
            "last": visible - 1,
            "frame": guide.frame_index,
        }[guide.position]
        if frame < 0 or frame >= visible:
            raise ValueError(
                f"Guide frame {frame} is outside this segment's {visible} output frames "
                f"(valid indices: 0–{visible - 1})."
            )
        resolved[prefix + frame] = guide
    return resolved


def apply_guide_frames(positive, latent, *, vae, guides, visible_frames: int, prefix_frames: int = 0):
    if not guides:
        return positive

    import node_helpers
    from ..lib.image_prep import fit_canvas
    from .h3_context_patches import CTX_FRAME_KEY, ensure_layout_patch, ensure_payload_patch
    from .h3_motion_context import _existing_keyframes, pixel_frames_for_latent_t, video_from_latent

    if vae is None:
        raise ValueError("Timed guide images require a video VAE.")
    positions = resolve_guide_positions(guides, visible_frames=visible_frames, prefix_frames=prefix_frames)
    ensure_layout_patch()
    ensure_payload_patch()
    video = video_from_latent(latent)
    width, height = int(video.shape[4]) * 16, int(video.shape[3]) * 16
    keyframes = {}
    for original in _existing_keyframes(positive):
        frame = int(original.get(CTX_FRAME_KEY, original.get("resolved_frame_index", 0)))
        keyframes[frame] = {**original, CTX_FRAME_KEY: frame}
    for frame, guide in positions.items():
        image = fit_canvas(guide.image, width, height, upscale_mode="lanczos")
        keyframes[frame] = {
            "resolved_frame_index": frame,
            CTX_FRAME_KEY: frame,
            GUIDE_FRAME_KEY: True,
            "latent": vae.encode(image),
        }
    return node_helpers.conditioning_set_values(positive, {
        "minimax_keyframes": list(keyframes.values()),
        "minimax_frame_count": pixel_frames_for_latent_t(int(video.shape[2])),
    })
