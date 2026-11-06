"""Select explicitly referenced media and compact official MiniMax ordinals."""

from __future__ import annotations

import logging
import re

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.references")
REFERENCE_TAG = re.compile(r"<\s*(Picture|Video|Audio)\s+(\d+)\s*>", re.IGNORECASE)


def explicit_reference_only_enabled(timeline):
    output = (timeline or {}).get("output") or {}
    value = output.get("explicitReferenceOnly")
    if value is None:
        value = output.get("explicit_reference_only", True)
    return value is None or value in (True, 1, "1", "true", "True", "on")


def _indexed_resources(resources, prefix):
    indexed = {}
    for key, value in (resources or {}).items():
        if value is None:
            continue
        match = re.fullmatch(re.escape(prefix) + r"(\d+)", str(key))
        if match is None:
            raise ValueError(f"Invalid MiniMax reference slot: {key!r}")
        indexed[int(match.group(1))] = value
    return dict(sorted(indexed.items()))


def filter_prompt_references(prompt, *, ref_images=None, ref_videos=None, ref_video_audios=None, ref_audios=None):
    text = prompt or ""
    requested = {kind: set() for kind in ("Picture", "Video", "Audio")}
    for match in REFERENCE_TAG.finditer(text):
        requested[match.group(1).capitalize()].add(int(match.group(2)) - 1)

    images = _indexed_resources(ref_images, "ref_image_")
    videos = _indexed_resources(ref_videos, "ref_video_")
    soundtracks = _indexed_resources(ref_video_audios, "ref_video_audio_")
    audios = _indexed_resources(ref_audios, "ref_audio_")
    audio_entries = {}
    for video_index in videos:
        audio = soundtracks.get(video_index)
        if isinstance(audio, dict) and audio.get("waveform") is not None:
            audio_entries[len(audio_entries)] = (video_index, audio)
    audio_offset = len(audio_entries)
    for index, audio in audios.items():
        if isinstance(audio, dict) and audio.get("waveform") is not None:
            audio_entries[audio_offset + index] = (None, audio)

    selected_images = [index for index in images if index in requested["Picture"]]
    selected_videos = [index for index in videos if index in requested["Video"]]
    selected_audios = [index for index in audio_entries if index in requested["Audio"]]
    paired_audios = [index for index in selected_audios if audio_entries[index][0] in selected_videos]
    separate_audios = [index for index in selected_audios if index not in paired_audios]
    audio_order = paired_audios + separate_audios
    mappings = {
        "Picture": {index: position + 1 for position, index in enumerate(selected_images)},
        "Video": {index: position + 1 for position, index in enumerate(selected_videos)},
        "Audio": {index: position + 1 for position, index in enumerate(audio_order)},
    }
    missing = [f"<{kind} {index + 1}>" for kind, indices in requested.items()
               for index in sorted(indices - mappings[kind].keys())]
    if missing:
        log.warning("Reference tags have no available media and were removed: %s", ", ".join(missing))

    def rewrite_tag(match):
        kind = match.group(1).capitalize()
        number = mappings[kind].get(int(match.group(2)) - 1)
        return f"<{kind} {number}>" if number is not None else ""

    rewritten = REFERENCE_TAG.sub(rewrite_tag, text)
    selected_soundtracks = {
        f"ref_video_audio_{mappings['Video'][audio_entries[index][0]] - 1}": audio_entries[index][1]
        for index in paired_audios
    }
    selected_standalone_audios = {
        f"ref_audio_{position}": audio_entries[index][1]
        for position, index in enumerate(separate_audios)
    }
    log.info(
        "Explicit reference selection: pictures %d/%d, videos %d/%d, audio %d/%d; model ordinals %s",
        len(selected_images), len(images), len(selected_videos), len(videos),
        len(audio_order), len(audio_entries), mappings,
    )
    return (
        rewritten,
        {f"ref_image_{position}": images[index] for position, index in enumerate(selected_images)} or None,
        {f"ref_video_{position}": videos[index] for position, index in enumerate(selected_videos)} or None,
        selected_soundtracks or None,
        selected_standalone_audios or None,
    )
