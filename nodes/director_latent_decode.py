"""Decode persisted Director AV samples independently of sampling."""

from __future__ import annotations

from comfy_execution.graph_utils import ExecutionBlocker

from ..director.deferred_decode import decode_latent_run


class MiniMaxH3DirectorLatentDecode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_folder": ("STRING", {"default": "", "tooltip": "导演台保存的 latent 任务文件夹或 manifest.json；相对路径基于 ComfyUI/output。"}),
                "video_vae": ("VAE",),
                "segments": ("STRING", {"default": "all", "tooltip": "all 或 1,3-5，按原时间轴编号，从 1 开始。"}),
                "output_directory": ("STRING", {"default": "", "tooltip": "留空存入任务文件夹下 decoded/；每次运行使用新子目录。"}),
                "tiled_decode": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64}),
                "temporal_size": ("INT", {"default": 64, "min": 16, "max": 4096, "step": 8}),
                "merge_video": ("BOOLEAN", {"default": False, "tooltip": "另存 merged.mp4；需在内存保留所有选中分段。"}),
                "return_images": ("BOOLEAN", {"default": False, "tooltip": "默认只保存文件并阻断 IMAGE/AUDIO 输出以节省内存；开启后输出每段完整画面/音频，便于下游处理。"}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "生成音频任务必需；静音或原声任务可不接。"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("images", "audio", "fps", "frame_count", "report", "output_directory")
    OUTPUT_IS_LIST = (True, True, False, False, False, False)
    FUNCTION = "execute"
    CATEGORY = "MiniMaxH3"
    OUTPUT_NODE = True
    DESCRIPTION = "独立解码导演台保存的 AV latent。无需 MODEL / CLIP，不重新采样。自动保存分段 MP4。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(
        self, latent_folder, video_vae, segments="all", output_directory="",
        tiled_decode=False, tile_size=512, temporal_size=64,
        merge_video=False, return_images=False, audio_vae=None,
    ):
        result = decode_latent_run(
            latent_folder, video_vae=video_vae, audio_vae=audio_vae, selection=segments,
            output_directory=output_directory, tiled=tiled_decode,
            tile_size=tile_size, temporal_size=temporal_size,
            merge_video=merge_video, return_images=return_images,
        )
        if not return_images:
            return [ExecutionBlocker(None)], [ExecutionBlocker(None)], *result[2:]
        return result
