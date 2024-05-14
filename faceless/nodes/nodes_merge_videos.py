import os
import shutil

import folder_paths

from ..vision import detect_video_fps, detect_video_resolution
from ..ffmpeg import merge_videos, extract_frames
from ..typing import FacelessVideo
from .nodes_save_video import NodesSaveVideo

class NodesMergeVideos(NodesSaveVideo):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("FACELESS_VIDEO",),
                "background_video": ("FACELESS_VIDEO",),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ("FACELESS_VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "merge_videos"

    def merge_videos(self, video: FacelessVideo, background_video: FacelessVideo):
        video_path = video["video_path"]
        bg_video_path = background_video["video_path"]
        resolution = video["resolution"]

        merged_video_filename = "merged_" + os.path.basename(video_path)
        base_dir = os.path.join(folder_paths.get_temp_directory(), "faceless", os.path.splitext(merged_video_filename)[0])
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

        merged_video_path = os.path.join(base_dir, merged_video_filename)
        if not merge_videos(video_path, bg_video_path, merged_video_path, resolution):
            raise Exception("merge videos failed")

        # Extract frames
        frames_dir = os.path.join(base_dir, "frames")
        if os.path.exists(frames_dir):
            shutil.rmtree(frames_dir)
        os.makedirs(frames_dir)

        video_resolution = detect_video_resolution(merged_video_path)
        video_fps = detect_video_fps(merged_video_path)
        if video_resolution is None or video_fps is None:
            raise Exception("Failed to detect video resolution and fps")

        if not extract_frames(merged_video_path, frames_dir, video_resolution, video_fps, None, None):
            raise Exception("Failed to extract frames")

        merged_video: FacelessVideo = {
            "video_path": merged_video_path,
            "extract_frames": True,
            "frames_dir": frames_dir,
            "fps": video_fps,
            "resolution": video_resolution,
            "output_path": "",
            "trim_frame_start": None,
            "trim_frame_end": None,
        }
        return (merged_video,)
