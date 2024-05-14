import folder_paths

import os
import shutil

from ..filesystem import is_video
from ..ffmpeg import extract_frames as process_extract_frames
from ..vision import detect_video_fps, detect_video_resolution
from ..typing import FacelessVideo

class NodesLoadVideo:

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if is_video(os.path.join(input_dir, f))]
        return {
            "required": {
                "video": (sorted(files),),
                "extract_frames": ("BOOLEAN", {
                    "default": True,
                    "label_off": "OFF",
                    "label_on": "ON",
                }),
                "trim_frame_start": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 999999,
                    "display": "number",
                }),
                "trim_frame_end": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 999999,
                    "display": "number",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, video, extract_frames: bool, trim_frame_start: int, trim_frame_end: int):
        if trim_frame_start != -1 and trim_frame_end != -1 and trim_frame_start >= trim_frame_end:
            return False
        return True

    CATEGORY = "faceless"
    RETURN_TYPES = ("FACELESS_VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "process"

    def process(self, video, extract_frames, trim_frame_start: int, trim_frame_end: int):
        video_path = folder_paths.get_annotated_filepath(video)
        video_name, _ = os.path.splitext(os.path.basename(video_path))
        frames_dir = os.path.join(folder_paths.get_temp_directory(), "faceless", video_name, "frames")

        video_resolution = detect_video_resolution(video_path)
        video_fps = detect_video_fps(video_path)
        if video_resolution is None or video_fps is None:
            raise Exception("Failed to detect video resolution and fps")

        if trim_frame_start == -1:
            final_trim_frame_start = None
        else:
            final_trim_frame_start = trim_frame_start
        if trim_frame_end == -1:
            final_trim_frame_end = None
        else:
            final_trim_frame_end = trim_frame_end

        if extract_frames:
            # Remove all cached frames
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            os.makedirs(frames_dir)

            if not process_extract_frames(video_path, frames_dir, video_resolution, video_fps, final_trim_frame_start, final_trim_frame_end):
                raise Exception("Failed to extract frames")

        faceless_video: FacelessVideo = {
            "video_path": video_path,
            "extract_frames": extract_frames,
            "frames_dir": frames_dir,
            "output_path": "",
            "resolution": video_resolution,
            "fps": video_fps,
            "trim_frame_start": final_trim_frame_start,
            "trim_frame_end": final_trim_frame_end,
        }
        return (faceless_video,)
