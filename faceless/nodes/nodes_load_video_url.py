import os
import shutil
import time
import hashlib
from urllib.request import urlretrieve
from urllib.parse import urlparse, unquote

import folder_paths

from ..ffmpeg import extract_frames
from ..vision import detect_video_fps, detect_video_resolution
from ..typing import FacelessVideo


class NodesLoadVideoUrl:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": ""
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

    CATEGORY = "faceless"
    RETURN_TYPES = ("FACELESS_VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "load_video_url"

    def load_video_url(self, url: str, trim_frame_start: int, trim_frame_end: int):
        hash = hashlib.md5()
        hash.update(url.encode('utf-8'))
        url_id = hash.hexdigest()

        # TODO Get file ext from url
        video_filepath = os.path.join(folder_paths.get_input_directory(), "faceless/download", f"{url_id}.mp4")
        if not os.path.exists(video_filepath):
            # Download video
            download_temp_filepath = os.path.join(folder_paths.get_temp_directory(), "faceless/download", f"{url_id}.mp4")
            if not os.path.exists(os.path.dirname(download_temp_filepath)):
                os.makedirs(os.path.dirname(download_temp_filepath))

            urlretrieve(url, download_temp_filepath)

            if not os.path.exists(os.path.dirname(video_filepath)):
                os.makedirs(os.path.dirname(video_filepath))
            shutil.move(download_temp_filepath, video_filepath)

            # Save video

        video_name, _ = os.path.splitext(os.path.basename(video_filepath))
        frames_path = os.path.join(folder_paths.get_temp_directory(), "faceless/frames", video_name)
        print("frames path: " + frames_path)

        # Remove all cached frames
        if os.path.exists(frames_path):
            shutil.rmtree(frames_path)
        os.makedirs(frames_path)

        video_resolution = detect_video_resolution(video_filepath)
        video_fps = detect_video_fps(video_filepath)
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

        if not extract_frames(video_filepath, frames_path, video_resolution, video_fps, final_trim_frame_start, final_trim_frame_end):
            raise Exception("Failed to extract frames")

        faceless_video: FacelessVideo = {
            'video_path': video_filepath,
            'output_path': frames_path,
            'resolution': video_resolution,
            'fps': video_fps,
            'trim_frame_start': final_trim_frame_start,
            'trim_frame_end': final_trim_frame_end,
        }
        return (faceless_video,)

