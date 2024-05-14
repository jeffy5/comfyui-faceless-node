import os
from os.path import basename, splitext
import shutil
import time

import folder_paths

from ..ffmpeg import merge_frames, restore_audio

from ..typing import FacelessVideo

class NodesSaveVideo:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("FACELESS_VIDEO",),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True

    def merge_frames_and_audio(self, video: FacelessVideo):
        video_path = video.get("video_path")
        frames_dir = video.get("frames_dir")
        fps = video.get("fps")
        trim_frame_start = video.get("trim_frame_start")
        trim_frame_end = video.get("trim_frame_end")

        resolution = video["resolution"]
        fps = video["fps"]

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        muted_path = os.path.join(folder_paths.get_temp_directory(), "faceless", video_name, "output", "muted_" + os.path.basename(video_path)) 
        output_path = os.path.join(folder_paths.get_temp_directory(), "faceless", video_name, "output", os.path.basename(video_path)) 
        if not os.path.exists(os.path.dirname(muted_path)):
            os.makedirs(os.path.dirname(muted_path))

        # Merge frames
        if not merge_frames(video_path, frames_dir, muted_path, resolution, fps):
            raise Exception("Failed to merge video")

        # Restore audio
        if not restore_audio(muted_path, video_path, output_path, fps, trim_frame_start, trim_frame_end):
            raise Exception("Failed to restore audio")
        return output_path

    def save_video(self, video: FacelessVideo):
        video_path = video["video_path"]

        output_dir = os.path.join(folder_paths.get_output_directory(), "faceless")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        now = int(time.time())
        output_path = os.path.join(output_dir, f"{now}_" + os.path.basename(video_path))

        if video["extract_frames"]:
            # Merge frames and video
            temp_output_path = self.merge_frames_and_audio(video)
            shutil.move(temp_output_path, output_path)
        else:
            # Just copy the video if not extract frames
            shutil.copy(video_path, output_path)
        video["output_path"] = output_path
        return ()
