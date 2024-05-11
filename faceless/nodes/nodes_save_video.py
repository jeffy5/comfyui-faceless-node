import os
import time

import folder_paths

from ..ffmpeg import merge_video, restore_audio

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

    def save_video(self, video: FacelessVideo):
        video_path = video.get("video_path")
        frames_path = video.get("output_path")
        fps = video.get("fps")
        trim_frame_start = video.get("trim_frame_start")
        trim_frame_end = video.get("trim_frame_end")

        output_dir = os.path.join(folder_paths.get_output_directory(), "faceless")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        resolution = video.get("resolution")
        fps = video.get("fps")

        output_temp_path = os.path.join(folder_paths.get_temp_directory(), "faceless/output", os.path.basename(video_path))
        if not os.path.exists(os.path.dirname(output_temp_path)):
            os.makedirs(os.path.dirname(output_temp_path))

        if not merge_video(frames_path, output_temp_path, resolution, fps):
            raise Exception("Failed to merge video")

        # Restore audio
        now = int(time.time())
        output_path = os.path.join(output_dir, f"{now}_" + os.path.basename(video_path))
        if not restore_audio(output_temp_path, video_path, output_path, fps, trim_frame_start, trim_frame_end):
            raise Exception("Failed to restore audio")
        video["output_path"] = output_path
        return ()
