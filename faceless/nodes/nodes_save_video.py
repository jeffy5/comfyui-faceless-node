import os
import time

import folder_paths

from ..ffmpeg import merge_video

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
        now = int(time.time())
        output_path = os.path.join(folder_paths.get_output_directory(), "faceless")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        output_filepath = os.path.join(output_path, f"{now}_" + os.path.basename(video_path))

        resolution = video.get("resolution")
        fps = video.get("fps")

        if not merge_video(frames_path, output_filepath, resolution, fps):
            raise Exception("Failed to merge video")

        # TODO Restore audio
        return ()
