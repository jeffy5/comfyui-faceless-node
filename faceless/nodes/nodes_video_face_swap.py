import os
import shutil

import folder_paths

from ..processors.face_swapper import process_frames
from ..typing import FacelessVideo

class NodesVideoFaceSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_video": ("FACELESS_VIDEO",),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ()
    RETURN_TYPES = ("FACELESS_VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "swapVideoFace"

    def swapVideoFace(self, source_image, target_video: FacelessVideo):
        video_path = target_video.get("video_path")
        video_name, _ = os.path.splitext(os.path.basename(video_path))

        frames_path = os.path.join(folder_paths.get_temp_directory(), "faceless/frames", video_name)
        output_path = os.path.join(folder_paths.get_temp_directory(), "faceless/swapped_frames", video_name)

        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        # TODO Check if has face on source image

        print("source image", len(source_image))
        # Fetch source image or change process_frames argument.
        process_frames(source_image[0], frames_path, output_path)

        target_video['output_path'] = output_path
        return (target_video,)
