import os
from PIL import Image

from ..vision import is_image
from ..typing import FacelessVideo
from .nodes_remove_background import NodesRemoveBackground

class NodesVideoRemoveBackground(NodesRemoveBackground):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("FACELESS_VIDEO",),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ()
    RETURN_TYPES = ("FACELESS_VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "remove_video_background"

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        return super().VALIDATE_INPUTS(())

    def remove_video_background(self, video: FacelessVideo):
        frames_dir = video["frames_dir"]

        self.load_model()

        # TODO Improve batch process performance
        frame_filenames = sorted(os.listdir(frames_dir))
        for frame_filename in frame_filenames:
            file_path = os.path.join(frames_dir, frame_filename)
            if not is_image(file_path):
                continue

            img = Image.open(file_path)
            new_im, _ = self.remove_background(img)
            new_im.save(file_path)
        return (video,)
