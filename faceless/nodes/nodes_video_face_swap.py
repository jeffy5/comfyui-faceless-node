import os

from ..processors.face_swapper import swap_video
from ..filesystem import get_face_detector_models, get_face_recognizer_models, get_face_swapper_models
from ..typing import FacelessVideo

class NodesVideoFaceSwap:
    @classmethod
    def INPUT_TYPES(cls):
        swapper_models = [os.path.basename(model) for model in get_face_swapper_models()]
        detector_models = [os.path.basename(model) for model in get_face_detector_models()]
        recognizer_models = [os.path.basename(model) for model in get_face_recognizer_models()]

        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_video": ("FACELESS_VIDEO",),
                "swapper_model": (swapper_models,),
                "detector_model": (detector_models,),
                "recognizer_model": (recognizer_models,),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ()
    RETURN_TYPES = ("FACELESS_VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "swap_video_face"

    def swap_video_face(self, source_image, target_video: FacelessVideo, swapper_model, detector_model, recognizer_model):
        if not target_video["extract_frames"]:
            raise Exception("target video must be extracted frames")
        frames_dir = target_video["frames_dir"]

        # Fetch source image or change process_frames argument.
        swap_video(source_image[0], frames_dir)
        return (target_video,)
