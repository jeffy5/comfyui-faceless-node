import os

from ..processors.face_swapper import FaceSwapper
from ..filesystem import check_faceless_model_exists, get_faceless_models
from ..typing import FacelessVideo

class NodesVideoFaceSwap:
    @classmethod
    def INPUT_TYPES(cls):
        swapper_models = [os.path.basename(model) for model in get_faceless_models('face_swapper')]
        detector_models = [os.path.basename(model) for model in get_faceless_models('face_detector')]
        recognizer_models = [os.path.basename(model) for model in get_faceless_models('face_recognizer')]

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

    @classmethod
    def VALIDATE_INPUTS(cls, source_image, target_video, swapper_model, detector_model, recognizer_model):
        swapper_model_exists = check_faceless_model_exists("face_swapper", swapper_model)
        detector_model_exists = check_faceless_model_exists("face_detector", detector_model)
        recognizer_model_exists = check_faceless_model_exists("face_recognizer", recognizer_model)
        return swapper_model_exists and detector_model_exists and recognizer_model_exists

    def swap_video_face(self, source_image, target_video: FacelessVideo, swapper_model, detector_model, recognizer_model):
        if not target_video["extract_frames"]:
            raise Exception("target video must be extracted frames")
        frames_dir = target_video["frames_dir"]

        swapper = FaceSwapper(swapper_model)

        # Fetch source image or change process_frames argument.
        swapper.swap_video(source_image[0], frames_dir)
        return (target_video,)
