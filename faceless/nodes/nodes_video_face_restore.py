import os

from ..processors.face_restoration import FaceRestoration

from ..filesystem import get_faceless_models

class NodesVideoFaceRestore:
    @classmethod
    def INPUT_TYPES(cls):
        restoration_models = [os.path.basename(model) for model in get_faceless_models('face_restoration')]
        return {
            "required": {
                "video": ("FACELESS_VIDEO",),
                "restoration_model": (restoration_models,),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ()
    RETURN_TYPES = ("FACELESS_VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "restoreVideoFace"

    def restoreVideoFace(self, video, restoration_model):
        frames_dir = video["frames_dir"]
        face_restoration = FaceRestoration(restoration_model)
        face_restoration.restore_video(frames_dir)
        return (video,)
