from ..typing import FacelessVideo

from ..processors.face_enhancer import enhance_video

class NodesVideoFaceRestore:
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
    FUNCTION = "restoreVideoFace"

    def restoreVideoFace(self, video):
        frames_dir = video["frames_dir"]
        enhance_video(frames_dir)
        return (video,)
