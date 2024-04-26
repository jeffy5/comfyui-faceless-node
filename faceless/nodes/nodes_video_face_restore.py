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
        print("video face restore" + video.video_path + " " + video.frames_path)
        return (video,)
