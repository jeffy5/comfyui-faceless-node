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

    def save_video(self):
        pass

