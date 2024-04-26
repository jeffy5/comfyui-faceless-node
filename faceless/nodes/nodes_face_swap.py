class NodesFaceSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ()
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "swapFace"

    def swapFace(self, images):
        print("swap face")
        return (images,)
