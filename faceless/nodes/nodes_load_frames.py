import os
from PIL import Image, ImageOps, ImageSequence
import torch
import numpy as np

from ..filesystem import is_image
from ..typing import FacelessVideo

class NodesLoadFrames:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("FACELESS_VIDEO",),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_frames"

    def load_frames(self, video: FacelessVideo):
        frames_path = video['frames_path']

        images = []
        for file in sorted(os.listdir(frames_path)):
            file_path = os.path.join(frames_path, file)
            if not is_image(file_path):
                continue
            img = Image.open(file_path)
            for i in ImageSequence.Iterator(img):
                i = ImageOps.exif_transpose(i)
                if i.mode == 'I':
                    i = i.point(lambda i: i * (1 / 255))
                image = i.convert("RGB")
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image)[None,]
                images.append(image)
                images.append(image)
        if len(images) > 1:
            output_image = torch.cat(images, dim=0)
        else:
            output_image = images[0]
        return (output_image,)
