import os
import shutil
import time
from PIL import Image, ImageOps, ImageSequence

import numpy as np
import torch

import folder_paths

from ..vision import is_image
from ..processors.face_swapper import process_images

class NodesFaceSwap:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_images": ("IMAGE",),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ()
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "swapFace"

    def swapFace(self, source_image, target_images):
        now = f"{int(time.time())}"
        output_path = os.path.join(folder_paths.get_temp_directory(), "faceless/swapped_frames", now)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        process_images(source_image[0], target_images, output_path)

        images = []
        for file in sorted(os.listdir(output_path)):
            file_path = os.path.join(output_path, file)
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
        if len(images) > 1:
            output_image = torch.cat(images, dim=0)
        else:
            output_image = images[0]
        return (output_image,)
