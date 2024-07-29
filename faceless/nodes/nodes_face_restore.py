import os
import time
from PIL import Image, ImageOps, ImageSequence
import shutil

import folder_paths

import numpy as np
import torch

from ..filesystem import get_faceless_models
from ..processors.face_restoration import FaceRestoration
from ..vision import is_image


class NodesFaceRestore:
    @classmethod
    def INPUT_TYPES(cls):
        restoration_models = [os.path.basename(model) for model in get_faceless_models('face_restoration')]
        return {
            "required": {
                "images": ("IMAGE",),
                "restoration_model": (restoration_models,),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ()
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "restoreFace"

    def restoreFace(self, images, restoration_model):
        face_restoration = FaceRestoration(restoration_model)

        now = f"{int(time.time())}"
        output_path = os.path.join(folder_paths.get_temp_directory(), "faceless/restored_frames", now)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        face_restoration.restore_images(images, output_path)

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
