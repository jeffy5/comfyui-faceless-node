import os
import shutil
import time
from PIL import Image, ImageOps, ImageSequence

import numpy as np
import torch

import folder_paths

from ..filesystem import check_faceless_model_exists, get_faceless_models
from ..vision import is_image
from ..processors.face_swapper import FaceSwapper

class NodesFaceSwap:
    @classmethod
    def INPUT_TYPES(cls):
        swapper_models = [os.path.basename(model) for model in get_faceless_models('face_swapper')]
        detector_models = [os.path.basename(model) for model in get_faceless_models('face_detector')]
        recognizer_models = [os.path.basename(model) for model in get_faceless_models('face_recognizer')]

        return {
            "required": {
                "images": ("IMAGE",),
                "face_image": ("IMAGE",),
                "swapper_model": (swapper_models,),
                "detector_model": (detector_models,),
                "recognizer_model": (recognizer_models,),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ()
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "swap_face"

    @classmethod
    def VALIDATE_INPUTS(cls, images, face_image, swapper_model, detector_model, recognizer_model):
        swapper_model_exists = check_faceless_model_exists("face_swapper", swapper_model)
        detector_model_exists = check_faceless_model_exists("face_detector", detector_model)
        recognizer_model_exists = check_faceless_model_exists("face_recognizer", recognizer_model)
        return swapper_model_exists and detector_model_exists and recognizer_model_exists

    def swap_face(self, images, face_image, swapper_model, detector_model, recognizer_model):
        # New swapper instance
        swapper = FaceSwapper(swapper_model)

        now = f"{int(time.time())}"
        output_path = os.path.join(folder_paths.get_temp_directory(), "faceless/swapped_frames", now)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)

        swapper.swap_images(images, face_image[0], output_path)
        # process_images(image[0], face_image, output_path)

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
