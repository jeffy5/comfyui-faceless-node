from urllib.request import urlretrieve
from urllib.parse import urlparse
from PIL import Image, ImageSequence, ImageOps
import os
import time

import torch
import numpy as np

import folder_paths

class NodesLoadImageUrl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {
                    "default": "",
                }),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "load_image_url"

    def load_image_url(self, url):
        # Download image
        now = int(time.time())
        u = urlparse(url)

        output_path = os.path.join(folder_paths.get_temp_directory(), "faceless")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        output_filename = os.path.join(output_path, f"{now}_{os.path.basename(u.path)}")
        urlretrieve(url, output_filename)

        img = Image.open(output_filename)
        images = []
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
