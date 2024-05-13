import os
from PIL import Image

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms.functional import normalize

from folder_paths import models_dir

from ..processors.briarmbg import BriaRMBG
from ..image_helper import tensor_to_pil, pil_to_tensor

class NodesRemoveBackground:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    CATEGORY = "faceless"
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "remove_images_background"

    @classmethod
    def VALIDATE_INPUTS(cls, images):
        if not os.path.exists(os.path.join(models_dir, "faceless/rmbg.pth")):
            return False
        return True

    def remove_images_background(self, images):
        self.load_model()

        processed_images = []
        processed_masks = []
        for image in images:
            orig_image = tensor_to_pil(image)

            new_im, pil_im = self.remove_background(orig_image)

            new_im_tensor = pil_to_tensor(new_im)
            pil_im_tensor = pil_to_tensor(pil_im)

            processed_images.append(new_im_tensor)
            processed_masks.append(pil_im_tensor)

        new_ims = torch.cat(processed_images, dim=0)
        new_masks = torch.cat(processed_masks, dim=0)
        return (new_ims, new_masks)

    def remove_background(self, orig_image):
        w, h = orig_image.size
        model_input_size = [1024,1024]
        image = self._preprocess_image(np.array(orig_image), model_input_size)

        if torch.cuda.is_available():
            image = image.to("cuda")
        elif torch.backends.mps.is_available():
            image = image.to("mps")

        result = self.rmbg(image)

        result_image = self._postprocess_image(result[0][0], [h, w])

        pil_im = Image.fromarray(result_image)
        no_bg_image = Image.new("RGBA", pil_im.size, (0,0,0,0))
        no_bg_image.paste(orig_image, mask=pil_im)
        return (no_bg_image, pil_im)

    def load_model(self):
        rmbg = BriaRMBG()
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model_path = os.path.join(models_dir, "faceless/rmbg.pth")
        rmbg.load_state_dict(torch.load(model_path, map_location=device))
        rmbg.to(device)
        rmbg.eval()
        self.rmbg = rmbg

    def _preprocess_image(self, im: np.ndarray, model_input_size: list) -> torch.Tensor:
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        # orig_im_size=im.shape[0:2]
        im_tensor = torch.tensor(im, dtype=torch.float32).permute(2,0,1)
        im_tensor = F.interpolate(torch.unsqueeze(im_tensor,0), size=model_input_size, mode='bilinear')
        image = torch.divide(im_tensor,255.0)
        image = normalize(image,[0.5,0.5,0.5],[1.0,1.0,1.0])
        return image

    def _postprocess_image(self, result: torch.Tensor, im_size: list)-> np.ndarray:
        result = torch.squeeze(F.interpolate(result, size=im_size, mode='bilinear') ,0)
        ma = torch.max(result)
        mi = torch.min(result)
        result = (result-mi)/(ma-mi)
        im_array = (result*255).permute(1,2,0).cpu().data.numpy().astype(np.uint8)
        im_array = np.squeeze(im_array)
        return im_array
