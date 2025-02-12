import os
import argparse
from urllib.request import urlretrieve

parser = argparse.ArgumentParser(description="Download faceless models")

parser.add_argument(
    "--all",
    action="store_true",
    help="Download all models, or just download required models",
    default=False,
)

args = parser.parse_args()

base_path = os.path.dirname(os.path.realpath(__file__))
models_dir = os.path.normpath(os.path.join(base_path, "../../models/faceless"))
facefusion_assets_url = (
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0"
)

# Define the directory structure
model_sets = {
    "face_detector": [
        {
            "name": "yoloface_8n.onnx",
            "url": facefusion_assets_url + "/yoloface_8n.onnx",
            "required": True,
        }
    ],
    "face_landmarker": [
        {
            "name": "2dfan4.onnx",
            "url": facefusion_assets_url + "/2dfan4.onnx",
            "required": True,
        },
        {
            "name": "face_landmarker_68_5.onnx",
            "url": facefusion_assets_url + "/face_landmarker_68_5.onnx",
            "required": True,
        },
    ],
    "face_recognizer": [
        {
            "name": "arcface_w600k_r50.onnx",
            "url": facefusion_assets_url + "/arcface_w600k_r50.onnx",
            "required": True,
        }
    ],
    "face_swapper": [
        {
            "name": "inswapper_128.onnx",
            "url": facefusion_assets_url + "/inswapper_128.onnx",
            "required": True,
        },
        {
            "name": "inswapper_128_fp16.onnx",
            "url": facefusion_assets_url + "/inswapper_128_fp16.onnx",
            "required": False,
        },
        {
            "name": "simswap_256.onnx",
            "url": facefusion_assets_url + "/simswap_256.onnx",
            "required": False,
        },
        {
            "name": "simswap_512_unofficial.onnx",
            "url": facefusion_assets_url + "/simswap_512_unofficial.onnx",
            "required": False,
        },
        {
            "name": "blendswap_256.onnx",
            "url": facefusion_assets_url + "/blendswap_256.onnx",
            "required": False,
        },
        {
            "name": "uniface_256.onnx",
            "url": facefusion_assets_url + "/uniface_256.onnx",
            "required": False,
        },
    ],
    "face_restoration": [
        {
            "name": "gfpgan_1.2.onnx",
            "url": facefusion_assets_url + "/gfpgan_1.2.onnx",
            "required": False,
        },
        {
            "name": "gfpgan_1.3.onnx",
            "url": facefusion_assets_url + "/gfpgan_1.3.onnx",
            "required": False,
        },
        {
            "name": "gfpgan_1.4.onnx",
            "url": facefusion_assets_url + "/gfpgan_1.4.onnx",
            "required": True,
        },
        {
            "name": "codeformer.onnx",
            "url": facefusion_assets_url + "/codeformer.onnx",
            "required": False,
        },
        {
            "name": "gpen_bfr_256.onnx",
            "url": facefusion_assets_url + "/gpen_bfr_256.onnx",
            "required": False,
        },
        {
            "name": "gpen_bfr_512.onnx",
            "url": facefusion_assets_url + "/gpen_bfr_512.onnx",
            "required": False,
        },
        {
            "name": "gpen_bfr_1024.onnx",
            "url": facefusion_assets_url + "/gpen_bfr_1024.onnx",
            "required": False,
        },
        {
            "name": "gpen_bfr_2048.onnx",
            "url": facefusion_assets_url + "/gpen_bfr_2048.onnx",
            "required": False,
        },
        {
            "name": "restoreformer_plus_plus.onnx",
            "url": facefusion_assets_url + "/restoreformer_plus_plus.onnx",
            "required": False,
        },
    ],
    ".": [
        {
            "name": "gender_age.onnx",
            "url": facefusion_assets_url + "/gender_age.onnx",
            "required": True,
        },
        {
            "name": "rmbg.pth",
            "url": "https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth?download=true",
            "required": True,
        },
    ],
}

# Create the directories they don't exist
for model_type in model_sets:
    full_dir = os.path.join(models_dir, model_type)
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)

# Download and save the models
for model_type, models in model_sets.items():
    for model in models:
        name = model["name"]
        url = model["url"]
        required = model["required"]

        if not args.all and not required:
            continue

        file_path = os.path.join(models_dir, model_type, name)
        if os.path.exists(file_path):
            continue

        print(f"Start downloading `{name}`")
        urlretrieve(url, file_path)
        print(f"Downloaded `{name}`")
