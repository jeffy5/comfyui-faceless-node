import os
import glob
import filetype

import folder_paths

from .typing import FrameFormat, ModelType

def is_file(file_path : str) -> bool:
    return bool(file_path and os.path.isfile(file_path))

def is_image(image_path : str) -> bool:
    return is_file(image_path) and filetype.helpers.is_image(image_path)


def is_video(video_path : str) -> bool:
    return is_file(video_path) and filetype.helpers.is_video(video_path)

def get_temp_frames_pattern(target_path : str, temp_frame_prefix : str, format: FrameFormat) -> str:
    return os.path.join(target_path, temp_frame_prefix + '.' + format)

def resolve_relative_path(path : str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))

def get_faceless_models(model_type: ModelType):
    models_path = os.path.join(folder_paths.models_dir,f"faceless/{model_type}/*")
    models = glob.glob(models_path)
    models = [x for x in models if x.endswith(".onnx") or x.endswith(".pth")]
    return models

def get_face_swapper_models():
    return get_faceless_models("face_swapper")

def get_face_detector_models():
    return get_faceless_models("face_detector")

def get_face_recognizer_models():
    return get_faceless_models("face_recognizer")

def get_face_landmarker_models():
    return get_faceless_models("face_landmarker")
