from typing import Optional, Tuple
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from .filesystem import is_video, is_image
from .typing import Resolution, VisionFrame, Fps

def detect_video_fps(video_path : str) -> Optional[Fps]:
    if is_video(video_path):
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            video_fps = video_capture.get(cv2.CAP_PROP_FPS)
            video_capture.release()
            return video_fps

def restrict_video_fps(video_path : str, fps : Fps) -> Fps:
    if is_video(video_path):
        video_fps = detect_video_fps(video_path)
        if video_fps is None:
            return fps
        if video_fps < fps:
            return video_fps
    return fps

def detect_video_resolution(video_path : str) -> Optional[Resolution]:
    if is_video(video_path):
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            video_capture.release()
            return int(width), int(height)

def normalize_resolution(resolution : Tuple[float, float]) -> Resolution:
    width, height = resolution

    if width and height:
        normalize_width = round(width / 2) * 2
        normalize_height = round(height / 2) * 2
        return normalize_width, normalize_height
    return 0, 0

def pack_resolution(resolution : Resolution) -> str:
    width, height = normalize_resolution(resolution)
    return str(width) + 'x' + str(height)

def unpack_resolution(resolution : str) -> Resolution:
    width, height = map(int, resolution.split('x'))
    return width, height

@lru_cache(maxsize = 128)
def read_static_image(image_path : str) -> Optional[VisionFrame]:
    return read_image(image_path)

def read_image(image_path : str) -> Optional[VisionFrame]:
    if is_image(image_path):
        return cv2.imread(image_path)
    return None

def tensor_to_vision_frame(image_tensor) -> Optional[VisionFrame]:
    i = 255. * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

    # cv2_image = np.transpose(np.array(img), (1, 2, 0))
    return cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

def write_image(image_path : str, vision_frame : VisionFrame) -> bool:
    if image_path:
        return cv2.imwrite(image_path, vision_frame)
    return False

def resize_frame_resolution(vision_frame : VisionFrame, max_resolution : Resolution) -> VisionFrame:
    height, width = vision_frame.shape[:2]
    max_width, max_height = max_resolution

    if height > max_height or width > max_width:
        scale = min(max_height / height, max_width / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(vision_frame, (new_width, new_height))
    return vision_frame

