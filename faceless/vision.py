from typing import Optional, Tuple

import cv2

from .filesystem import is_video
from .typing import Resolution

def detect_video_fps(video_path : str) -> Optional[float]:
    if is_video(video_path):
        video_capture = cv2.VideoCapture(video_path)
        if video_capture.isOpened():
            video_fps = video_capture.get(cv2.CAP_PROP_FPS)
            video_capture.release()
            return video_fps

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
