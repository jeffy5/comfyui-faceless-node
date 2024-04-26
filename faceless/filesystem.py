import os
import filetype

from .typing import FrameFormat

def is_file(file_path : str) -> bool:
    return bool(file_path and os.path.isfile(file_path))

def is_image(image_path : str) -> bool:
    return is_file(image_path) and filetype.helpers.is_image(image_path)


def is_video(video_path : str) -> bool:
    return is_file(video_path) and filetype.helpers.is_video(video_path)

def get_temp_frames_pattern(target_path : str, temp_frame_prefix : str, format: FrameFormat) -> str:
    return os.path.join(target_path, temp_frame_prefix + '.' + format)


