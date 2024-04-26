from typing import List, Optional

import subprocess

from .vision import pack_resolution

from .typing import Fps, FrameFormat, Resolution
from .filesystem import get_temp_frames_pattern

def run_ffmpeg(args : List[str]):
    commands = [ 'ffmpeg', '-hide_banner', '-loglevel', 'error' ]
    commands.extend(args)
    process = subprocess.Popen(commands, stderr = subprocess.PIPE, stdout = subprocess.PIPE)
    # TODO Make timeout be a option?
    return process.wait(timeout = 300) == 0

def extract_frames(video_path: str, frames_path: str, video_resolution : Resolution, video_fps : Fps, trim_frame_start : Optional[int] = None, trim_frame_end: Optional[int] = None, frame_format: FrameFormat = 'png') -> bool:
    # TODO Get frame image format from options.
    temp_frames_pattern = get_temp_frames_pattern(frames_path, '%04d', frame_format)
    commands = [ '-hwaccel', 'auto', '-i', video_path, '-q:v', '0' ]

    format_resolution = pack_resolution(video_resolution)

    if trim_frame_start is not None and trim_frame_end is not None:
        commands.extend([ '-vf', 'trim=start_frame=' + str(trim_frame_start) + ':end_frame=' + str(trim_frame_end) + ',scale=' + format_resolution + ',fps=' + str(video_fps) ])
    elif trim_frame_start is not None:
        commands.extend([ '-vf', 'trim=start_frame=' + str(trim_frame_start) + ',scale=' + format_resolution + ',fps=' + str(video_fps) ])
    elif trim_frame_end is not None:
        commands.extend([ '-vf', 'trim=end_frame=' + str(trim_frame_end) + ',scale=' + format_resolution + ',fps=' + str(video_fps) ])
    else:
        commands.extend([ '-vf', 'scale=' + format_resolution + ',fps=' + str(video_fps) ])
    commands.extend([ '-vsync', '0', temp_frames_pattern ])
    return run_ffmpeg(commands)
