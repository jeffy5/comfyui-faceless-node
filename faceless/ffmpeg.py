from typing import List, Optional

import os
import subprocess

from .vision import pack_resolution, restrict_video_fps

from .typing import Fps, FrameFormat, OutputVideoEncoder, OutputVideoPreset, Resolution
from .filesystem import get_temp_frames_pattern


def run_ffmpeg(args : List[str]):

    ffmpeg_path = os.environ.get("COMFYUI_FACELESS_FFMPEG") or "/usr/bin/ffmpeg"
    commands = [ ffmpeg_path, '-hide_banner', '-loglevel', 'error' ]
    commands.extend(args)
    process = subprocess.Popen(commands, stderr = subprocess.PIPE, stdout = subprocess.PIPE)
    code = process.wait(timeout = 300)

    if code != 0:
        msg = f"code: {code}"
        stderr = "stderr: "
        stdout = "stdout: "
        if process.stderr is not None:
            stderr += process.stderr.readline().decode('utf-8')
        if process.stdout is not None:
            stdout += process.stdout.readline().decode('utf-8')
        print(', '.join([msg, stderr, stdout]))
    return code == 0

def extract_frames(video_path: str, frames_path: str, video_resolution : Resolution, video_fps : Fps, trim_frame_start : Optional[int] = None, trim_frame_end: Optional[int] = None, frame_format: FrameFormat = 'png') -> bool:
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

def merge_frames(video_path: str, frames_dir: str, output_path: str, video_resolution: Resolution, video_fps: Fps, output_video_encoder: OutputVideoEncoder = 'libx264', output_video_quality: int = 80, output_video_preset: OutputVideoPreset = 'veryfast', frame_format: FrameFormat = 'png') -> bool:
    temp_video_fps = restrict_video_fps(video_path, video_fps)
    temp_frames_pattern = get_temp_frames_pattern(frames_dir, '%04d', frame_format)
    commands = [ '-hwaccel', 'auto', '-s', pack_resolution(video_resolution), '-r', str(temp_video_fps), '-i', temp_frames_pattern, '-c:v', output_video_encoder ]

    if output_video_encoder in [ 'libx264', 'libx265' ]:
        output_video_compression = round(51 - (output_video_quality * 0.51))
        commands.extend([ '-crf', str(output_video_compression), '-preset', output_video_preset ])
    if output_video_encoder in [ 'libvpx-vp9' ]:
        output_video_compression = round(63 - (output_video_quality * 0.63))
        commands.extend([ '-crf', str(output_video_compression) ])
    if output_video_encoder in [ 'h264_nvenc', 'hevc_nvenc' ]:
        output_video_compression = round(51 - (output_video_quality * 0.51))
        commands.extend([ '-cq', str(output_video_compression), '-preset', output_video_preset ])
    if output_video_encoder in [ 'h264_amf', 'hevc_amf' ]:
        output_video_compression = round(51 - (output_video_quality * 0.51))
        commands.extend([ '-qp_i', str(output_video_compression), '-qp_p', str(output_video_compression), '-quality', map_amf_preset(output_video_preset) ])
    commands.extend([ '-vf', 'framerate=fps=' + str(video_fps), '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-y', output_path ])
    return run_ffmpeg(commands)

# ffmpeg -i ~/Downloads/temp/bg.mp4 -framerate 30 -i %04d.png -filter_complex "[0:v]scale=528:960:force_original_aspect_ratio=increase,crop=528:960[bg];[1:v]format=rgba[overlay];[bg][overlay]overlay=shortest=1" -pix_fmt yuv420p -c:a copy output_final.mp4
def merge_frames_and_bg_video(video_path: str, bg_video_path: str, frames_dir: str, output_path: str, video_resolution: Resolution, video_fps: Fps, output_video_encoder: OutputVideoEncoder = 'libx264', output_video_quality: int = 80, output_video_preset: OutputVideoPreset = 'veryfast', frame_format: FrameFormat = 'png'):
    temp_video_fps = restrict_video_fps(video_path, video_fps)
    temp_frames_pattern = get_temp_frames_pattern(frames_dir, '%04d', frame_format)
    video_resolution[0]
    commands = [ '-hwaccel', 'auto', '-r', str(temp_video_fps), '-i', bg_video_path, '-i', temp_frames_pattern, '-c:v', output_video_encoder ]

    width = video_resolution[0]
    height = video_resolution[1]
    commands.extend(["-filter_complex", f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}[bg];[1:v]format=rgba[overlay];[bg][overlay]overlay=shortest=1"])

    if output_video_encoder in [ 'libx264', 'libx265' ]:
        output_video_compression = round(51 - (output_video_quality * 0.51))
        commands.extend([ '-crf', str(output_video_compression), '-preset', output_video_preset ])
    if output_video_encoder in [ 'libvpx-vp9' ]:
        output_video_compression = round(63 - (output_video_quality * 0.63))
        commands.extend([ '-crf', str(output_video_compression) ])
    if output_video_encoder in [ 'h264_nvenc', 'hevc_nvenc' ]:
        output_video_compression = round(51 - (output_video_quality * 0.51))
        commands.extend([ '-cq', str(output_video_compression), '-preset', output_video_preset ])
    if output_video_encoder in [ 'h264_amf', 'hevc_amf' ]:
        output_video_compression = round(51 - (output_video_quality * 0.51))
        commands.extend([ '-qp_i', str(output_video_compression), '-qp_p', str(output_video_compression), '-quality', map_amf_preset(output_video_preset) ])
    commands.extend(['-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-y', output_path])
    return run_ffmpeg(commands)

# ffmpeg -i bg.mp4 -i fg.mov -filter_complex "[0:v]scale=528:960:force_original_aspect_ratio=increase,crop=528:960[bg];[1:v]scale=528:960:force_original_aspect_ratio=increase,crop=528:960[fg];[bg][fg]overlay=shortest=1" -c:a copy output_final.mp4
def merge_videos(fg_video_path: str, bg_video_path: str, output_path: str, resolution: Resolution):
    width = resolution[0]
    height = resolution[1]
    filter_complex = f"[0:v]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}[bg];[1:v]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}[fg];[bg][fg]overlay=shortest=1"
    commands = ["-i", bg_video_path, "-i", fg_video_path, "-filter_complex", filter_complex, "-c:a", "copy", "-y", output_path]
    return run_ffmpeg(commands)

def restore_audio(video_path: str, audio_path: str, output_path: str, output_video_fps: Fps, trim_frame_start: Optional[int] = None, trim_frame_end: Optional[int] = None) -> bool:
    commands = [ '-hwaccel', 'auto', '-i', video_path ]

    if trim_frame_start is not None:
        start_time = trim_frame_start / output_video_fps
        commands.extend([ '-ss', str(start_time) ])
    if trim_frame_end is not None:
        end_time = trim_frame_end / output_video_fps
        commands.extend([ '-to', str(end_time) ])
    commands.extend([ '-i', audio_path, '-c', 'copy', '-map', '0:v:0', '-map', '1:a?:0', '-shortest', '-y', output_path ])
    return run_ffmpeg(commands)

def map_amf_preset(output_video_preset : OutputVideoPreset) -> str:
    if output_video_preset in [ 'ultrafast', 'superfast', 'veryfast' ]:
        return 'speed'
    if output_video_preset in [ 'faster', 'fast', 'medium' ]:
        return 'balanced'
    if output_video_preset in [ 'slow', 'slower', 'veryslow' ]:
        return 'quality'
    return 'balanced'
