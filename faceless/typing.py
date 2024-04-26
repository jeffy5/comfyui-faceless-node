from typing import Literal, Tuple, TypedDict


# Vision
Resolution = Tuple[int, int]
Fps = float

FrameFormat = Literal['jpg', 'png', 'bmp']

FacelessVideo = TypedDict('FacelessVideo', {
    'video_path': str,
    'frames_path': str,
})
