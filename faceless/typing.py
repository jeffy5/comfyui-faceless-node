from typing import Literal, Tuple, TypedDict, Any, Dict, List, Optional
from collections import namedtuple

import numpy


BoundingBox = numpy.ndarray[Any, Any]
FaceLandmark5 = numpy.ndarray[Any, Any]
FaceLandmark68 = numpy.ndarray[Any, Any]
FaceLandmarkSet = TypedDict('FaceLandmarkSet',
{
    '5' : FaceLandmark5, # type: ignore[valid-type]
    '5/68' : FaceLandmark5, # type: ignore[valid-type]
    '68' : FaceLandmark68, # type: ignore[valid-type]
    '68/5' : FaceLandmark68 # type: ignore[valid-type]
})
Score = float
FaceScoreSet = TypedDict('FaceScoreSet',
{
    'detector' : Score,
    'landmarker' : Score
})

WarpTemplate = Literal['arcface_112_v1', 'arcface_112_v2', 'arcface_128_v2', 'ffhq_512']
WarpTemplateSet = Dict[WarpTemplate, numpy.ndarray[Any, Any]]


# Vision
Fps = float
Padding = Tuple[int, int, int, int]
Resolution = Tuple[int, int]

VisionFrame = numpy.ndarray[Any, Any]
Mask = numpy.ndarray[Any, Any]
Matrix = numpy.ndarray[Any, Any]
Translation = numpy.ndarray[Any, Any]

FrameFormat = Literal['jpg', 'png', 'bmp']

FacelessVideo = TypedDict('FacelessVideo', {
    # raw vidoe file path
    'video_path': str,
    # frames dir
    'frames_dir': str,
    # output vidoe file path
    'output_path': str,
    'resolution': Resolution,
    'fps': Fps,
    'trim_frame_start': Optional[int],
    'trim_frame_end': Optional[int],
})

Embedding = numpy.ndarray[Any, Any]
Face = namedtuple('Face',
[
    'bounding_box',
    'landmarks',
    'scores',
    'embedding',
    'normed_embedding',
    'gender',
    'age'
])

# Face Store
FaceSet = Dict[str, List[Face]]
FaceStore = TypedDict('FaceStore',
{
    'static_faces' : FaceSet,
    'reference_faces': FaceSet
})


ModelType = Literal['face_swapper', 'face_detector', 'face_recognizer', 'face_landmarker']
FaceDetectorModel = Literal['many', 'retinaface', 'scrfd', 'yoloface', 'yunet']
FaceRecognizerModel = Literal['arcface_blendswap', 'arcface_inswapper', 'arcface_simswap', 'arcface_uniface']

ModelValue = Dict[str, Any]
ModelSet = Dict[str, ModelValue]
OptionsWithModel = TypedDict('OptionsWithModel',
{
	'model' : ModelValue
})

ValueAndUnit = TypedDict('ValueAndUnit',
{
    'value' : str,
    'unit' : str
})
ExecutionDeviceFramework = TypedDict('ExecutionDeviceFramework',
{
    'name' : str,
    'version' : str
})
ExecutionDeviceProduct = TypedDict('ExecutionDeviceProduct',
{
    'vendor' : str,
    'name' : str,
    'architecture' : str,
})
ExecutionDeviceVideoMemory = TypedDict('ExecutionDeviceVideoMemory',
{
    'total' : ValueAndUnit,
    'free' : ValueAndUnit
})
ExecutionDeviceUtilization = TypedDict('ExecutionDeviceUtilization',
{
    'gpu' : ValueAndUnit,
    'memory' : ValueAndUnit
})
ExecutionDevice = TypedDict('ExecutionDevice',
{
    'driver_version' : str,
    'framework' : ExecutionDeviceFramework,
    'product' : ExecutionDeviceProduct,
    'video_memory' : ExecutionDeviceVideoMemory,
    'utilization' : ExecutionDeviceUtilization
})


FaceAnalyserOrder = Literal['left-right', 'right-left', 'top-bottom', 'bottom-top', 'small-large', 'large-small', 'best-worst', 'worst-best']
FaceAnalyserAge = Literal['child', 'teen', 'adult', 'senior']
FaceAnalyserGender = Literal['female', 'male']
FaceSelectorMode = Literal['many', 'one', 'reference']

FaceMaskRegion = Literal['skin', 'left-eyebrow', 'right-eyebrow', 'left-eye', 'right-eye', 'glasses', 'nose', 'mouth', 'upper-lip', 'lower-lip']

OutputVideoEncoder = Literal['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc', 'h264_amf', 'hevc_amf']
OutputVideoPreset = Literal['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']
