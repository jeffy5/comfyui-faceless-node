import os
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from typing import List, Optional, Literal

import cv2
import numpy
import onnxruntime

from ..processors.face_analyser import get_many_faces
from ..execution import apply_execution_provider_options
from ..face_helper import warp_face_by_face_landmark_5, paste_back
from ..face_masker import create_static_box_mask, create_occlusion_mask
from ..vision import read_image, write_image
from ..filesystem import resolve_relative_path
from ..typing import VisionFrame, ModelSet, Any, OptionsWithModel

execution_thread_count = 4
execution_queue_count = 1
execution_providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']

model_template = "ffhq_512"
model_size = (512, 512)
model_path = (512, 512)
face_mask_blur = 0.3
face_mask_types = ['box']
face_enhancer_blend = 80
face_enhancer_model = 'gfpgan_1.4'

THREAD_LOCK : threading.Lock = threading.Lock()
THREAD_SEMAPHORE : threading.Semaphore = threading.Semaphore()
FRAME_PROCESSOR = None
MODELS : ModelSet =\
{
    'codeformer':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/codeformer.onnx',
        'path': resolve_relative_path('../../../models/faceless/codeformer.onnx'),
        'template': 'ffhq_512',
        'size': (512, 512)
    },
    'gfpgan_1.2':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.2.onnx',
        'path': resolve_relative_path('../../../models/faceless/gfpgan_1.2.onnx'),
        'template': 'ffhq_512',
        'size': (512, 512)
    },
    'gfpgan_1.3':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.3.onnx',
        'path': resolve_relative_path('../../../models/faceless/gfpgan_1.3.onnx'),
        'template': 'ffhq_512',
        'size': (512, 512)
    },
    'gfpgan_1.4':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gfpgan_1.4.onnx',
        'path': resolve_relative_path('../../../models/faceless/gfpgan_1.4.onnx'),
        'template': 'ffhq_512',
        'size': (512, 512)
    },
    'gpen_bfr_256':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_256.onnx',
        'path': resolve_relative_path('../../../models/faceless/gpen_bfr_256.onnx'),
        'template': 'arcface_128_v2',
        'size': (256, 256)
    },
    'gpen_bfr_512':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_512.onnx',
        'path': resolve_relative_path('../../../models/faceless/gpen_bfr_512.onnx'),
        'template': 'ffhq_512',
        'size': (512, 512)
    },
    'gpen_bfr_1024':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_1024.onnx',
        'path': resolve_relative_path('../../../models/faceless/gpen_bfr_1024.onnx'),
        'template': 'ffhq_512',
        'size': (1024, 1024)
    },
    'gpen_bfr_2048':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gpen_bfr_2048.onnx',
        'path': resolve_relative_path('../../../models/faceless/gpen_bfr_2048.onnx'),
        'template': 'ffhq_512',
        'size': (2048, 2048)
    },
    'restoreformer_plus_plus':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/restoreformer_plus_plus.onnx',
        'path': resolve_relative_path('../../../models/faceless/restoreformer_plus_plus.onnx'),
        'template': 'ffhq_512',
        'size': (512, 512)
    }
}
OPTIONS : Optional[OptionsWithModel] = None

def apply_enhance(crop_vision_frame : VisionFrame) -> VisionFrame:
    frame_processor = get_frame_processor()
    frame_processor_inputs = {}

    for frame_processor_input in frame_processor.get_inputs():
        if frame_processor_input.name == 'input':
            frame_processor_inputs[frame_processor_input.name] = crop_vision_frame
        if frame_processor_input.name == 'weight':
            weight = numpy.array([ 1 ]).astype(numpy.double)
            frame_processor_inputs[frame_processor_input.name] = weight
    with THREAD_SEMAPHORE:
        crop_vision_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
    return crop_vision_frame

def enhance_face(face, frame: VisionFrame) -> VisionFrame:
    crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(frame, face.landmarks.get('5/68'), model_template, model_size)
    box_mask = create_static_box_mask(crop_vision_frame.shape[:2][::-1], face_mask_blur, (0, 0, 0, 0))
    crop_mask_list =\
    [
        box_mask
    ]

    if 'occlusion' in face_mask_types:
        occlusion_mask = create_occlusion_mask(crop_vision_frame)
        crop_mask_list.append(occlusion_mask)
    crop_vision_frame = prepare_crop_frame(crop_vision_frame)
    crop_vision_frame = apply_enhance(crop_vision_frame)
    crop_vision_frame = normalize_crop_frame(crop_vision_frame)
    crop_mask = numpy.minimum.reduce(crop_mask_list).clip(0, 1)
    paste_vision_frame = paste_back(frame, crop_vision_frame, crop_mask, affine_matrix)
    temp_vision_frame = blend_frame(frame, paste_vision_frame)
    return temp_vision_frame


def process_frame(frame: VisionFrame):
    # Support one face and many face mode
    faces = get_many_faces(frame)
    target_vision_frame = None
    for face in faces:
        target_vision_frame = enhance_face(face, frame)
    return target_vision_frame

def process_frames(target_frames_dir: str, queue_payloads: List[str]):
    count = len(queue_payloads)
    for index, frame_filename in enumerate(queue_payloads):
        print(f"progress: {index + 1}/{count}")
        frame_filepath = os.path.join(target_frames_dir, frame_filename)

        target_vision_frame = read_image(frame_filepath)
        if target_vision_frame is None:
            raise Exception("invalid target image")
        output_vision_frame = process_frame(target_vision_frame)
        if output_vision_frame is None:
            continue
            # raise Exception("process frame failed")
        write_image(frame_filepath, output_vision_frame)

def enhance_video(frames_dir: str):
    frames_filenames = os.listdir(frames_dir)
    queue_payloads = sorted(frames_filenames)

    with ThreadPoolExecutor(max_workers = execution_thread_count) as executor:
        futures = []
        queue : Queue[str] = create_queue(queue_payloads)
        queue_per_future = max(len(queue_payloads) // execution_thread_count * execution_queue_count, 1)
        while not queue.empty():
            future = executor.submit(process_frames, frames_dir, pick_queue(queue, queue_per_future))
            futures.append(future)
        for future_done in as_completed(futures):
            future_done.result()

def create_queue(queue_payloads : List[str]) -> Queue[str]:
    queue : Queue[str] = Queue()
    for queue_payload in queue_payloads:
        queue.put(queue_payload)
    return queue

def pick_queue(queue : Queue[str], queue_per_future : int) -> List[str]:
    queues = []
    for _ in range(queue_per_future):
        if not queue.empty():
            queues.append(queue.get())
    return queues


def prepare_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
    crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
    crop_vision_frame = (crop_vision_frame - 0.5) / 0.5
    crop_vision_frame = numpy.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
    return crop_vision_frame

def normalize_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
    crop_vision_frame = numpy.clip(crop_vision_frame, -1, 1)
    crop_vision_frame = (crop_vision_frame + 1) / 2
    crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
    crop_vision_frame = (crop_vision_frame * 255.0).round()
    crop_vision_frame = crop_vision_frame.astype(numpy.uint8)[:, :, ::-1]
    return crop_vision_frame

def blend_frame(temp_vision_frame : VisionFrame, paste_vision_frame : VisionFrame) -> VisionFrame:
    final_face_enhancer_blend = 1 - (face_enhancer_blend / 100)
    temp_vision_frame = cv2.addWeighted(temp_vision_frame, final_face_enhancer_blend, paste_vision_frame, 1 - final_face_enhancer_blend, 0)
    return temp_vision_frame

def get_options(key : Literal['model']) -> Any:
    global OPTIONS

    if OPTIONS is None:
        OPTIONS =\
        {
            'model': MODELS[face_enhancer_model]
        }
    return OPTIONS.get(key)

def get_frame_processor() -> Any:
    global FRAME_PROCESSOR

    with THREAD_LOCK:
        if FRAME_PROCESSOR is None:
            model_path = get_options('model').get('path')
            FRAME_PROCESSOR = onnxruntime.InferenceSession(model_path, providers = apply_execution_provider_options(execution_providers))
    return FRAME_PROCESSOR
