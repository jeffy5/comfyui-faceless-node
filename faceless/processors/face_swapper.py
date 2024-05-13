import os
import threading
from typing import Optional, Literal, Any, List
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy
import onnx
from onnx import numpy_helper
import onnxruntime

from ..processors.face_analyser import get_average_face, get_many_faces, get_one_face
from ..face_helper import warp_face_by_face_landmark_5, paste_back
from ..face_masker import create_static_box_mask, create_occlusion_mask, create_region_mask
from ..execution import apply_execution_provider_options
from ..typing import Face, VisionFrame, FaceSelectorMode, ModelSet, OptionsWithModel, Embedding
from ..vision import read_image, write_image, tensor_to_vision_frame
from ..filesystem import resolve_relative_path

# TODO load from options
face_selector_mode: FaceSelectorMode = 'many'
face_mask_blur = 0.3
face_mask_padding = (0, 0, 0, 0)
face_mask_regions = []
face_mask_types = ['box']
face_swapper_model = 'inswapper_128'

execution_thread_count = 4
execution_queue_count = 1

execution_providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']

THREAD_LOCK : threading.Lock = threading.Lock()
MODEL_INITIALIZER = None
FRAME_PROCESSOR = None

model_template = 'arcface_128_v2'
model_size = (128, 128)

MODELS : ModelSet =\
{
    'blendswap_256':
    {
        'type': 'blendswap',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/blendswap_256.onnx',
        'path': resolve_relative_path('../../../models/faceless/blendswap_256.onnx'),
        'template': 'ffhq_512',
        'size': (256, 256),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    },
    'inswapper_128':
    {
        'type': 'inswapper',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx',
        'path': resolve_relative_path('../../../models/faceless/face_swapper/inswapper_128.onnx'),
        'template': 'arcface_128_v2',
        'size': (128, 128),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    },
    'inswapper_128_fp16':
    {
        'type': 'inswapper',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx',
        'path': resolve_relative_path('../../../models/faceless/face_swapper/inswapper_128_fp16.onnx'),
        'template': 'arcface_128_v2',
        'size': (128, 128),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    },
    'simswap_256':
    {
        'type': 'simswap',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/simswap_256.onnx',
        'path': resolve_relative_path('../../../models/faceless/face_swapper/simswap_256.onnx'),
        'template': 'arcface_112_v1',
        'size': (256, 256),
        'mean': [ 0.485, 0.456, 0.406 ],
        'standard_deviation': [ 0.229, 0.224, 0.225 ]
    },
    'simswap_512_unofficial':
    {
        'type': 'simswap',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/simswap_512_unofficial.onnx',
        'path': resolve_relative_path('../../../models/faceless/simswap_512_unofficial.onnx'),
        'template': 'arcface_112_v1',
        'size': (512, 512),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    },
    'uniface_256':
    {
        'type': 'uniface',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/uniface_256.onnx',
        'path': resolve_relative_path('../../../models/faceless/uniface_256.onnx'),
        'template': 'ffhq_512',
        'size': (256, 256),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    }
}
OPTIONS : Optional[OptionsWithModel] = None

def swap_face(source_face: Face, target_face: Face, source_vision_frame, target_vision_frame: VisionFrame) -> VisionFrame:
    crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(target_vision_frame, target_face.landmarks.get('5/68'), model_template, model_size)
    crop_mask_list = []

    if 'box' in face_mask_types:
        box_mask = create_static_box_mask(crop_vision_frame.shape[:2][::-1], face_mask_blur, face_mask_padding)
        crop_mask_list.append(box_mask)
    if 'occlusion' in face_mask_types:
        occlusion_mask = create_occlusion_mask(crop_vision_frame)
        crop_mask_list.append(occlusion_mask)
    crop_vision_frame = prepare_crop_frame(crop_vision_frame)
    crop_vision_frame = apply_swap(source_face, source_vision_frame, crop_vision_frame)
    crop_vision_frame = normalize_crop_frame(crop_vision_frame)
    if 'region' in face_mask_types:
        region_mask = create_region_mask(crop_vision_frame, face_mask_regions)
        crop_mask_list.append(region_mask)
    crop_mask = numpy.minimum.reduce(crop_mask_list).clip(0, 1)
    target_vision_frame = paste_back(target_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
    return target_vision_frame

def process_frame(source_face: Face, source_vision_frame: VisionFrame, target_vision_frame: VisionFrame) -> Optional[VisionFrame]:
    if face_selector_mode == 'many':
        target_faces = get_many_faces(target_vision_frame)
        for target_face in target_faces:
            target_vision_frame = swap_face(source_face, target_face, source_vision_frame, target_vision_frame)
    if face_selector_mode == 'one':
        target_face = get_one_face(target_vision_frame)
        if target_face:
            target_vision_frame = swap_face(source_face, target_face, source_vision_frame, target_vision_frame)
    return target_vision_frame

def process_images(source_image, target_images, output_frames_path):
    source_frame = tensor_to_vision_frame(source_image)
    if source_frame is None:
        raise Exception('cannot read source image')
    source_face = get_average_face([source_frame])
    if source_face is None:
        raise Exception('cannot find source face')

    count = len(target_images)
    for (index, target_image) in enumerate(target_images):
        print(f"progress: {index + 1}/{count}")
        filename = f"{index + 1}".ljust(4, "0") + ".png"
        output_filepath = os.path.join(output_frames_path, filename)

        target_vision_frame = tensor_to_vision_frame(target_image)
        if target_vision_frame is None:
            raise Exception("invalid target image")
        output_vision_frame = process_frame(source_face, source_frame, target_vision_frame)
        if output_vision_frame is None:
            raise Exception("process frame failed")
        write_image(output_filepath, output_vision_frame)

def process_frames(source_image, target_frames_dir: str, queue_payloads: List[str]):
    source_frame = tensor_to_vision_frame(source_image)
    if source_frame is None:
        raise Exception("cannot read source image")
    source_face = get_average_face([source_frame])
    if source_face is None:
        raise Exception("cannot find source face")

    count = len(queue_payloads)
    for index, frame_filename in enumerate(queue_payloads):
        print(f"progress: {index + 1}/{count}")
        frame_filepath = os.path.join(target_frames_dir, frame_filename)

        target_vision_frame = read_image(frame_filepath)
        if target_vision_frame is None:
            raise Exception("invalid target image")
        output_vision_frame = process_frame(source_face, source_frame, target_vision_frame)
        if output_vision_frame is None:
            raise Exception("process frame failed")
        write_image(frame_filepath, output_vision_frame)

def swap_video(source_image, target_frames_dir: str):
    frames_filenames = os.listdir(target_frames_dir)
    queue_payloads = sorted(frames_filenames)
    with ThreadPoolExecutor(max_workers = execution_thread_count) as executor:
        futures = []
        queue : Queue[str] = create_queue(queue_payloads)
        queue_per_future = max(len(queue_payloads) // execution_thread_count * execution_queue_count, 1)
        while not queue.empty():
            future = executor.submit(process_frames, source_image, target_frames_dir, pick_queue(queue, queue_per_future))
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

def apply_swap(source_face : Face, source_vision_frame: VisionFrame, crop_vision_frame : VisionFrame) -> VisionFrame:
    frame_processor = get_frame_processor()
    model_type = get_options('model').get('type')
    frame_processor_inputs = {}

    for frame_processor_input in frame_processor.get_inputs():
        if frame_processor_input.name == 'source':
            if model_type == 'blendswap' or model_type == 'uniface':
                frame_processor_inputs[frame_processor_input.name] = prepare_source_frame(source_face, source_vision_frame)
            else:
                frame_processor_inputs[frame_processor_input.name] = prepare_source_embedding(source_face)
        if frame_processor_input.name == 'target':
            frame_processor_inputs[frame_processor_input.name] = crop_vision_frame
    crop_vision_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
    return crop_vision_frame

def prepare_source_frame(source_face : Face, source_vision_frame: VisionFrame) -> VisionFrame:
    model_type = get_options('model').get('type')
    if model_type == 'blendswap':
        source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame, source_face.landmarks.get('5/68'), 'arcface_112_v2', (112, 112))
    if model_type == 'uniface':
        source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame, source_face.landmarks.get('5/68'), 'ffhq_512', (256, 256))
    source_vision_frame = source_vision_frame[:, :, ::-1] / 255.0
    source_vision_frame = source_vision_frame.transpose(2, 0, 1)
    source_vision_frame = numpy.expand_dims(source_vision_frame, axis = 0).astype(numpy.float32)
    return source_vision_frame


def prepare_source_embedding(source_face : Face) -> Embedding:
    model_type = get_options('model').get('type')
    if model_type == 'inswapper':
        model_initializer = get_model_initializer()
        source_embedding = source_face.embedding.reshape((1, -1))
        source_embedding = numpy.dot(source_embedding, model_initializer) / numpy.linalg.norm(source_embedding)
    else:
        source_embedding = source_face.normed_embedding.reshape(1, -1)
    return source_embedding


def prepare_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
    model_mean = get_options('model').get('mean')
    model_standard_deviation = get_options('model').get('standard_deviation')
    crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
    crop_vision_frame = (crop_vision_frame - model_mean) / model_standard_deviation
    crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
    crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis = 0).astype(numpy.float32)
    return crop_vision_frame


def normalize_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
    crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
    crop_vision_frame = (crop_vision_frame * 255.0).round()
    crop_vision_frame = crop_vision_frame[:, :, ::-1]
    return crop_vision_frame


def get_options(key : Literal['model']) -> Any:
    global OPTIONS

    if OPTIONS is None:
        OPTIONS =\
        {
            'model': MODELS[face_swapper_model]
        }
    return OPTIONS.get(key)


def get_model_initializer() -> Any:
    global MODEL_INITIALIZER

    with THREAD_LOCK:
        if MODEL_INITIALIZER is None:
            model_path = get_options('model').get('path')
            model = onnx.load(model_path)
            MODEL_INITIALIZER = numpy_helper.to_array(model.graph.initializer[-1])
    return MODEL_INITIALIZER


def get_frame_processor() -> Any:
    global FRAME_PROCESSOR

    with THREAD_LOCK:
        if FRAME_PROCESSOR is None:
            model_path = get_options('model').get('path')
            FRAME_PROCESSOR = onnxruntime.InferenceSession(model_path, providers = apply_execution_provider_options(execution_providers))
    return FRAME_PROCESSOR
