import os
import threading
from typing import Optional, Any, List
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
from ..typing import Embedding, Face, VisionFrame, FaceSelectorMode, ModelSet
from ..vision import read_image, write_image, tensor_to_vision_frame
from ..filesystem import get_faceless_model_path

THREAD_LOCK : threading.Lock = threading.Lock()

MODELS : ModelSet =\
{
    'blendswap_256':
    {
        'type': 'blendswap',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/blendswap_256.onnx',
        'template': 'ffhq_512',
        'size': (256, 256),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    },
    'inswapper_128':
    {
        'type': 'inswapper',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx',
        'template': 'arcface_128_v2',
        'size': (128, 128),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    },
    'inswapper_128_fp16':
    {
        'type': 'inswapper',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128_fp16.onnx',
        'template': 'arcface_128_v2',
        'size': (128, 128),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    },
    'simswap_256':
    {
        'type': 'simswap',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/simswap_256.onnx',
        'template': 'arcface_112_v1',
        'size': (256, 256),
        'mean': [ 0.485, 0.456, 0.406 ],
        'standard_deviation': [ 0.229, 0.224, 0.225 ]
    },
    'simswap_512_unofficial':
    {
        'type': 'simswap',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/simswap_512_unofficial.onnx',
        'template': 'arcface_112_v1',
        'size': (512, 512),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    },
    'uniface_256':
    {
        'type': 'uniface',
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/uniface_256.onnx',
        'template': 'ffhq_512',
        'size': (256, 256),
        'mean': [ 0.0, 0.0, 0.0 ],
        'standard_deviation': [ 1.0, 1.0, 1.0 ]
    }
}

class FaceSwapper:

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name

        self._face_selector_mode: FaceSelectorMode = 'many'

        self._execution_queue_count = 1
        self._execution_thread_count = 4

        self._frame_processor = None
        self._model_initializer = None

        self._face_mask_types = ['box']
        self._face_mask_blur = 0.3
        self._face_mask_padding = (0, 0, 0, 0)
        self._face_mask_regions = []

        self._face_selector_mode: FaceSelectorMode = 'one'

    def swap_images(self, images, face_image, output_path):
        source_frame = tensor_to_vision_frame(face_image)
        if source_frame is None:
            raise Exception('cannot read source image')
        source_face = get_average_face([source_frame])
        if source_face is None:
            raise Exception('cannot find source face')

        count = len(images)
        for (index, target_image) in enumerate(images):
            print(f"progress: {index + 1}/{count}")
            filename = f"{index + 1}".ljust(4, "0") + ".png"
            output_filepath = os.path.join(output_path, filename)

            target_vision_frame = tensor_to_vision_frame(target_image)
            if target_vision_frame is None:
                raise Exception("invalid target image")
            output_vision_frame = self._process_frame(source_face, source_frame, target_vision_frame)
            if output_vision_frame is None:
                raise Exception("process frame failed")
            write_image(output_filepath, output_vision_frame)

    def swap_video(self, source_image, target_frames_dir: str):
        frames_filenames = os.listdir(target_frames_dir)
        queue_payloads = sorted(frames_filenames)
        with ThreadPoolExecutor(max_workers = self._execution_thread_count) as executor:
            futures = []
            queue : Queue[str] = self._create_queue(queue_payloads)
            queue_per_future = max(len(queue_payloads) // self._execution_thread_count * self._execution_queue_count, 1)
            while not queue.empty():
                future = executor.submit(self._process_frames, source_image, target_frames_dir, self._pick_queue(queue, queue_per_future))
                futures.append(future)
            for future_done in as_completed(futures):
                future_done.result()

    def _process_frames(self, source_image, target_frames_dir: str, queue_payloads: List[str]):
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
            output_vision_frame = self._process_frame(source_face, source_frame, target_vision_frame)
            if output_vision_frame is None:
                raise Exception("process frame failed")
            write_image(frame_filepath, output_vision_frame)

    def _process_frame(self, source_face: Face, source_vision_frame: VisionFrame, target_vision_frame: VisionFrame) -> Optional[VisionFrame]:
        if self._face_selector_mode == 'many':
            target_faces = get_many_faces(target_vision_frame)
            for target_face in target_faces:
                target_vision_frame = self._swap_face(source_face, target_face, source_vision_frame, target_vision_frame)
        if self._face_selector_mode == 'one':
            target_face = get_one_face(target_vision_frame)
            if target_face:
                target_vision_frame = self._swap_face(source_face, target_face, source_vision_frame, target_vision_frame)
        return target_vision_frame

    def _create_queue(self, queue_payloads: List[str]) -> Queue[str]:
        queue: Queue[str] = Queue()
        for queue_payload in queue_payloads:
            queue.put(queue_payload)
        return queue

    def _pick_queue(self, queue : Queue[str], queue_per_future : int) -> List[str]:
        queues = []
        for _ in range(queue_per_future):
            if not queue.empty():
                queues.append(queue.get())
        return queues

    def _get_model_initializer(self) -> Any:
        with THREAD_LOCK:
            if self._model_initializer is None:
                model_path = get_faceless_model_path('face_swapper', self._model_name)
                model = onnx.load(model_path)
                self._model_initializer = numpy_helper.to_array(model.graph.initializer[-1])
        return self._model_initializer

    def _get_frame_processor(self) -> Any:
        with THREAD_LOCK:
            if self._frame_processor is None:
                model_path = get_faceless_model_path('face_swapper', self._model_name)
                if model_path is None:
                    raise Exception("can not get model path")
                self._frame_processor = onnxruntime.InferenceSession(model_path, providers = apply_execution_provider_options())
        return self._frame_processor

    def _swap_face(self, source_face: Face, target_face: Face, source_vision_frame, target_vision_frame: VisionFrame) -> VisionFrame:
        model_template = self._get_model_options().get('template')
        model_size = self._get_model_options().get('size')
        crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(target_vision_frame, target_face.landmarks.get('5/68'), model_template, model_size)
        crop_mask_list = []

        if 'box' in self._face_mask_types:
            box_mask = create_static_box_mask(crop_vision_frame.shape[:2][::-1], self._face_mask_blur, self._face_mask_padding)
            crop_mask_list.append(box_mask)
        if 'occlusion' in self._face_mask_types:
            occlusion_mask = create_occlusion_mask(crop_vision_frame)
            crop_mask_list.append(occlusion_mask)
        crop_vision_frame = self._prepare_crop_frame(crop_vision_frame)
        crop_vision_frame = self._apply_swap(source_face, source_vision_frame, crop_vision_frame)
        crop_vision_frame = self._normalize_crop_frame(crop_vision_frame)
        if 'region' in self._face_mask_types:
            region_mask = create_region_mask(crop_vision_frame, self._face_mask_regions)
            crop_mask_list.append(region_mask)
        crop_mask = numpy.minimum.reduce(crop_mask_list).clip(0, 1)
        target_vision_frame = paste_back(target_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
        return target_vision_frame

    def _prepare_crop_frame(self, crop_vision_frame : VisionFrame) -> VisionFrame:
        model_mean = self._get_model_options().get('mean')
        model_standard_deviation = self._get_model_options().get('standard_deviation')
        crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
        crop_vision_frame = (crop_vision_frame - model_mean) / model_standard_deviation
        crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
        crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis = 0).astype(numpy.float32)
        return crop_vision_frame

    def _apply_swap(self, source_face : Face, source_vision_frame: VisionFrame, crop_vision_frame : VisionFrame) -> VisionFrame:
        frame_processor = self._get_frame_processor()
        model_type = self._get_model_options().get('type')
        frame_processor_inputs = {}

        for frame_processor_input in frame_processor.get_inputs():
            if frame_processor_input.name == 'source':
                if model_type == 'blendswap' or model_type == 'uniface':
                    frame_processor_inputs[frame_processor_input.name] = self._prepare_source_frame(source_face, source_vision_frame)
                else:
                    frame_processor_inputs[frame_processor_input.name] = self._prepare_source_embedding(source_face)
            if frame_processor_input.name == 'target':
                frame_processor_inputs[frame_processor_input.name] = crop_vision_frame
        crop_vision_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
        return crop_vision_frame

    def _normalize_crop_frame(self, crop_vision_frame : VisionFrame) -> VisionFrame:
        crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
        crop_vision_frame = (crop_vision_frame * 255.0).round()
        crop_vision_frame = crop_vision_frame[:, :, ::-1]
        return crop_vision_frame

    def _prepare_source_frame(self, source_face : Face, source_vision_frame: VisionFrame) -> VisionFrame:
        model_type = self._get_model_options().get('type')
        if model_type == 'blendswap':
            source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame, source_face.landmarks.get('5/68'), 'arcface_112_v2', (112, 112))
        if model_type == 'uniface':
            source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame, source_face.landmarks.get('5/68'), 'ffhq_512', (256, 256))
        source_vision_frame = source_vision_frame[:, :, ::-1] / 255.0
        source_vision_frame = source_vision_frame.transpose(2, 0, 1)
        source_vision_frame = numpy.expand_dims(source_vision_frame, axis = 0).astype(numpy.float32)
        return source_vision_frame

    def _prepare_source_embedding(self, source_face : Face) -> Embedding:
        model_type = self._get_model_options().get('type')
        if model_type == 'inswapper':
            model_initializer = self._get_model_initializer()
            source_embedding = source_face.embedding.reshape((1, -1))
            source_embedding = numpy.dot(source_embedding, model_initializer) / numpy.linalg.norm(source_embedding)
        else:
            source_embedding = source_face.normed_embedding.reshape(1, -1)
        return source_embedding

    def _get_model_options(self) -> Any:
        names = os.path.splitext(self._model_name)
        return MODELS.get(names[0])
