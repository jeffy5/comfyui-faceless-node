from typing import List, Optional, Tuple, Any

import numpy
import cv2
import threading
import onnxruntime
import traceback

from ..face_store import get_static_faces, set_static_faces
from ..face_helper import create_static_anchors, distance_to_bounding_box, distance_to_face_landmark_5, warp_face_by_face_landmark_5, warp_face_by_translation, estimate_matrix_by_face_landmark_5, categorize_age, categorize_gender, apply_nms, convert_face_landmark_68_to_5
from ..execution import apply_execution_provider_options
from ..vision import unpack_resolution, resize_frame_resolution
from ..filesystem import resolve_relative_path
from ..typing import FaceLandmark68, FaceLandmarkSet, FaceScoreSet, FaceRecognizerModel, VisionFrame, Face, FaceDetectorModel, BoundingBox, FaceLandmark5, Score, ModelSet, FaceAnalyserOrder, FaceAnalyserAge, FaceAnalyserGender, Embedding

THREAD_SEMAPHORE : threading.Semaphore = threading.Semaphore()
THREAD_LOCK : threading.Lock = threading.Lock()

FACE_ANALYSER = None

# TODO load from options
face_detector_model: FaceDetectorModel = "yoloface"
face_recognizer_model: Optional[FaceRecognizerModel] = 'arcface_inswapper'
face_detector_score = 0.5
face_landmarker_score = 0.5
face_detector_size = '640x640'
face_analyser_order: FaceAnalyserOrder = 'left-right'
face_analyser_age: Optional[FaceAnalyserAge] = None
face_analyser_gender: Optional[FaceAnalyserGender] = None

MODELS : ModelSet =\
{
    'face_detector_retinaface':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/retinaface_10g.onnx',
        'path': resolve_relative_path('../../../models/faceless/retinaface_10g.onnx')
    },
    'face_detector_scrfd':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/scrfd_2.5g.onnx',
        'path': resolve_relative_path('../../../models/faceless/scrfd_2.5g.onnx')
    },
    'face_detector_yoloface':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/yoloface_8n.onnx',
        'path': resolve_relative_path('../../../models/faceless/face_detector/yoloface_8n.onnx')
    },
    'face_detector_yunet':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/yunet_2023mar.onnx',
        'path': resolve_relative_path('../../../models/faceless/yunet_2023mar.onnx')
    },
    'face_recognizer_arcface_blendswap':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx',
        'path': resolve_relative_path('../../../models/faceless/face_recognizer/arcface_w600k_r50.onnx')
    },
    'face_recognizer_arcface_inswapper':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx',
        'path': resolve_relative_path('../../../models/faceless/face_recognizer/arcface_w600k_r50.onnx')
    },
    'face_recognizer_arcface_simswap':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_simswap.onnx',
        'path': resolve_relative_path('../../../models/faceless/arcface_simswap.onnx')
    },
    'face_recognizer_arcface_uniface':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/arcface_w600k_r50.onnx',
        'path': resolve_relative_path('../../../models/faceless/arcface_w600k_r50.onnx')
    },
    'face_landmarker_68':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/2dfan4.onnx',
        'path': resolve_relative_path('../../../models/faceless/face_landmarker/2dfan4.onnx')
    },
    'face_landmarker_68_5':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/face_landmarker_68_5.onnx',
        'path': resolve_relative_path('../../../models/faceless/face_landmarker/face_landmarker_68_5.onnx')
    },
    'gender_age':
    {
        'url': 'https://github.com/facefusion/facefusion-assets/releases/download/models/gender_age.onnx',
        'path': resolve_relative_path('../../../models/faceless/gender_age.onnx')
    }
}

def get_face_analyser() -> Any:
    global FACE_ANALYSER

    face_detectors = {}
    face_landmarkers = {}
    face_recognizer = None

    with THREAD_LOCK:
        if FACE_ANALYSER is None:
            if face_detector_model in [ 'many', 'retinaface' ]:
                face_detectors['retinaface'] = onnxruntime.InferenceSession(MODELS['face_detector_retinaface']['path'], providers = apply_execution_provider_options())
            if face_detector_model in [ 'many', 'scrfd' ]:
                face_detectors['scrfd'] = onnxruntime.InferenceSession(MODELS['face_detector_scrfd']['path'], providers = apply_execution_provider_options())
            if face_detector_model in [ 'many', 'yoloface' ]:
                face_detectors['yoloface'] = onnxruntime.InferenceSession(MODELS['face_detector_yoloface']['path'], providers = apply_execution_provider_options())
            if face_detector_model in [ 'yunet' ]:
                face_detectors['yunet'] = cv2.FaceDetectorYN.create(MODELS['face_detector_yunet']['path'], '', (0, 0))
            if face_recognizer_model == 'arcface_blendswap':
                face_recognizer = onnxruntime.InferenceSession(MODELS['face_recognizer_arcface_blendswap']['path'], providers = apply_execution_provider_options())
            if face_recognizer_model == 'arcface_inswapper':
                face_recognizer = onnxruntime.InferenceSession(MODELS['face_recognizer_arcface_inswapper']['path'], providers = apply_execution_provider_options())
            if face_recognizer_model == 'arcface_simswap':
                face_recognizer = onnxruntime.InferenceSession(MODELS['face_recognizer_arcface_simswap']['path'], providers = apply_execution_provider_options())
            if face_recognizer_model == 'arcface_uniface':
                face_recognizer = onnxruntime.InferenceSession(MODELS['face_recognizer_arcface_uniface']['path'], providers = apply_execution_provider_options())
            face_landmarkers['68'] = onnxruntime.InferenceSession(MODELS['face_landmarker_68']['path'], providers = apply_execution_provider_options())
            face_landmarkers['68_5'] = onnxruntime.InferenceSession(MODELS['face_landmarker_68_5']['path'], providers = apply_execution_provider_options())
            gender_age = onnxruntime.InferenceSession(MODELS['gender_age']['path'], providers = apply_execution_provider_options())
            FACE_ANALYSER =\
            {
                'face_detectors': face_detectors,
                'face_recognizer': face_recognizer,
                'face_landmarkers': face_landmarkers,
                'gender_age': gender_age
            }
    return FACE_ANALYSER

def detect_with_retinaface(vision_frame : VisionFrame, face_detector_size : str) -> Tuple[List[BoundingBox], List[FaceLandmark5], List[Score]]:
    face_detector = get_face_analyser().get('face_detectors').get('retinaface')
    face_detector_width, face_detector_height = unpack_resolution(face_detector_size)
    temp_vision_frame = resize_frame_resolution(vision_frame, (face_detector_width, face_detector_height))
    ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
    ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
    feature_strides = [ 8, 16, 32 ]
    feature_map_channel = 3
    anchor_total = 2
    bounding_box_list = []
    face_landmark_5_list = []
    score_list = []

    detect_vision_frame = prepare_detect_frame(temp_vision_frame, face_detector_size)
    with THREAD_SEMAPHORE:
        detections = face_detector.run(None,
        {
            face_detector.get_inputs()[0].name: detect_vision_frame
        })
    for index, feature_stride in enumerate(feature_strides):
        keep_indices = numpy.where(detections[index] >= face_detector_score)[0]
        if keep_indices.any():
            stride_height = face_detector_height // feature_stride
            stride_width = face_detector_width // feature_stride
            anchors = create_static_anchors(feature_stride, anchor_total, stride_height, stride_width)
            bounding_box_raw = detections[index + feature_map_channel] * feature_stride
            face_landmark_5_raw = detections[index + feature_map_channel * 2] * feature_stride
            for bounding_box in distance_to_bounding_box(anchors, bounding_box_raw)[keep_indices]:
                bounding_box_list.append(numpy.array(
                [
                    bounding_box[0] * ratio_width,
                    bounding_box[1] * ratio_height,
                    bounding_box[2] * ratio_width,
                    bounding_box[3] * ratio_height
                ]))
            for face_landmark_5 in distance_to_face_landmark_5(anchors, face_landmark_5_raw)[keep_indices]:
                face_landmark_5_list.append(face_landmark_5 * [ ratio_width, ratio_height ])
            for score in detections[index][keep_indices]:
                score_list.append(score[0])
    return bounding_box_list, face_landmark_5_list, score_list


def detect_with_scrfd(vision_frame : VisionFrame, face_detector_size : str) -> Tuple[List[BoundingBox], List[FaceLandmark5], List[Score]]:
    face_detector = get_face_analyser().get('face_detectors').get('scrfd')
    face_detector_width, face_detector_height = unpack_resolution(face_detector_size)
    temp_vision_frame = resize_frame_resolution(vision_frame, (face_detector_width, face_detector_height))
    ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
    ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
    feature_strides = [ 8, 16, 32 ]
    feature_map_channel = 3
    anchor_total = 2
    bounding_box_list = []
    face_landmark_5_list = []
    score_list = []

    detect_vision_frame = prepare_detect_frame(temp_vision_frame, face_detector_size)
    with THREAD_SEMAPHORE:
        detections = face_detector.run(None,
        {
            face_detector.get_inputs()[0].name: detect_vision_frame
        })
    for index, feature_stride in enumerate(feature_strides):
        keep_indices = numpy.where(detections[index] >= face_detector_score)[0]
        if keep_indices.any():
            stride_height = face_detector_height // feature_stride
            stride_width = face_detector_width // feature_stride
            anchors = create_static_anchors(feature_stride, anchor_total, stride_height, stride_width)
            bounding_box_raw = detections[index + feature_map_channel] * feature_stride
            face_landmark_5_raw = detections[index + feature_map_channel * 2] * feature_stride
            for bounding_box in distance_to_bounding_box(anchors, bounding_box_raw)[keep_indices]:
                bounding_box_list.append(numpy.array(
                [
                    bounding_box[0] * ratio_width,
                    bounding_box[1] * ratio_height,
                    bounding_box[2] * ratio_width,
                    bounding_box[3] * ratio_height
                ]))
            for face_landmark_5 in distance_to_face_landmark_5(anchors, face_landmark_5_raw)[keep_indices]:
                face_landmark_5_list.append(face_landmark_5 * [ ratio_width, ratio_height ])
            for score in detections[index][keep_indices]:
                score_list.append(score[0])
    return bounding_box_list, face_landmark_5_list, score_list


def detect_with_yoloface(vision_frame : VisionFrame, face_detector_size : str) -> Tuple[List[BoundingBox], List[FaceLandmark5], List[Score]]:
    face_detector = get_face_analyser().get('face_detectors').get('yoloface')
    face_detector_width, face_detector_height = unpack_resolution(face_detector_size)
    temp_vision_frame = resize_frame_resolution(vision_frame, (face_detector_width, face_detector_height))
    ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
    ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
    bounding_box_list = []
    face_landmark_5_list = []
    score_list = []

    detect_vision_frame = prepare_detect_frame(temp_vision_frame, face_detector_size)
    with THREAD_SEMAPHORE:
        detections = face_detector.run(None,
        {
            face_detector.get_inputs()[0].name: detect_vision_frame
        })
    detections = numpy.squeeze(detections).T
    bounding_box_raw, score_raw, face_landmark_5_raw = numpy.split(detections, [ 4, 5 ], axis = 1)
    keep_indices = numpy.where(score_raw > face_detector_score)[0]
    if keep_indices.any():
        bounding_box_raw, face_landmark_5_raw, score_raw = bounding_box_raw[keep_indices], face_landmark_5_raw[keep_indices], score_raw[keep_indices]
        for bounding_box in bounding_box_raw:
            bounding_box_list.append(numpy.array(
            [
                (bounding_box[0] - bounding_box[2] / 2) * ratio_width,
                (bounding_box[1] - bounding_box[3] / 2) * ratio_height,
                (bounding_box[0] + bounding_box[2] / 2) * ratio_width,
                (bounding_box[1] + bounding_box[3] / 2) * ratio_height
            ]))
        face_landmark_5_raw[:, 0::3] = (face_landmark_5_raw[:, 0::3]) * ratio_width
        face_landmark_5_raw[:, 1::3] = (face_landmark_5_raw[:, 1::3]) * ratio_height
        for face_landmark_5 in face_landmark_5_raw:
            face_landmark_5_list.append(numpy.array(face_landmark_5.reshape(-1, 3)[:, :2]))
        score_list = score_raw.ravel().tolist()
    return bounding_box_list, face_landmark_5_list, score_list


def detect_with_yunet(vision_frame : VisionFrame, face_detector_size : str) -> Tuple[List[BoundingBox], List[FaceLandmark5], List[Score]]:
    face_detector = get_face_analyser().get('face_detectors').get('yunet')
    face_detector_width, face_detector_height = unpack_resolution(face_detector_size)
    temp_vision_frame = resize_frame_resolution(vision_frame, (face_detector_width, face_detector_height))
    ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
    ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
    bounding_box_list = []
    face_landmark_5_list = []
    score_list = []

    face_detector.setInputSize((temp_vision_frame.shape[1], temp_vision_frame.shape[0]))
    face_detector.setScoreThreshold(face_detector_score)
    with THREAD_SEMAPHORE:
        _, detections = face_detector.detect(temp_vision_frame)
    if numpy.any(detections):
        for detection in detections:
            bounding_box_list.append(numpy.array(
            [
                detection[0] * ratio_width,
                detection[1] * ratio_height,
                (detection[0] + detection[2]) * ratio_width,
                (detection[1] + detection[3]) * ratio_height
            ]))
            face_landmark_5_list.append(detection[4:14].reshape((5, 2)) * [ ratio_width, ratio_height ])
            score_list.append(detection[14])
    return bounding_box_list, face_landmark_5_list, score_list


def prepare_detect_frame(temp_vision_frame : VisionFrame, face_detector_size : str) -> VisionFrame:
    face_detector_width, face_detector_height = unpack_resolution(face_detector_size)
    detect_vision_frame = numpy.zeros((face_detector_height, face_detector_width, 3))
    detect_vision_frame[:temp_vision_frame.shape[0], :temp_vision_frame.shape[1], :] = temp_vision_frame
    detect_vision_frame = (detect_vision_frame - 127.5) / 128.0
    detect_vision_frame = numpy.expand_dims(detect_vision_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
    return detect_vision_frame


def create_faces(vision_frame : VisionFrame, bounding_box_list : List[BoundingBox], face_landmark_5_list : List[FaceLandmark5], score_list : List[Score]) -> List[Face]:
    faces = []
    if face_detector_score > 0:
        sort_indices = numpy.argsort(-numpy.array(score_list))
        bounding_box_list = [ bounding_box_list[index] for index in sort_indices ]
        face_landmark_5_list = [face_landmark_5_list[index] for index in sort_indices]
        score_list = [ score_list[index] for index in sort_indices ]
        iou_threshold = 0.1 if face_detector_model == 'many' else 0.4
        keep_indices = apply_nms(bounding_box_list, iou_threshold)
        for index in keep_indices:
            bounding_box = bounding_box_list[index]
            face_landmark_5_68 = face_landmark_5_list[index]
            face_landmark_68_5 = expand_face_landmark_68_from_5(face_landmark_5_68)
            face_landmark_68 = face_landmark_68_5
            face_landmark_68_score = 0.0
            if face_landmarker_score > 0:
                face_landmark_68, face_landmark_68_score = detect_face_landmark_68(vision_frame, bounding_box)
                if face_landmark_68_score > face_landmarker_score:
                    face_landmark_5_68 = convert_face_landmark_68_to_5(face_landmark_68)
            landmarks : FaceLandmarkSet =\
            {
                '5': face_landmark_5_list[index],
                '5/68': face_landmark_5_68,
                '68': face_landmark_68,
                '68/5': face_landmark_68_5
            }
            scores : FaceScoreSet = \
            {
                'detector': score_list[index],
                'landmarker': face_landmark_68_score
            }
            embedding, normed_embedding = calc_embedding(vision_frame, landmarks.get('5/68'))
            gender, age = detect_gender_age(vision_frame, bounding_box)
            faces.append(Face(
                bounding_box = bounding_box,
                landmarks = landmarks,
                scores = scores,
                embedding = embedding,
                normed_embedding = normed_embedding,
                gender = gender,
                age = age
            ))
    return faces


def calc_embedding(temp_vision_frame : VisionFrame, face_landmark_5 : FaceLandmark5) -> Tuple[Embedding, Embedding]:
    face_recognizer = get_face_analyser().get('face_recognizer')
    crop_vision_frame, _ = warp_face_by_face_landmark_5(temp_vision_frame, face_landmark_5, 'arcface_112_v2', (112, 112))
    crop_vision_frame = crop_vision_frame / 127.5 - 1
    crop_vision_frame = crop_vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
    crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis = 0)
    embedding = face_recognizer.run(None,
    {
        face_recognizer.get_inputs()[0].name: crop_vision_frame
    })[0]
    embedding = embedding.ravel()
    normed_embedding = embedding / numpy.linalg.norm(embedding)
    return embedding, normed_embedding


def detect_face_landmark_68(temp_vision_frame : VisionFrame, bounding_box : BoundingBox) -> Tuple[FaceLandmark68, Score]:
    face_landmarker = get_face_analyser().get('face_landmarkers').get('68')
    scale = 195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max()
    translation = (256 - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
    crop_vision_frame, affine_matrix = warp_face_by_translation(temp_vision_frame, translation, scale, (256, 256))
    crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_RGB2Lab)
    if numpy.mean(crop_vision_frame[:, :, 0]) < 30:
        crop_vision_frame[:, :, 0] = cv2.createCLAHE(clipLimit = 2).apply(crop_vision_frame[:, :, 0])
    crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_Lab2RGB)
    crop_vision_frame = crop_vision_frame.transpose(2, 0, 1).astype(numpy.float32) / 255.0
    face_landmark_68, face_heatmap = face_landmarker.run(None,
    {
        face_landmarker.get_inputs()[0].name: [ crop_vision_frame ]
    })
    face_landmark_68 = face_landmark_68[:, :, :2][0] / 64
    face_landmark_68 = face_landmark_68.reshape(1, -1, 2) * 256
    face_landmark_68 = cv2.transform(face_landmark_68, cv2.invertAffineTransform(affine_matrix))
    face_landmark_68 = face_landmark_68.reshape(-1, 2)
    face_landmark_68_score = numpy.amax(face_heatmap, axis = (2, 3))
    face_landmark_68_score = numpy.mean(face_landmark_68_score)
    return face_landmark_68, face_landmark_68_score


def expand_face_landmark_68_from_5(face_landmark_5 : FaceLandmark5) -> FaceLandmark68:
    face_landmarker = get_face_analyser().get('face_landmarkers').get('68_5')
    affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, 'ffhq_512', (1, 1))
    face_landmark_5 = cv2.transform(face_landmark_5.reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
    face_landmark_68_5 = face_landmarker.run(None,
    {
        face_landmarker.get_inputs()[0].name: [ face_landmark_5 ]
    })[0][0]
    face_landmark_68_5 = cv2.transform(face_landmark_68_5.reshape(1, -1, 2), cv2.invertAffineTransform(affine_matrix)).reshape(-1, 2)
    return face_landmark_68_5


def detect_gender_age(temp_vision_frame : VisionFrame, bounding_box : BoundingBox) -> Tuple[int, int]:
    gender_age = get_face_analyser().get('gender_age')
    bounding_box = bounding_box.reshape(2, -1)
    scale = 64 / numpy.subtract(*bounding_box[::-1]).max()
    translation = 48 - bounding_box.sum(axis = 0) * scale * 0.5
    crop_vision_frame, affine_matrix = warp_face_by_translation(temp_vision_frame, translation, scale, (96, 96))
    crop_vision_frame = crop_vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
    crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis = 0)
    prediction = gender_age.run(None,
    {
        gender_age.get_inputs()[0].name: crop_vision_frame
    })[0][0]
    gender = int(numpy.argmax(prediction[:2]))
    age = int(numpy.round(prediction[2] * 100))
    return gender, age

def get_average_face(vision_frames : List[VisionFrame], position : int = 0) -> Optional[Face]:
    average_face = None
    faces = []
    embedding_list = []
    normed_embedding_list = []

    for vision_frame in vision_frames:
        face = get_one_face(vision_frame, position)
        if face:
            faces.append(face)
            embedding_list.append(face.embedding)
            normed_embedding_list.append(face.normed_embedding)
    if faces:
        first_face = faces[0]
        average_face = Face(
            bounding_box = first_face.bounding_box,
            landmarks = first_face.landmarks,
            scores = first_face.scores,
            embedding = numpy.mean(embedding_list, axis = 0),
            normed_embedding = numpy.mean(normed_embedding_list, axis = 0),
            gender = first_face.gender,
            age = first_face.age
        )
    return average_face

def get_one_face(vision_frame : VisionFrame, position : int = 0) -> Optional[Face]:
    many_faces = get_many_faces(vision_frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None

def get_many_faces(vision_frame : VisionFrame) -> List[Face]:
    faces = []
    # try:
    faces_cache = get_static_faces(vision_frame)
    if faces_cache:
        faces = faces_cache
    else:
        bounding_box_list = []
        face_landmark_5_list = []
        score_list = []

        if face_detector_model in [ 'many', 'retinaface']:
            bounding_box_list_retinaface, face_landmark_5_list_retinaface, score_list_retinaface = detect_with_retinaface(vision_frame, face_detector_size)
            bounding_box_list.extend(bounding_box_list_retinaface)
            face_landmark_5_list.extend(face_landmark_5_list_retinaface)
            score_list.extend(score_list_retinaface)
        if face_detector_model in [ 'many', 'scrfd' ]:
            bounding_box_list_scrfd, face_landmark_5_list_scrfd, score_list_scrfd = detect_with_scrfd(vision_frame, face_detector_size)
            bounding_box_list.extend(bounding_box_list_scrfd)
            face_landmark_5_list.extend(face_landmark_5_list_scrfd)
            score_list.extend(score_list_scrfd)
        if face_detector_model in [ 'many', 'yoloface' ]:
            bounding_box_list_yoloface, face_landmark_5_list_yoloface, score_list_yoloface = detect_with_yoloface(vision_frame, face_detector_size)
            bounding_box_list.extend(bounding_box_list_yoloface)
            face_landmark_5_list.extend(face_landmark_5_list_yoloface)
            score_list.extend(score_list_yoloface)
        if face_detector_model in [ 'yunet' ]:
            bounding_box_list_yunet, face_landmark_5_list_yunet, score_list_yunet = detect_with_yunet(vision_frame, face_detector_size)
            bounding_box_list.extend(bounding_box_list_yunet)
            face_landmark_5_list.extend(face_landmark_5_list_yunet)
            score_list.extend(score_list_yunet)

        if bounding_box_list and face_landmark_5_list and score_list:
            faces = create_faces(vision_frame, bounding_box_list, face_landmark_5_list, score_list)
        if faces:
            set_static_faces(vision_frame, faces)
    if face_analyser_order:
        faces = sort_by_order(faces, face_analyser_order)
    if face_analyser_age:
        faces = filter_by_age(faces, face_analyser_age)
    if face_analyser_gender:
        faces = filter_by_gender(faces, face_analyser_gender)
    # except (AttributeError, ValueError) as e:
    #     print("error", e)
    return faces

def sort_by_order(faces : List[Face], order : FaceAnalyserOrder) -> List[Face]:
    if order == 'left-right':
        return sorted(faces, key = lambda face: face.bounding_box[0])
    if order == 'right-left':
        return sorted(faces, key = lambda face: face.bounding_box[0], reverse = True)
    if order == 'top-bottom':
        return sorted(faces, key = lambda face: face.bounding_box[1])
    if order == 'bottom-top':
        return sorted(faces, key = lambda face: face.bounding_box[1], reverse = True)
    if order == 'small-large':
        return sorted(faces, key = lambda face: (face.bounding_box[2] - face.bounding_box[0]) * (face.bounding_box[3] - face.bounding_box[1]))
    if order == 'large-small':
        return sorted(faces, key = lambda face: (face.bounding_box[2] - face.bounding_box[0]) * (face.bounding_box[3] - face.bounding_box[1]), reverse = True)
    if order == 'best-worst':
        return sorted(faces, key = lambda face: face.scores.get('detector'), reverse = True)
    if order == 'worst-best':
        return sorted(faces, key = lambda face: face.scores.get('detector'))
    return faces


def filter_by_age(faces : List[Face], age : FaceAnalyserAge) -> List[Face]:
    filter_faces = []
    for face in faces:
        if categorize_age(face.age) == age:
            filter_faces.append(face)
    return filter_faces


def filter_by_gender(faces : List[Face], gender : FaceAnalyserGender) -> List[Face]:
    filter_faces = []
    for face in faces:
        if categorize_gender(face.gender) == gender:
            filter_faces.append(face)
    return filter_faces
