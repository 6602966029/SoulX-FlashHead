from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps
from .types import PortraitContext, PortraitOptions


def fit_canvas(image, output_size, margin=0):
    width, height = output_size
    source_h, source_w = image.shape[:2]
    if min(width, height, source_h, source_w) <= 0:
        raise ValueError('图片和输出尺寸必须大于零')
    if margin < 0 or 2 * margin >= min(width, height):
        raise ValueError('画布留白超出输出尺寸')
    scale = min((width - 2 * margin) / source_w, (height - 2 * margin) / source_h)
    target_w, target_h = max(1, round(source_w * scale)), max(1, round(source_h * scale))
    left, top = (width - target_w) // 2, (height - target_h) // 2
    corners = image[[0, 0, -1, -1], [0, -1, 0, -1]]
    canvas = np.empty((height, width, 3), np.uint8)
    canvas[:] = np.median(corners, axis=0).astype(np.uint8)
    canvas[top:top + target_h, left:left + target_w] = cv2.resize(image, (target_w, target_h))
    # Record the actual rounded size used by OpenCV, not an idealized scale.
    sx, sy = target_w / source_w, target_h / source_h
    matrix = np.array([[sx, 0, left + (sx - 1) / 2],
                       [0, sy, top + (sy - 1) / 2]], np.float64)
    return canvas, matrix


def square_crop(image, bbox, size=512):
    x1, y1, x2, y2 = map(float, bbox)
    side = 2 * (x2 - x1)
    if side <= 0 or y2 <= y1:
        raise ValueError('人脸框无效')
    left = (x1 + x2) / 2 - side / 2
    top = (y1 + y2) / 2 - side * .55
    scale = size / side
    matrix = np.array([[scale, 0, -left * scale], [0, scale, -top * scale]], np.float64)
    corners = image[[0, 0, -1, -1], [0, -1, 0, -1]]
    background = tuple(np.median(corners, axis=0).astype(float))
    crop = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=background)
    return crop, matrix


def select_face(boxes, scores):
    valid = [box for box, score in zip(boxes, scores) if score >= .5]
    if not valid:
        raise ValueError('没有检测到清晰人脸，请上传单人正面照片')
    if len(valid) != 1:
        raise ValueError('检测到多张人脸，请使用单人正面照片，不要上传三视图拼图')
    return np.asarray(valid[0], np.float64)


def face_landmarks(image, mesh):
    result = mesh.process(np.ascontiguousarray(image))
    if not result.multi_face_landmarks:
        raise ValueError('无法定位脸部关键点，请换用清晰、无遮挡的正面照片')
    height, width = image.shape[:2]
    return np.array([(point.x * width, point.y * height)
                     for point in result.multi_face_landmarks[0].landmark], np.float64)


def prepare_portrait(image_path, work_dir, options=None):
    import mediapipe as mp
    from flash_head.utils.cpu_face_handler import CPUFaceHandler
    options = options or PortraitOptions()
    with Image.open(image_path) as source:
        image = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
    detector = CPUFaceHandler(min_detection_confidence=.5)
    try:
        boxes, scores = detector(image)
    finally:
        detector.detector.close()
    box = select_face(boxes, scores) * np.array([image.shape[1], image.shape[0]] * 2)
    head, image_to_head = square_crop(image, box)
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True,
                                       min_detection_confidence=.5) as mesh:
        landmarks = face_landmarks(head, mesh)
    canvas, image_to_canvas = fit_canvas(image, options.output_size, options.canvas_margin)
    head_to_image = cv2.invertAffineTransform(image_to_head)
    head_to_canvas = (np.vstack([image_to_canvas, [0, 0, 1]]) @
                      np.vstack([head_to_image, [0, 0, 1]]))[:2]
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    head_path = work_dir / 'head-reference.png'
    Image.fromarray(head).save(head_path)
    return PortraitContext(canvas, head, head_path, head_to_canvas, landmarks,
                           (box[2] - box[0]) * image_to_head[0, 0], options)
