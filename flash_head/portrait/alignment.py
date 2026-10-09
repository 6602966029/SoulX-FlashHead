import cv2
import numpy as np

ANCHORS = [33, 133, 362, 263, 168, 6, 197]


def estimate_alignment(source, target, face_width):
    source, target = np.asarray(source, np.float64), np.asarray(target, np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
        raise ValueError('关键点形状无效')
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('关键点包含无效数值')
    matrix, _ = cv2.estimateAffinePartial2D(source, target, method=cv2.LMEDS)
    if matrix is None or not np.isfinite(matrix).all():
        raise ValueError('无法对齐生成脸部')
    scale = np.hypot(matrix[0, 0], matrix[1, 0])
    angle = np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))
    error = np.linalg.norm(cv2.transform(source[None], matrix)[0] - target, axis=1).mean()
    if not .8 <= scale <= 1.25 or abs(angle) > 15 or error > face_width * .05:
        raise ValueError('生成头部移动过大，无法自然融回原图')
    return matrix


def smooth_alignment(matrix, previous):
    params = np.array([matrix[0, 2], matrix[1, 2],
                       np.arctan2(matrix[1, 0], matrix[0, 0]),
                       np.hypot(matrix[0, 0], matrix[1, 0])])
    if previous is not None:
        params[2] = previous[2] + np.arctan2(np.sin(params[2] - previous[2]), np.cos(params[2] - previous[2]))
        params = .7 * params + .3 * previous
    tx, ty, angle, scale = params
    c, s = scale * np.cos(angle), scale * np.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty]]), params
