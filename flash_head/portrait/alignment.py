import cv2
import numpy as np

ANCHORS = [33, 133, 362, 263, 168, 6, 197]


def head_scale_matrix(source, reference, face_width, previous=None):
    """Limit generated head enlargement without aligning its pose or position."""
    source, reference = np.asarray(source, np.float64), np.asarray(reference, np.float64)
    pairs = [(127, 356), (33, 263)]  # temples and outer eye corners, not the moving jaw
    ratios = []
    for a, b in pairs:
        original = np.linalg.norm(reference[a] - reference[b])
        current = np.linalg.norm(source[a] - source[b])
        if not np.isfinite([original, current]).all():
            raise ValueError('头部比例关键点无效')
        if original > 1:
            ratios.append(current / original)
    ratio = np.median(ratios) if ratios else float('nan')
    if not np.isfinite(ratio) or not .5 <= ratio <= 1.25:
        raise ValueError('生成头部比例变化过大，请换用更清晰的照片')
    # Do not enlarge a face made narrower by a turn. Uniform scaling keeps
    # angles and expressions; the original neck root stays at the torso join.
    scale = min(1., 1 / ratio)
    if previous is not None:
        scale = .35 * scale + .65 * previous
    pivot = reference[152] + [0, face_width * .45]
    matrix = np.array([[scale, 0, pivot[0] * (1 - scale)],
                       [0, scale, pivot[1] * (1 - scale)]], np.float64)
    return matrix, scale


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
