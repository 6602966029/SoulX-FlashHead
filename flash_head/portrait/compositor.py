import cv2
import numpy as np
from .alignment import ANCHORS, estimate_alignment, smooth_alignment
from .preparation import face_landmarks


def blend_local(base, patch, mask):
    alpha = mask[:, :, None]
    result = np.rint(base.astype(np.float32) * (1 - alpha) + patch.astype(np.float32) * alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


class FrameHold:
    def __init__(self):
        self.last = None
        self.failures = 0
        self.held_frames = 0

    def accept(self, patch):
        self.last = patch
        self.failures = 0
        return patch

    def missing(self):
        self.failures += 1
        if self.last is None or self.failures > 3:
            raise ValueError('脸部连续对齐失败，请使用清晰正面照片或更短的音频测试')
        self.held_frames += 1
        return self.last


def make_mouth_mask(landmarks, face_width):
    import mediapipe as mp
    lips = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_LIPS for i in edge})
    oval = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in edge})
    center = np.mean(landmarks[lips], axis=0)
    mask = np.zeros((512, 512), np.float32)
    cv2.ellipse(mask, tuple(np.rint(center).astype(int)),
                (round(face_width * .32), round(face_width * .20)), 0, 0, 360, 1, -1)
    feather = max(1, face_width * .04)
    mask = cv2.GaussianBlur(mask, (0, 0), feather)
    safe = np.zeros((512, 512), np.uint8)
    hull = cv2.convexHull(np.rint(landmarks[oval]).astype(np.int32))
    cv2.fillConvexPoly(safe, hull, 1)
    inset = max(1, round(face_width * .03))
    safe = cv2.erode(safe, np.ones((inset * 2 + 1, inset * 2 + 1), np.uint8))
    inward_feather = np.clip(cv2.distanceTransform(safe, cv2.DIST_L2, 3) / feather, 0, 1)
    mask *= inward_feather
    mask[mask < .01] = 0
    mask[:max(0, round(landmarks[2, 1] + 2))] = 0
    mask[min(512, round(landmarks[152, 1] - 3)):] = 0
    return mask


def make_head_mask(landmarks, face_width):
    """Keep generated eyes, expression and head pose; fade at crop edges and neck."""
    import mediapipe as mp
    y, x = np.mgrid[:512, :512].astype(np.float32)
    feather = max(8, face_width * .12)
    distance = np.minimum.reduce([x, 511 - x, y])
    edge_alpha = np.clip(distance / feather, 0, 1)
    edge_alpha = edge_alpha * edge_alpha * (3 - 2 * edge_alpha)
    oval = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in edge})
    face = np.zeros((512, 512), np.uint8)
    cv2.fillConvexPoly(face, cv2.convexHull(np.rint(landmarks[oval]).astype(np.int32)), 1)
    face_alpha = np.clip(1 - cv2.distanceTransform(1 - face, cv2.DIST_L2, 3) /
                         max(2, face_width * .02), 0, 1)
    # Hair and ears need a wide upper region; the lower region must follow
    # the jaw instead of cutting a rectangular corner into the static neck.
    upper_alpha = np.clip((landmarks[1, 1] + face_width * .15 - y) /
                          max(2, face_width * .15), 0, 1)
    upper_alpha = upper_alpha * upper_alpha * (3 - 2 * upper_alpha)
    alpha = np.maximum(face_alpha, upper_alpha) * edge_alpha
    alpha[alpha < .01] = 0
    return alpha


class MouthCompositor:
    def __init__(self, context):
        import mediapipe as mp
        self.context = context
        self.previous = None
        self.hold = FrameHold()
        landmarks = context.landmarks
        lips = sorted({index for edge in mp.solutions.face_mesh.FACEMESH_LIPS for index in edge})
        self.lip_indices = lips
        mask = make_mouth_mask(landmarks, context.face_width)
        self.head_mask = mask
        corners = cv2.transform(np.array([[[0, 0], [512, 0], [512, 512], [0, 512]]], np.float64),
                                context.head_to_canvas)[0]
        width, height = context.options.output_size
        left, top = np.maximum(np.floor(corners.min(axis=0)), 0).astype(int)
        right, bottom = np.minimum(np.ceil(corners.max(axis=0)), [width, height]).astype(int)
        if right <= left or bottom <= top:
            raise ValueError('脸部合成区域位于画布之外')
        self.bounds = (left, top, right, bottom)
        self.mapping = context.head_to_canvas.copy()
        self.mapping[:, 2] -= [left, top]
        self.roi_size = (right - left, bottom - top)
        self.mask = cv2.warpAffine(mask, self.mapping, self.roi_size)
        self.base = context.canvas[top:bottom, left:right]
        self.ring = (mask > .01) & (mask < .2)
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=True,
                                                  min_detection_confidence=.5, min_tracking_confidence=.5)

    def render(self, head_rgb):
        try:
            landmarks = face_landmarks(head_rgb, self.mesh)
            matrix = estimate_alignment(landmarks[ANCHORS], self.context.landmarks[ANCHORS],
                                        self.context.face_width)
            matrix, params = smooth_alignment(matrix, self.previous)
            aligned_lips = cv2.transform(landmarks[self.lip_indices][None], matrix)[0]
            lip_pixels = np.rint(aligned_lips).astype(int)
            if ((lip_pixels < 0).any() or (lip_pixels >= 512).any() or
                    np.any(self.head_mask[lip_pixels[:, 1], lip_pixels[:, 0]] < .05)):
                raise ValueError('嘴部运动超出固定合成区域')
            aligned = cv2.warpAffine(head_rgb, matrix, (512, 512), borderMode=cv2.BORDER_REPLICATE)
            if self.ring.any():
                difference = np.median(self.context.head_rgb[self.ring].astype(float) -
                                       aligned[self.ring].astype(float), axis=0)
                aligned = np.clip(aligned.astype(float) + np.clip(difference, -20, 20), 0, 255).astype(np.uint8)
            patch = cv2.warpAffine(aligned, self.mapping, self.roi_size, borderMode=cv2.BORDER_REPLICATE)
            self.previous = params
            patch = self.hold.accept(patch)
        except ValueError:
            patch = self.hold.missing()
        result = self.context.canvas.copy()
        left, top, right, bottom = self.bounds
        result[top:bottom, left:right] = blend_local(self.base, patch, self.mask)
        return result

    def close(self):
        self.mesh.close()


class HeadCompositor(MouthCompositor):
    """Paste head frames in reference coordinates without cancelling their motion."""

    def __init__(self, context):
        import mediapipe as mp
        super().__init__(context)
        self.oval_indices = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in edge})
        self.head_mask = make_head_mask(context.landmarks, context.face_width)
        self.mask = cv2.warpAffine(self.head_mask, self.mapping, self.roi_size)

    def render(self, head_rgb):
        try:
            if head_rgb.shape != (512, 512, 3) or head_rgb.dtype != np.uint8:
                raise ValueError('头部帧格式无效')
            landmarks = face_landmarks(head_rgb, self.mesh)
            points = landmarks[self.oval_indices]
            if not np.isfinite(points).all():
                raise ValueError('头部关键点无效')
            pixels = np.rint(points).astype(int)
            if (pixels < 4).any() or (pixels >= 508).any():
                raise ValueError('头部运动超出合成区域，请使用更宽的头部参考构图')
            # The moving face may occlude the original collar, but generated
            # clothing outside the face never extends the reference neck mask.
            face_mask = np.zeros((512, 512), np.uint8)
            cv2.fillConvexPoly(face_mask, cv2.convexHull(pixels.astype(np.int32)), 1)
            face_alpha = np.clip(cv2.distanceTransform(face_mask, cv2.DIST_L2, 3) / 2, 0, 1)
            mask = cv2.warpAffine(np.maximum(self.head_mask, face_alpha), self.mapping, self.roi_size)
            # Only the fixed crop-to-canvas transform is applied. Face alignment
            # would remove the generated turn, tilt and translation.
            patch = cv2.warpAffine(head_rgb, self.mapping, self.roi_size,
                                   borderMode=cv2.BORDER_REPLICATE)
            patch, mask = self.hold.accept((patch, mask))
        except ValueError as error:
            try:
                patch, mask = self.hold.missing()
            except ValueError:
                raise ValueError(f'头部帧连续不可用：{error}') from error
        self.mask = mask
        result = self.context.canvas.copy()
        left, top, right, bottom = self.bounds
        result[top:bottom, left:right] = blend_local(self.base, patch, mask)
        return result
