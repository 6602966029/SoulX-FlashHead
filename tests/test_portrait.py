import importlib
import unittest
import numpy as np


def optional_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


class PreparationTests(unittest.TestCase):
    def test_fit_keeps_top_and_bottom_without_stretching(self):
        module = optional_module('flash_head.portrait.preparation')
        self.assertIsNotNone(module, 'Full-body preparation is missing')
        image = np.zeros((40, 20, 3), np.uint8)
        image[0:4] = [240, 0, 0]
        image[-4:] = [0, 0, 240]
        canvas, matrix = module.fit_canvas(image, (100, 100))
        self.assertEqual(canvas.shape, (100, 100, 3))
        self.assertEqual(matrix.shape, (2, 3))
        np.testing.assert_array_equal(canvas[0, 30], [240, 0, 0])
        np.testing.assert_array_equal(canvas[99, 30], [0, 0, 240])

    def test_border_crop_records_padding_in_inverse(self):
        module = optional_module('flash_head.portrait.preparation')
        self.assertIsNotNone(module, 'Square crop metadata is missing')
        crop, matrix = module.square_crop(np.zeros((40, 20, 3), np.uint8), [0, 0, 10, 10])
        self.assertEqual(crop.shape, (512, 512, 3))
        # Side 20, origin (-5,-6): a point at source (0,0) maps to (128,153.6).
        np.testing.assert_allclose(matrix @ [0, 0, 1], [128, 153.6], atol=1e-4)

    def test_zero_and_multiple_confident_faces_are_rejected(self):
        module = optional_module('flash_head.portrait.preparation')
        self.assertIsNotNone(module, 'Strict face validation is missing')
        for boxes, scores in [([], []), ([[0, 0, 1, 1]] * 2, [.9, .8])]:
            with self.assertRaises(ValueError):
                module.select_face(boxes, scores)
        np.testing.assert_array_equal(module.select_face([[0, 0, 1, 1]], [.9]), [0, 0, 1, 1])


class CompositionTests(unittest.TestCase):
    def test_resize_and_recorded_affine_agree_on_pixel_centers(self):
        import cv2
        module = optional_module('flash_head.portrait.preparation')
        image = np.repeat(np.arange(8, dtype=np.uint8)[None, :, None] * 20, 12, axis=0)
        image = np.repeat(image, 3, axis=2)
        canvas, matrix = module.fit_canvas(image, (80, 120))
        mapped = cv2.warpAffine(image, matrix, (80, 120), borderMode=cv2.BORDER_REPLICATE)
        self.assertLess(np.abs(canvas.astype(float) - mapped).mean(), 1)

    def test_mouth_mask_cannot_cross_narrow_face_contour(self):
        import mediapipe as mp
        module = optional_module('flash_head.portrait.compositor')
        self.assertTrue(hasattr(module, 'make_mouth_mask'), 'Safe facial contour clipping is missing')
        landmarks = np.full((478, 2), [256, 320], dtype=float)
        oval = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in edge})
        # Deliberately narrow rectangular contour: only 32 pixels each side of the mouth.
        for n, i in enumerate(oval):
            landmarks[i] = [[224, 200], [288, 200], [288, 410], [224, 410]][n % 4]
        landmarks[2] = [256, 240]
        landmarks[152] = [256, 410]
        mask = module.make_mouth_mask(landmarks, 256)
        self.assertTrue(mask.any())
        self.assertFalse(mask[:, :225].any())
        self.assertFalse(mask[:, 288:].any())

    def test_known_translation_is_removed(self):
        module = optional_module('flash_head.portrait.alignment')
        self.assertIsNotNone(module, 'Facial alignment is missing')
        target = np.array([[20, 20], [80, 20], [50, 50], [40, 40]], np.float64)
        matrix = module.estimate_alignment(target + [7, -4], target, 100)
        np.testing.assert_allclose(matrix, [[1, 0, -7], [0, 1, 4]], atol=1e-5)
        with self.assertRaises(ValueError):
            module.estimate_alignment(target * 2, target, 100)

    def test_blending_changes_only_nonzero_mask_pixels(self):
        module = optional_module('flash_head.portrait.compositor')
        self.assertIsNotNone(module, 'Local mouth compositing is missing')
        base = np.full((10, 10, 3), 20, np.uint8)
        patch = np.full((10, 10, 3), 100, np.uint8)
        mask = np.zeros((10, 10), np.float32)
        mask[4:6, 4:6] = 1
        result = module.blend_local(base, patch, mask)
        np.testing.assert_array_equal(result[mask == 0], base[mask == 0])
        np.testing.assert_array_equal(result[mask == 1], patch[mask == 1])
        np.testing.assert_array_equal(base, np.full((10, 10, 3), 20, np.uint8))

    def test_hold_is_bounded_and_first_failure_cannot_succeed(self):
        module = optional_module('flash_head.portrait.compositor')
        self.assertIsNotNone(module, 'Bounded failed-frame handling is missing')
        hold = module.FrameHold()
        with self.assertRaises(ValueError):
            hold.missing()
        hold.accept(np.array([123]))
        for _ in range(3):
            np.testing.assert_array_equal(hold.missing(), [123])
        with self.assertRaises(ValueError):
            hold.missing()
        hold.accept(np.array([124]))
        np.testing.assert_array_equal(hold.missing(), [124])


if __name__ == '__main__':
    unittest.main()
