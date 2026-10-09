import importlib
import unittest
import numpy as np


def optional_module(name):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        return None


class PreparationTests(unittest.TestCase):
    def test_canvas_margin_keeps_full_photo_and_space_for_head_motion(self):
        module = optional_module('flash_head.portrait.preparation')
        image = np.full((40, 20, 3), 40, np.uint8)
        image[:4, 2:18] = [240, 0, 0]
        image[-4:, 2:18] = [0, 0, 240]
        try:
            canvas, matrix = module.fit_canvas(image, (100, 100), margin=10)
        except TypeError as error:
            self.fail(f'Head motion has no canvas margin: {error}')
        np.testing.assert_array_equal(canvas[0, 38], [40, 40, 40])
        np.testing.assert_array_equal(canvas[10, 38], [240, 0, 0])
        np.testing.assert_array_equal(canvas[89, 38], [0, 0, 240])
        np.testing.assert_array_equal(canvas[99, 38], [40, 40, 40])
        np.testing.assert_allclose(matrix @ [0, 0, 1], [30.5, 10.5])

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

    def test_border_crop_does_not_stretch_top_edge_hair_into_a_pillar(self):
        module = optional_module('flash_head.portrait.preparation')
        image = np.full((40, 20, 3), 210, np.uint8)
        image[:3, 4:7] = 10  # hair touching the photograph's top edge
        crop, _ = module.square_crop(image, [0, 0, 10, 10])
        # This location is above the source photograph, at x=5 in the original.
        np.testing.assert_array_equal(crop[40, 256], [210, 210, 210])

    def test_zero_and_multiple_confident_faces_are_rejected(self):
        module = optional_module('flash_head.portrait.preparation')
        self.assertIsNotNone(module, 'Strict face validation is missing')
        for boxes, scores in [([], []), ([[0, 0, 1, 1]] * 2, [.9, .8])]:
            with self.assertRaises(ValueError):
                module.select_face(boxes, scores)
        np.testing.assert_array_equal(module.select_face([[0, 0, 1, 1]], [.9]), [0, 0, 1, 1])


class CompositionTests(unittest.TestCase):
    def test_head_mask_keeps_eyes_and_hair_but_stops_above_body(self):
        module = optional_module('flash_head.portrait.compositor')
        self.assertTrue(hasattr(module, 'make_head_mask'), 'Whole-head motion mask is missing')
        landmarks = np.full((478, 2), [256, 240], dtype=float)
        landmarks[152] = [256, 350]
        mask = module.make_head_mask(landmarks, 200)
        self.assertEqual(mask[80, 200], 1)  # hair
        self.assertEqual(mask[200, 150], 1)  # eye
        self.assertEqual(mask[350, 256], 1)  # chin
        self.assertTrue(((mask > 0) & (mask < 1)).any())
        self.assertFalse(mask[394:].any())

    def test_full_head_keeps_generated_eye_motion_without_moving_body(self):
        from pathlib import Path
        from unittest.mock import patch
        from flash_head.portrait.types import PortraitContext, PortraitOptions
        module = optional_module('flash_head.portrait.compositor')
        self.assertTrue(hasattr(module, 'HeadCompositor'), 'Whole-head compositor is missing')
        landmarks = np.full((478, 2), [256, 240], dtype=float)
        landmarks[33] = [150, 200]
        landmarks[263] = [350, 200]
        landmarks[152] = [256, 350]
        head = np.full((512, 512, 3), 40, np.uint8)
        head[195:205, 145:155] = [200, 20, 10]
        canvas = np.full((700, 512, 3), 40, np.uint8)
        canvas[:512] = head
        context = PortraitContext(canvas, head, Path('unused.png'),
                                  np.array([[1, 0, 0], [0, 1, 0]], dtype=float),
                                  landmarks, 200, PortraitOptions((512, 700)))
        generated = np.full_like(head, 40)
        generated[201:211, 165:175] = [200, 20, 10]
        composer = module.HeadCompositor(context)
        try:
            # Only the external detector is substituted; the actual compositor processes the image.
            with patch.object(module, 'face_landmarks', return_value=landmarks + [20, 6]):
                result = composer.render(generated)
            np.testing.assert_array_equal(result[206, 170], generated[206, 170])
            np.testing.assert_array_equal(result[200, 150], [40, 40, 40])
            np.testing.assert_array_equal(result[400:], canvas[400:])
        finally:
            composer.close()

    def test_moving_chin_is_preserved_while_generated_collar_is_excluded(self):
        import mediapipe as mp
        from pathlib import Path
        from unittest.mock import patch
        from flash_head.portrait.types import PortraitContext, PortraitOptions
        module = optional_module('flash_head.portrait.compositor')
        oval = sorted({i for edge in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in edge})
        landmarks = np.full((478, 2), [256, 240], dtype=float)
        for i, angle in zip(oval, np.linspace(0, 2 * np.pi, len(oval), endpoint=False)):
            landmarks[i] = [256 + 100 * np.cos(angle), 240 + 110 * np.sin(angle)]
        landmarks[152] = [256, 350]
        landmarks[33], landmarks[263] = [200, 200], [310, 200]
        self.assertEqual(module.make_head_mask(landmarks, 200)[348, 150], 0,
                         'A rectangular jaw mask leaves a visible corner beside the neck')
        canvas = np.full((700, 512, 3), 40, np.uint8)
        head = canvas[:512].copy()
        context = PortraitContext(canvas, head, Path('unused.png'),
                                  np.array([[1, 0, 0], [0, 1, 0]], dtype=float),
                                  landmarks, 200, PortraitOptions((512, 700)))
        generated = head.copy()
        generated[350:] = [10, 20, 200]  # changing collar outside the generated face
        generated[370:381, 250:263] = [200, 20, 10]  # lower moving chin
        composer = module.HeadCompositor(context)
        try:
            with patch.object(module, 'face_landmarks', return_value=landmarks + [0, 40]):
                try:
                    result = composer.render(generated)
                except ValueError as error:
                    self.fail(f'Natural downward head motion was incorrectly rejected: {error}')
            np.testing.assert_array_equal(result[376, 256], [200, 20, 10])
            np.testing.assert_array_equal(result[365, 140], canvas[365, 140])
            np.testing.assert_array_equal(result[400:], canvas[400:])
        finally:
            composer.close()

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
