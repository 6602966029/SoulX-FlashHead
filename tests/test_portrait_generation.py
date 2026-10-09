import importlib
import unittest
import numpy as np


class AudioBoundaryTests(unittest.TestCase):
    def test_short_audio_is_padded_once_without_repeating_speech(self):
        try:
            module = importlib.import_module('flash_head.services.portrait_generation')
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, 'Portrait generation boundary is missing')
        chunks = list(module.audio_slices(np.array([1, 2, 3], np.float32), 5))
        self.assertEqual(len(chunks), 1)
        np.testing.assert_array_equal(chunks[0], [1, 2, 3, 0, 0])
        with self.assertRaises(ValueError):
            list(module.audio_slices(np.array([]), 5))


if __name__ == '__main__':
    unittest.main()
