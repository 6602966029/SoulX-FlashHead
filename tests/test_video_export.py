import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import numpy as np
import soundfile as sf


class ExportTests(unittest.TestCase):
    def scratch(self):
        root = Path(__file__).resolve().parents[1] / 'runtime/test-export'
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=root)

    def module(self):
        try:
            module = importlib.import_module('flash_head.services.video_export')
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, 'Incremental synchronized export is missing')
        return module

    def test_export_trims_padded_frames_to_original_audio(self):
        module = self.module()
        with self.scratch() as directory:
            root = Path(directory)
            audio = root / 'input.wav'
            sf.write(audio, np.zeros(16000), 16000)
            consumed = []
            def frames():
                for i in range(100):
                    consumed.append(i)
                    yield np.full((64, 32, 3), i, np.uint8)
            output = root / 'result.mp4'
            module.export_video(frames(), audio, output, 25)
            self.assertEqual(len(consumed), 25)
            probe = shutil.which('ffprobe') or str(Path('tools/bin/ffprobe.exe').resolve())
            info = json.loads(subprocess.check_output([probe, '-v', 'error', '-show_streams', '-of', 'json', str(output)]))
            streams = {s['codec_type']: s for s in info['streams']}
            self.assertEqual(streams['video']['width'], 32)
            self.assertEqual(streams['video']['nb_frames'], '25')
            self.assertLessEqual(abs(float(streams['audio']['duration']) - float(streams['video']['duration'])), .04)

    def test_short_generator_does_not_publish_video(self):
        module = self.module()
        with self.scratch() as directory:
            root = Path(directory)
            audio = root / 'input.wav'
            sf.write(audio, np.zeros(16000), 16000)
            output = root / 'result.mp4'
            with self.assertRaises(ValueError):
                module.export_video(iter([np.zeros((64, 32, 3), np.uint8)]), audio, output, 25)
            self.assertFalse(output.exists())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['input.wav'])


if __name__ == '__main__':
    unittest.main()
