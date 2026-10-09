"""Launch the single-GPU Gradio interface with the project-local Windows runtime."""
import os
from pathlib import Path
import runpy
import shutil
import sys

import imageio_ffmpeg

root = Path(__file__).resolve().parent
os.chdir(root)
sys.path.insert(0, str(root))
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding='utf-8')
bin_dir = root / 'tools' / 'bin'
bin_dir.mkdir(parents=True, exist_ok=True)
ffmpeg = bin_dir / 'ffmpeg.exe'
if not ffmpeg.exists():
    shutil.copy2(imageio_ffmpeg.get_ffmpeg_exe(), ffmpeg)
os.environ['PATH'] = str(bin_dir) + os.pathsep + os.environ.get('PATH', '')
os.environ['GRADIO_SERVER_NAME'] = '127.0.0.1'
os.environ['GRADIO_SERVER_PORT'] = '7861' if '--streaming' in sys.argv else '7860'
os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
os.environ['PYTHONUTF8'] = '1'
os.environ.setdefault('FLASHHEAD_MODEL_TYPE', 'lite' if (root / 'models/SoulX-FlashHead-1_3B/Model_Lite/diffusion_pytorch_model.safetensors').is_file() else 'pro')
for name in ('NO_PROXY', 'no_proxy'):
    current = os.environ.get(name, '')
    os.environ[name] = current + ',127.0.0.1,localhost'
runpy.run_path(str(root / ('gradio_app_streaming.py' if '--streaming' in sys.argv else 'gradio_app.py')), run_name='__main__')
