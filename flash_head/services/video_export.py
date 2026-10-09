import itertools
from contextlib import suppress
import math
from pathlib import Path
import shutil
import subprocess
import numpy as np
import soundfile as sf


def export_video(frames, audio_path, output_path, fps):
    import imageio_ffmpeg
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    info = sf.info(str(audio_path))
    if info.frames <= 0 or fps <= 0:
        raise ValueError('音频为空或视频帧率无效')
    count = math.ceil(info.frames * fps / info.samplerate)
    frames = iter(frames)
    process = None
    temporary = output.with_name('.' + output.stem + '.partial.mp4')
    log_path = temporary.with_suffix('.log')
    try:
        first = next(frames, None)
        if first is None:
            raise ValueError('没有生成视频帧')
        height, width = first.shape[:2]
        if width % 2 or height % 2:
            raise ValueError('视频尺寸必须为偶数')
        ffmpeg = shutil.which('ffmpeg') or imageio_ffmpeg.get_ffmpeg_exe()
        command = [ffmpeg, '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                   '-s', f'{width}x{height}', '-r', str(fps), '-i', 'pipe:0',
                   '-i', str(audio_path), '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                   '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-shortest', '-movflags', '+faststart', str(temporary)]
        with log_path.open('wb') as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            written = 0
            for frame in itertools.chain([first], itertools.islice(frames, count - 1)):
                if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                    raise ValueError('视频帧必须具有相同尺寸和 RGB uint8 格式')
                process.stdin.write(np.ascontiguousarray(frame).tobytes())
                written += 1
            if written != count:
                raise ValueError(f'生成帧不足：需要 {count} 帧，得到 {written} 帧')
            process.stdin.close()
            code = process.wait(timeout=120)
        if code:
            raise RuntimeError('FFmpeg 编码失败：' + log_path.read_text(encoding='utf-8', errors='replace')[-1200:])
        temporary.replace(output)
        return str(output)
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if process is not None and process.stdin is not None and not process.stdin.closed:
            with suppress(OSError):
                process.stdin.close()
        close = getattr(frames, 'close', None)
        if close:
            close()
        temporary.unlink(missing_ok=True)
        log_path.unlink(missing_ok=True)
