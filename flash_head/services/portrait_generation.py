from collections import deque
from datetime import datetime
from pathlib import Path
import tempfile
import time
import numpy as np
from loguru import logger
from flash_head.portrait.preparation import prepare_portrait
from flash_head.portrait.compositor import MouthCompositor
from .video_export import export_video


def audio_slices(audio, slice_length):
    if len(audio) == 0 or slice_length <= 0:
        raise ValueError('音频不能为空')
    padded = np.pad(audio, (0, (-len(audio)) % slice_length))
    yield from padded.reshape(-1, slice_length)


def generate_portrait(pipeline, image_path, audio_path, seed, progress=lambda *args, **kwargs: None):
    import librosa
    import torch
    from flash_head.inference import get_base_data, get_infer_params, get_audio_embedding, run_pipeline
    started = time.monotonic()
    scratch = Path('runtime/portrait-jobs')
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='job-', dir=scratch) as directory:
        progress(.05, desc='检查全身照片与脸部关键点…')
        context = prepare_portrait(image_path, directory)
        get_base_data(pipeline, str(context.head_path), int(seed), False)
        params = get_infer_params()
        rate, fps = params['sample_rate'], params['tgt_fps']
        if fps != context.options.fps:
            raise ValueError('全身模式需要 25 fps 推理配置')
        audio, _ = librosa.load(audio_path, sr=rate, mono=True)
        motion = params['motion_frames_num']
        slice_len = params['frame_num'] - motion
        slice_samples = slice_len * rate // fps
        total = max(1, int(np.ceil(len(audio) / slice_samples)))
        cached = rate * params['cached_audio_duration']
        audio_end = params['cached_audio_duration'] * fps
        dq = deque([0.0] * cached, maxlen=cached)
        composer = MouthCompositor(context)
        output = Path('gradio_results') / ('res_' + datetime.now().strftime('%Y%m%d-%H%M%S-%f') + '.mp4')
        rendered = 0
        def frames():
            nonlocal rendered
            for index, samples in enumerate(audio_slices(audio, slice_samples)):
                dq.extend(samples.tolist())
                embedding = get_audio_embedding(pipeline, np.asarray(dq), audio_end - params['frame_num'], audio_end)
                torch.cuda.synchronize()
                generated = run_pipeline(pipeline, embedding)[motion:]
                head_frames = generated.detach().cpu().numpy().astype(np.uint8)
                for frame in head_frames:
                    yield composer.render(frame)
                    rendered += 1
                progress(.1 + .8 * (index + 1) / total, desc=f'全身合成 {index + 1}/{total}')
        try:
            result = export_video(frames(), audio_path, output, fps)
            logger.info('Portrait complete: frames={}, held={}, elapsed={:.2f}s, output={}',
                        rendered, composer.hold.held_frames, time.monotonic() - started, result)
            progress(1, desc='全身视频已完成')
            return result
        finally:
            composer.close()
