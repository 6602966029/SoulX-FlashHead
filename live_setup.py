"""Local setup and OBS preview. Automatic narration/comment integration is separate."""
import base64
import json
import os
from pathlib import Path
import re

from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import requests
import uvicorn

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
MEDIA = ROOT / 'gradio_results'
RUNTIME = ROOT / 'runtime'
RUNTIME.mkdir(exist_ok=True)
app = FastAPI(docs_url=None, redoc_url=None)
ENV_KEYS = ['LIVE_LLM_BASE_URL', 'LIVE_LLM_MODEL', 'LIVE_LLM_API_KEY', 'TENCENT_SECRET_ID', 'TENCENT_SECRET_KEY', 'TENCENT_TTS_REGION', 'TENCENT_TTS_VOICE']


def settings():
    defaults = dotenv_values(ROOT / '.env.example')
    defaults.update(dotenv_values(ROOT / '.env'))
    return {key: str(defaults.get(key) or '') for key in ENV_KEYS}


def clip_path(name):
    if not re.fullmatch(r'res_[\w-]+\.mp4', name):
        raise HTTPException(404)
    path = MEDIA / name
    if not path.is_file():
        raise HTTPException(404)
    return path


def check_local(request):
    if request.headers.get('host') not in ('127.0.0.1:7862', 'localhost:7862'):
        raise HTTPException(403, 'Only localhost is allowed')
    origin = request.headers.get('origin')
    if origin and origin not in ('http://127.0.0.1:7862', 'http://localhost:7862'):
        raise HTTPException(403, 'Cross-origin changes are blocked')


class SetupValues(BaseModel):
    llm_base_url: str
    llm_model: str
    llm_api_key: str = ''
    tencent_secret_id: str = ''
    tencent_secret_key: str = ''
    tts_region: str = 'ap-guangzhou'
    tts_voice: str = '101001'


@app.get('/')
def setup_page():
    return FileResponse(ROOT / 'live_ui/setup.html', media_type='text/html')


@app.get('/player')
def player():
    return FileResponse(ROOT / 'live_ui/player.html', media_type='text/html')


@app.get('/api/status')
def status():
    values = settings()
    files = ['Model_Lite/config.json', 'Model_Lite/diffusion_pytorch_model.safetensors', 'VAE_LTX/config.json', 'VAE_LTX/diffusion_pytorch_model.safetensors']
    return {'lite_ready': all((ROOT / 'models/SoulX-FlashHead-1_3B' / name).is_file() for name in files),
            'llm_configured': bool(values['LIVE_LLM_API_KEY']),
            'tts_configured': bool(values['TENCENT_SECRET_ID'] and values['TENCENT_SECRET_KEY']),
            'llm_base_url': values['LIVE_LLM_BASE_URL'], 'llm_model': values['LIVE_LLM_MODEL'],
            'tts_region': values['TENCENT_TTS_REGION'], 'tts_voice': values['TENCENT_TTS_VOICE'],
            'comments': '未接入', 'automatic_narration': '待开发', 'ai_label': 'AI数字人直播'}


@app.post('/api/settings')
def save_settings(data: SetupValues, request: Request):
    check_local(request)
    if not data.llm_base_url.startswith('https://'):
        raise HTTPException(400, '文字服务地址必须使用 HTTPS')
    values = settings()
    values.update(LIVE_LLM_BASE_URL=data.llm_base_url.rstrip('/'), LIVE_LLM_MODEL=data.llm_model,
                  TENCENT_TTS_REGION=data.tts_region, TENCENT_TTS_VOICE=data.tts_voice)
    for key, value in [('LIVE_LLM_API_KEY', data.llm_api_key), ('TENCENT_SECRET_ID', data.tencent_secret_id), ('TENCENT_SECRET_KEY', data.tencent_secret_key)]:
        if value.strip():
            values[key] = value.strip()
    temporary = ROOT / '.env.tmp'
    temporary.write_text(''.join(f'{key}={json.dumps(value, ensure_ascii=False)}\n' for key, value in values.items()), encoding='utf-8')
    temporary.replace(ROOT / '.env')
    return {'message': '已保存到本机。密钥不会返回页面。'}


@app.post('/api/test/{provider}')
def test_provider(provider: str, request: Request):
    check_local(request)
    values = settings()
    try:
        if provider == 'llm':
            if not values['LIVE_LLM_API_KEY']:
                raise HTTPException(400, '尚未配置文字模型 API 密钥')
            response = requests.post(values['LIVE_LLM_BASE_URL'] + '/chat/completions', timeout=45,
                headers={'Authorization': 'Bearer ' + values['LIVE_LLM_API_KEY']},
                json={'model': values['LIVE_LLM_MODEL'], 'messages': [{'role': 'user', 'content': '请只回复：连接成功'}], 'max_tokens': 32})
            if not response.ok:
                return {'ok': False, 'message': f'文字服务返回 HTTP {response.status_code}，请检查密钥、模型和账户余额。'}
            response.json()['choices'][0]['message']['content']
            return {'ok': True, 'message': '文字模型连接成功'}
        if provider == 'tts':
            if not values['TENCENT_SECRET_ID'] or not values['TENCENT_SECRET_KEY']:
                raise HTTPException(400, '尚未配置腾讯云 SecretId / SecretKey')
            from tencentcloud.common import credential
            from tencentcloud.tts.v20190823 import tts_client, models
            client = tts_client.TtsClient(credential.Credential(values['TENCENT_SECRET_ID'], values['TENCENT_SECRET_KEY']), values['TENCENT_TTS_REGION'])
            params = models.TextToVoiceRequest()
            params.from_json_string(json.dumps({'Text': '语音连接测试成功。', 'SessionId': 'soulx-setup', 'VoiceType': int(values['TENCENT_TTS_VOICE']), 'SampleRate': 16000, 'Codec': 'wav'}))
            result = client.TextToVoice(params)
            (RUNTIME / 'tts-test.wav').write_bytes(base64.b64decode(result.Audio))
            return {'ok': True, 'message': '语音合成成功，可以试听', 'audio': '/api/test-audio'}
        raise HTTPException(404)
    except HTTPException:
        raise
    except Exception:
        return {'ok': False, 'message': '连接测试失败，请检查服务是否开通、密钥权限和网络。密钥与服务原始响应不会写入日志。'}


@app.get('/api/test-audio')
def test_audio():
    if not (RUNTIME / 'tts-test.wav').exists():
        raise HTTPException(404)
    return FileResponse(RUNTIME / 'tts-test.wav', media_type='audio/wav')


@app.get('/api/clips')
def clips():
    paths = sorted(MEDIA.glob('res_*.mp4'), key=lambda path: path.stat().st_mtime, reverse=True)
    return [{'name': path.name, 'bytes': path.stat().st_size} for path in paths if path.stat().st_size > 1024]


@app.get('/api/playback')
def playback():
    path = RUNTIME / 'playback.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    available = clips()
    return {'clip': available[0]['name'] if available else None}


@app.post('/api/playback')
async def set_playback(request: Request):
    check_local(request)
    data = await request.json()
    clip_path(data.get('clip', ''))
    (RUNTIME / 'playback.json').write_text(json.dumps({'clip': data['clip']}), encoding='utf-8')
    return {'ok': True}


@app.get('/media/{name}')
def media(name: str):
    return FileResponse(clip_path(name), media_type='video/mp4')


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=7862)
