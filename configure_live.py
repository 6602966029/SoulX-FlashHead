"""Prepare only the isolated SoulXLive OBS files, while OBS is closed."""
import configparser
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import winreg

ROOT = Path(__file__).resolve().parent
FRIENDLY_NAME = '{a45c254e-df1c-4efd-8020-67d146a850e0},14'


def cable_endpoints(direction, prefix):
    path = rf'SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\{direction}'
    found = []
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as parent:
        for index in range(winreg.QueryInfoKey(parent)[0]):
            guid = winreg.EnumKey(parent, index)
            try:
                with winreg.OpenKey(parent, guid) as device:
                    if winreg.QueryValueEx(device, 'DeviceState')[0] != 1:
                        continue
                    with winreg.OpenKey(device, 'Properties') as props:
                        name = winreg.QueryValueEx(props, FRIENDLY_NAME)[0]
                    if name.startswith(prefix + ' ('):
                        flow = '0' if direction == 'Render' else '1'
                        found.append((name, '{0.0.' + flow + '.00000000}.' + guid))
            except FileNotFoundError:
                continue
    return found


def main():
    processes = subprocess.check_output(
        ['tasklist', '/FI', 'IMAGENAME eq obs64.exe', '/FO', 'CSV', '/NH'])
    if b'obs64.exe' in processes.lower():
        print('OBS is running. Save and close OBS before applying SoulXLive settings.')
        return 2
    base = Path(os.environ['APPDATA']) / 'obs-studio/basic'
    scene_path = base / 'scenes/SoulXLive.json'
    profile_path = base / 'profiles/SoulXLive/basic.ini'
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    template = ROOT / 'config/obs-scene.json'
    collection = json.loads((scene_path if scene_path.exists() else template).read_text(encoding='utf-8-sig'))
    profile = configparser.ConfigParser(interpolation=None)
    profile.optionxform = str
    if profile_path.exists():
        profile.read(profile_path, encoding='utf-8-sig')
    for section in ['General', 'Video', 'Audio', 'Output', 'SimpleOutput']:
        if not profile.has_section(section):
            profile.add_section(section)
    values = {
        'General': {'Name': 'SoulXLive'},
        'Video': {'BaseCX': '1080', 'BaseCY': '1920', 'OutputCX': '1080', 'OutputCY': '1920', 'FPSType': '1', 'FPSInt': '25'},
        'Audio': {'SampleRate': '48000', 'ChannelSetup': 'Stereo'},
        'Output': {'Mode': 'Simple'},
        'SimpleOutput': {'VBitrate': '4500', 'ABitrate': '160', 'StreamEncoder': 'nvenc', 'RecQuality': 'Small', 'RecEncoder': 'nvenc'},
    }
    for section, options in values.items():
        for key, value in options.items():
            profile.set(section, key, value)
    render = cable_endpoints('Render', 'CABLE Input')
    capture = cable_endpoints('Capture', 'CABLE Output')
    browser = next((source for source in collection['sources']
                    if source['name'] == 'SoulX AI直播画面' and source['id'] == 'browser_source'), None)
    if browser is None:
        raise RuntimeError('SoulXLive browser source is missing. No OBS files were changed.')
    if len(render) == 1 and len(capture) == 1:
        profile.set('Audio', 'MonitoringDeviceName', render[0][0])
        profile.set('Audio', 'MonitoringDeviceId', render[0][1])
        browser['monitoring_type'] = 2
        audio_status = 'VB-CABLE detected: OBS monitor -> CABLE Input; companion microphone -> CABLE Output.'
    else:
        browser['monitoring_type'] = 0
        audio_status = 'VB-CABLE is unavailable or ambiguous. Audio monitoring stays off; reboot and check the driver.'
    backup = ROOT / 'runtime/obs-backups' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    backup.mkdir(parents=True)
    for path in (scene_path, profile_path):
        if path.exists():
            shutil.copy2(path, backup / path.name)
    scene_tmp = scene_path.with_suffix('.json.tmp')
    scene_tmp.write_text(json.dumps(collection, ensure_ascii=False, indent=2), encoding='utf-8')
    profile_tmp = profile_path.with_suffix('.ini.tmp')
    with profile_tmp.open('w', encoding='utf-8') as handle:
        profile.write(handle, space_around_delimiters=False)
    scene_tmp.replace(scene_path)
    profile_tmp.replace(profile_path)
    print('SoulXLive configured: 1080x1920, 25 fps. Existing files backed up under runtime/obs-backups.')
    print(audio_status)
    return 0


if __name__ == '__main__':
    sys.exit(main())
