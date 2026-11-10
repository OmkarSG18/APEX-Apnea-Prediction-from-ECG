import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor

TARGET_DIR = os.path.join('data', 'apnea-ecg')
BASE_URL = 'https://physionet.org/files/apnea-ecg/1.0.0/'

os.makedirs(TARGET_DIR, exist_ok=True)

train_records = [f'a{i:02d}' for i in range(1, 21)] + \
                [f'b{i:02d}' for i in range(1, 6)] + \
                [f'c{i:02d}' for i in range(1, 11)]

test_records = [f'x{i:02d}' for i in range(1, 36)]
all_records = train_records + test_records
extensions = ['.dat', '.hea', '.apn']

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
session = requests.Session()
session.headers.update(headers)

def download_file(rec, ext):
    fn = f'{rec}{ext}'
    local_path = os.path.join(TARGET_DIR, fn)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return f'EXISTS: {fn}'
    
    url = f'{BASE_URL}{fn}'
    for attempt in range(3):
        try:
            r = session.get(url, timeout=20)
            if r.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(r.content)
                return f'DOWNLOADED: {fn} ({len(r.content)} bytes)'
            else:
                return f'STATUS {r.status_code}: {fn}'
        except Exception as e:
            time.sleep(1)
    return f'FAILED: {fn}'

def main():
    print(f'Sequential/Low-concurrency download of {len(all_records)} records...')
    tasks = []
    for rec in all_records:
        for ext in extensions:
            tasks.append((rec, ext))
            
    with ThreadPoolExecutor(max_workers=2) as executor:
        for rec, ext in tasks:
            res = download_file(rec, ext)
            if 'DOWNLOADED' in res or 'FAILED' in res:
                print(res)

    print('Download complete!')

if __name__ == '__main__':
    main()
