import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

TARGET_DIR = os.path.join('data', 'apnea-ecg')
BASE_URL = 'https://physionet.org/files/apnea-ecg/1.0.0/'

os.makedirs(TARGET_DIR, exist_ok=True)

train_records = [f'a{i:02d}' for i in range(1, 21)] + \
                [f'b{i:02d}' for i in range(1, 6)] + \
                [f'c{i:02d}' for i in range(1, 11)]

test_records = [f'x{i:02d}' for i in range(1, 36)]
all_records = train_records + test_records

extensions = ['.dat', '.hea', '.apn', '.qrs']

def download_file(rec, ext):
    fn = f'{rec}{ext}'
    local_path = os.path.join(TARGET_DIR, fn)
    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
        return f'EXISTS: {fn}'
    
    url = f'{BASE_URL}{fn}'
    try:
        urllib.request.urlretrieve(url, local_path)
        return f'DOWNLOADED: {fn} ({os.path.getsize(local_path)} bytes)'
    except Exception as e:
        if ext == '.qrs':
            return f'OPTIONAL MISSING: {fn}'
        return f'ERROR {fn}: {e}'

def main():
    print(f'Starting download of {len(all_records)} records into {TARGET_DIR}...')
    tasks = []
    for rec in all_records:
        for ext in extensions:
            tasks.append((rec, ext))
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_file, rec, ext) for rec, ext in tasks]
        for f in futures:
            res = f.result()
            if 'ERROR' in res or 'DOWNLOADED' in res:
                print(res)

    print('Download completed successfully!')

if __name__ == '__main__':
    main()
