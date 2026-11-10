import os
import requests

TARGET_DIR = os.path.join('data', 'apnea-ecg')
BASE_URL = 'https://physionet.org/files/apnea-ecg/1.0.0/'

os.makedirs(TARGET_DIR, exist_ok=True)

test_records = [f'x{i:02d}' for i in range(1, 36)]
extensions = ['.dat', '.hea', '.apn']

headers = {'User-Agent': 'Mozilla/5.0'}
session = requests.Session()
session.headers.update(headers)

def main():
    print('Downloading test split records (x01-x35)...')
    for rec in test_records:
        for ext in extensions:
            fn = f'{rec}{ext}'
            local_p = os.path.join(TARGET_DIR, fn)
            if os.path.exists(local_p) and os.path.getsize(local_p) > 0:
                continue
            url = f'{BASE_URL}{fn}'
            try:
                r = session.get(url, timeout=15)
                if r.status_code == 200:
                    with open(local_p, 'wb') as f:
                        f.write(r.content)
                    print(f'Fetched {fn}')
            except Exception as e:
                print(f'Error {fn}: {e}')

if __name__ == '__main__':
    main()
