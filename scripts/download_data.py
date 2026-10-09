"""Recover exact Kaggle source (or verified UCI equivalent); offline gzip first."""
from pathlib import Path
import argparse, gzip, hashlib, io, urllib.request, zipfile

ROOT=Path(__file__).resolve().parents[1]
# Exact SHA-256 of the source CSV distributed by the selected Kaggle version.
EXPECTED='233e260d5d1d506a2c10381da5b8c2f75f2c08d1b373ca7b825e39ebe1bd30df'

def extract_csv(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        for name in z.namelist():
            if name.endswith('bank-additional-full.csv'): return z.read(name)
        for name in z.namelist():
            if name.endswith('bank-additional.zip'): return extract_csv(z.read(name))
    raise ValueError('Expected full additional CSV not found in archive.')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--online',action='store_true',help='Force re-download instead of bundled gzip'); args=parser.parse_args()
    dest=ROOT/'data/raw/bank-additional-full.csv'; compressed=dest.with_suffix('.csv.gz'); dest.parent.mkdir(parents=True,exist_ok=True)
    if compressed.exists() and not args.online:
        data=gzip.decompress(compressed.read_bytes()); origin='bundled gzip: offline'
    else:
        sources=[('Kaggle version 1','https://www.kaggle.com/api/v1/datasets/download/sahistapatel96/bankadditionalfullcsv?datasetVersionNumber=1'),('UCI verified primary-source fallback','https://archive.ics.uci.edu/static/public/222/bank+marketing.zip')]
        failures=[]
        for origin,url in sources:
            try:
                request=urllib.request.Request(url,headers={'User-Agent':'tfm-bank-marketing/1.0'})
                with urllib.request.urlopen(request,timeout=60) as response: data=extract_csv(response.read())
                if hashlib.sha256(data).hexdigest()!=EXPECTED: raise ValueError('Source content changed; exact checksum mismatch.')
                break
            except Exception as e: failures.append(f'{origin}: {e}')
        else: raise RuntimeError('Download failed. '+'; '.join(failures))
    digest=hashlib.sha256(data).hexdigest()
    if digest!=EXPECTED: raise ValueError(f'Checksum mismatch: {digest}')
    dest.write_bytes(data); print(f'{dest.name}: verified SHA-256 {digest}; {origin}')

if __name__=='__main__': main()
