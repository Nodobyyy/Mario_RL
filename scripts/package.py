"""Portable source distribution with built frontend and original pretrained policy."""
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','.venv','.venv311','.python311','tmp','node_modules','__pycache__','.pytest_cache','data','release'}


def main():
    out=ROOT/'release';out.mkdir(exist_ok=True)
    dest=out/'Mario-RL-Dashboard.zip'
    manifest=[]
    with ZipFile(dest,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for current, directories, files in os.walk(ROOT):
            directories[:]=sorted(d for d in directories if d not in EXCLUDED)
            for filename in sorted(files):
                path=Path(current)/filename
                if path.suffix=='.pyc' or path.is_symlink():continue
                rel=path.relative_to(ROOT)
                z.write(path,Path('Mario-RL-Dashboard')/rel)
                manifest.append(str(rel))
    digest=hashlib.sha256(dest.read_bytes()).hexdigest()
    (out/'SHA256SUMS.txt').write_text(f'{digest}  {dest.name}\n',encoding='utf-8')
    print(json.dumps(dict(path=str(dest),bytes=dest.stat().st_size,sha256=digest,files=len(manifest)),indent=2))


if __name__=='__main__':main()
