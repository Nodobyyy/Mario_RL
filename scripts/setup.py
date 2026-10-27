"""Cross-platform setup; Windows needs no system Python or C++ installation."""
import argparse
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def call(args, **kwargs):
    print('>', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str,args)), check=True, cwd=ROOT, **kwargs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cuda', action='store_true', help='Install CUDA 12.1 PyTorch wheels instead of CPU wheels')
    p.add_argument('--dev', action='store_true')
    args = p.parse_args()
    if sys.version_info[:2] != (3,11):
        raise SystemExit('Use Python 3.11. On Windows, run setup.ps1 to download a project-local runtime automatically.')
    target = ROOT / '.venv311'
    python = target / ('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not python.exists():
        venv.create(target, with_pip=True)
    if Path(sys.prefix).resolve() != target.resolve():
        call([python, __file__, *sys.argv[1:]])
        return
    call([python,'-m','pip','install','--upgrade','pip','setuptools<81','wheel'])
    if importlib.util.find_spec('nes_py') is None:
        wheels = list((ROOT/'vendor/wheels').glob('*win_amd64.whl')) if os.name=='nt' else []
        if wheels:
            call([python,'-m','pip','install','--no-deps',wheels[0]])
        else:
            call([python, ROOT/'scripts/build_nes.py'])
    if args.cuda:
        call([python,'-m','pip','install','--upgrade','--force-reinstall','--no-deps','torch==2.4.1','--index-url','https://download.pytorch.org/whl/cu121'])
    elif os.name=='nt' or sys.platform.startswith('linux'):
        call([python,'-m','pip','install','torch==2.4.1','--index-url','https://download.pytorch.org/whl/cpu'])
    call([python,'-m','pip','install','-r','requirements-dev.txt' if args.dev else 'backend/requirements.txt'])
    call([python,ROOT/'scripts/doctor.py'])
    if not (ROOT/'frontend/dist/index.html').exists():
        print('Frontend build missing. Install Node.js 20+ then run npm install && npm run build in frontend/.')
    print('SETUP OK. Run start.bat (Windows) or python scripts/serve.py.')


if __name__=='__main__':
    main()
