"""Build official nes-py 8.2.1 ctypes core without installing Visual Studio.

The core uses no CPython API; a py3-none-win_amd64 wheel is valid.
All C++ and Python source comes from the hash-verified PyPI source release.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    if sys.platform != 'win32':
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'nes-py==8.2.1'], check=True)
        return
    out = ROOT / 'tmp' / 'nes-build'
    out.mkdir(parents=True, exist_ok=True)
    meta = json.load(urllib.request.urlopen('https://pypi.org/pypi/nes-py/8.2.1/json', timeout=60))
    item = next(x for x in meta['urls'] if x['packagetype'] == 'sdist')
    archive = out / 'nes.tar.gz'
    if not archive.exists():
        urllib.request.urlretrieve(item['url'], archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != item['digests']['sha256']:
        raise RuntimeError('NES source SHA256 mismatch')
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            target = (out / member.name).resolve()
            if not target.is_relative_to(out.resolve()) or member.issym() or member.islnk():
                raise RuntimeError('Unsafe archive')
        t.extractall(out)
    source = out / 'nes_py-8.2.1'
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'ziglang==0.13.0', 'wheel', 'setuptools<81'], check=True)
    cpp = list((source / 'nes_py/nes/src').glob('*.cpp')) + list((source / 'nes_py/nes/src/mappers').glob('*.cpp'))
    dll = source / 'nes_py/lib_nes_env.dll'
    env = os.environ.copy()
    # Keep build caches in the project, avoiding global compiler installation.
    env['ZIG_GLOBAL_CACHE_DIR'] = str(out / 'zig-cache')
    subprocess.run([sys.executable, '-m', 'ziglang', 'c++', '-shared', '-O3', '-std=c++14',
                    '-target', 'x86_64-windows-gnu', '-I', str(source / 'nes_py/nes/include'),
                    *map(str, cpp), '-o', str(dll)], check=True, env=env)
    (source / 'setup.py').write_text('''from setuptools import setup, find_packages
setup(name="nes-py", version="8.2.1", packages=find_packages(),
      package_data={"nes_py":["lib_nes_env.dll"]}, license="MIT",
      install_requires=["gym>=0.17.2", "numpy>=1.18.5", "pyglet>=1.4.0,<=1.5.21", "tqdm>=4.48.2"])
''', encoding='utf-8')
    subprocess.run([sys.executable, 'setup.py', 'bdist_wheel', '--plat-name', 'win_amd64'], cwd=source, check=True)
    wheel = next((source / 'dist').glob('*.whl'))
    wheels = ROOT / 'vendor' / 'wheels'
    wheels.mkdir(parents=True, exist_ok=True)
    shutil.copy2(wheel, wheels / wheel.name)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', str(wheels / wheel.name)], check=True)


if __name__ == '__main__':
    main()
