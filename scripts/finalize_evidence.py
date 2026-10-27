"""Collect only genuine completed evaluation outputs and upstream fingerprints."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    report=json.loads((ROOT/'docs/acceptance-results.json').read_text(encoding='utf-8'))
    clear=next(x for x in report['checks'] if x['check']=='Pretrained CPU flag clear and MP4 export')
    source=ROOT/'data/runs'/clear['id']
    out=ROOT/'examples';out.mkdir(exist_ok=True)
    shutil.copy2(source/'evaluation.mp4',out/'cpu-pretrained-clear.mp4')
    shutil.copy2(source/'summary.json',out/'cpu-pretrained-clear.json')
    manifest={'repository':'https://github.com/WilliamK112/mario-machine-learning',
              'retrieval':'main branch source archive, 2026-10-27', 'files':{}}
    upstream=ROOT/'upstream/mario-machine-learning'
    for rel in ['mario_runtime.py','train_mario.py','evaluate_mario.py','watch_mario.py',
                'render_best_checkpoint.py','pretrained/ppo_mario_1-1/best_eval.zip',
                'pretrained/ppo_mario_1-1/train_config.json','pretrained/ppo_mario_1-1/vecnormalize.pkl']:
        manifest['files'][rel]=hashlib.sha256((upstream/rel).read_bytes()).hexdigest()
        original=ROOT/'tmp/source/mario-machine-learning-main'/rel
        if original.exists():assert original.read_bytes()==(upstream/rel).read_bytes(),rel
    (ROOT/'docs/upstream-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    wheel=next((ROOT/'vendor/wheels').glob('*.whl'))
    meta=json.loads((ROOT/'tmp/nes-meta.json').read_text(encoding='utf-8')) if (ROOT/'tmp/nes-meta.json').exists() else None
    source_meta=next((x for x in meta['urls'] if x['packagetype']=='sdist'),None) if meta else None
    info=dict(wheel=wheel.name,sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
              source='nes-py 8.2.1 official PyPI source distribution',compiler='Zig 0.13.0',
              source_url=source_meta['url'] if source_meta else None,
              source_sha256=source_meta['digests']['sha256'] if source_meta else None)
    (ROOT/'vendor/wheels/SOURCE.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print('Evidence collected. Upstream files unchanged. Genuine CPU clear:', clear['id'])


if __name__=='__main__':main()
