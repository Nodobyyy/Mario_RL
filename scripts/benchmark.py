"""Paired end-to-end CPU PPO throughput measurements, including preview overhead."""
import json
from pathlib import Path
import statistics
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from backend.app.main import app


def main():
    samples=[]
    with TestClient(app) as c:
        for fps in (0,8,8,0):
            response=c.post('/api/runs',json=dict(name=f'基准 · 预览 {fps} FPS',device='cuda',
                            total_timesteps=1024,n_steps=128,batch_size=64,n_epochs=2,
                            threads=2,checkpoint_freq=5000,eval_freq=0,preview_fps=fps))
            response.raise_for_status()
            job=response.json();started=time.monotonic()
            while True:
                job=c.get('/api/runs/'+job['id']).json()
                if job['status']=='failed':
                    raise RuntimeError(c.get('/api/jobs/'+job['id']+'/log').json()['text'])
                if job['status']=='completed':break
                if time.monotonic()-started>180:raise TimeoutError(job)
                time.sleep(.2)
            state=job['state']
            assert state['device']=='cpu' and state['device_fallback'] is True
            samples.append(dict(id=job['id'],preview_fps=fps,training_fps=state['fps'],
                                elapsed_seconds=state['elapsed_seconds'],steps=state['num_timesteps']))
            print(json.dumps(samples[-1]),flush=True)
    baseline=statistics.median(x['training_fps'] for x in samples if x['preview_fps']==0)
    preview=statistics.median(x['training_fps'] for x in samples if x['preview_fps']==8)
    result=dict(samples=samples,baseline_median_fps=baseline,preview_median_fps=preview,
                measured_throughput_change_percent=(preview/baseline-1)*100,
                note='Windows CPU, 1 dummy env, 1024 steps, 2 epochs, 2 Torch threads; 2 paired samples per setting. Includes PPO updates and checkpoint I/O; short-run measurements contain noise. Explicit cuda requests fell back to CPU.')
    (ROOT/'docs/preview-benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
