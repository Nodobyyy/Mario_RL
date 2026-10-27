"""Verify real CPU PPO preview delivery and live controls in isolated storage."""
import json
import os
from pathlib import Path
import sys
import time
import uuid
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Do not occupy or mutate the user's interactive training workspace.
os.environ['MARIO_DATA_DIR']=str(ROOT/'tmp'/('preview-check-'+uuid.uuid4().hex[:8]))
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app import storage as s


def main():
    result={'timestamp':s.now(),'device':'cpu','samples':[]}
    with TestClient(app) as client:
        def get():
            response=client.get('/api/runs/'+run_id)
            response.raise_for_status()
            return response.json()
        def wait_for(predicate, timeout=60):
            started=time.monotonic()
            while time.monotonic()-started<timeout:
                job=get()
                if job['status']=='failed':
                    raise RuntimeError(client.get('/api/jobs/'+run_id+'/log').json()['text'])
                if predicate(job):return job
                time.sleep(.05)
            raise TimeoutError(get())
        def set_rate(fps):
            response=client.post('/api/runs/'+run_id+'/preview',json={'fps':fps})
            response.raise_for_status()
        response=client.post('/api/runs',json=dict(name='高帧率验收 · CPU PPO',total_timesteps=20000,
                    device='cpu',threads=2,n_steps=128,batch_size=64,n_epochs=1,eval_freq=0))
        response.raise_for_status()
        job=response.json();run_id=job['id']
        assert job['config']['preview_fps']==30
        frame_path=s.job_dir(run_id)/'frame.json'
        try:
            wait_for(lambda job:job['status']=='running' and frame_path.exists())
            with client.websocket_connect('/ws/runs/'+run_id+'/frames') as ws:
                for target in (30,60):
                    set_rate(target)
                    # Drain pre-change frames and allow the I/O thread's config poll.
                    cutoff=time.time()+.8
                    while True:
                        message=ws.receive_json()
                        if datetime.fromisoformat(message['timestamp']).timestamp()>=cutoff:break
                    first_capture=datetime.fromisoformat(message['timestamp']).timestamp()
                    started=time.monotonic();count=1
                    while time.monotonic()-started<3:
                        message=ws.receive_json()
                        assert message['mode']=='training_preview'
                        count+=1
                    elapsed=time.monotonic()-started
                    last_capture=datetime.fromisoformat(message['timestamp']).timestamp()
                    sample=dict(target_fps=target,delivered_fps=(count-1)/elapsed,
                                captured_fps=(count-1)/(last_capture-first_capture),frames=count)
                    assert sample['delivered_fps']>12,sample
                    result['samples'].append(sample)
                    print(json.dumps(sample),flush=True)
            before=get()['state']['num_timesteps']
            set_rate(0)
            time.sleep(1)
            frozen=json.loads(frame_path.read_text())['timestamp']
            time.sleep(.4)
            assert json.loads(frame_path.read_text())['timestamp']==frozen
            wait_for(lambda job:job['state']['num_timesteps']>before)
            result['disable_preview_keeps_training']=True
            set_rate(30)
            wait_for(lambda job:json.loads(frame_path.read_text())['timestamp']!=frozen)
            result['resume_preview_without_worker_restart']=True
        finally:
            job=get()
            if job['status'] in ('starting','running','pausing','paused'):
                client.post('/api/runs/'+run_id+'/stop',json={}).raise_for_status()
                wait_for(lambda job:job['status']=='stopped')
        result['run_id']=run_id
        result['data_dir']=str(s.DATA)
        result['final_status']=get()['status']
    (ROOT/'docs/preview-upgrade-results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PREVIEW VERIFICATION PASSED',flush=True)


if __name__=='__main__':main()
