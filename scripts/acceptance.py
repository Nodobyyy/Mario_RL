"""Real CPU acceptance through the same REST/WS routes as the browser.

Run after setup with --no server needed; TestClient drives the actual app and
real detached worker processes, not fake metrics or a surrogate environment.
"""
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from fastapi.testclient import TestClient
from backend.app.main import app


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parallel', action='store_true')
    args=parser.parse_args()
    report={'started_at':time.strftime('%Y-%m-%dT%H:%M:%S'), 'checks':[]}
    def passed(name, **details):
        report['checks'].append({'check':name,'passed':True,**details})
        print('PASS',name,json.dumps(details,ensure_ascii=False),flush=True)
    with TestClient(app) as c:
        def get(path):
            res=c.get('/api'+path);res.raise_for_status();return res.json()
        def post(path,payload):
            res=c.post('/api'+path,json=payload);res.raise_for_status();return res.json()
        def wait(kind, job_id, predicate, timeout=240):
            started=time.monotonic()
            while time.monotonic()-started<timeout:
                job=get(f'/{kind}/{job_id}')
                if job['status']=='failed':
                    raise RuntimeError(get(f'/jobs/{job_id}/log')['text'])
                if predicate(job):return job
                time.sleep(.3)
            raise TimeoutError(get(f'/jobs/{job_id}/log')['text'])
        hw=get('/system/status');assert not hw['missing']
        passed('CPU environment ready',python=hw['python'],torch=hw['torch'],cuda=hw['cuda'])
        ev=post('/evaluations',dict(episodes=1,max_steps=4000,realtime=False,record_video=True,deterministic=False,seed=42,device='cpu'))
        ev=wait('evaluations',ev['id'],lambda j:j['status']=='completed')
        assert ev['state']['flags_cleared']>0,ev
        video=c.get(f"/api/evaluations/{ev['id']}/video")
        assert video.status_code==200 and len(video.content)>10000
        passed('Pretrained CPU flag clear and MP4 export',id=ev['id'],flags=ev['state']['flags_cleared'],steps=ev['state']['eval_steps'],video_bytes=len(video.content))
        cfg=dict(name='验收 · CPU 真实 PPO',total_timesteps=20000,device='cpu',threads=2,n_steps=128,
                 batch_size=64,n_epochs=2,checkpoint_freq=512,eval_freq=0,preview_fps=8)
        run=post('/runs',cfg);rid=run['id']
        run=wait('runs',rid,lambda j:j['status']=='running' and j['state'].get('num_timesteps',0)>=128)
        post(f'/runs/{rid}/pause',{})
        paused=wait('runs',rid,lambda j:j['status']=='paused')
        n=paused['state']['num_timesteps'];time.sleep(.8)
        assert get(f'/runs/{rid}')['state']['num_timesteps']==n
        passed('Pause at safe rollout boundary',id=rid,timesteps=n)
        post(f'/runs/{rid}/resume',{})
        run=wait('runs',rid,lambda j:j['state'].get('episodes',0)>0 and j['state'].get('num_timesteps',0)>n)
        with c.websocket_connect(f'/ws/runs/{rid}/frames') as ws:
            frame=ws.receive_json()
            assert frame['mode']=='training_preview' and frame['image'].startswith('data:image/jpeg;base64,')
        with c.websocket_connect(f'/ws/runs/{rid}/metrics') as ws:
            assert ws.receive_json()['job_id']==rid
        passed('Real CPU PPO, training frames and metrics WebSockets',timesteps=run['state']['num_timesteps'],episodes=run['state']['episodes'])
        points=get(f'/runs/{rid}/episodes')['items']
        episode=next(e for e in points if e['reason'] not in ('interrupted','reset'))
        trajectory=get(f"/episodes/{episode['id']}/trajectory")
        assert len(trajectory['points'])==episode['steps']
        passed('Full raw trajectory saved',episode_id=episode['id'],points=len(trajectory['points']))
        post(f'/runs/{rid}/stop',{})
        stopped=wait('runs',rid,lambda j:j['status']=='stopped')
        cp=stopped['state']['latest_checkpoint']
        passed('Safe stop and checkpoint',checkpoint_id=cp,timesteps=stopped['state']['num_timesteps'])
        restored=post('/runs',{**cfg,'name':'验收 · Checkpoint 恢复','resume_checkpoint':cp,'total_timesteps':128})
        restored=wait('runs',restored['id'],lambda j:j['status']=='completed')
        assert restored['state']['start_timesteps']==stopped['state']['num_timesteps']
        assert restored['state']['num_timesteps']>=restored['state']['start_timesteps']+128
        passed('Resume model, optimizer and VecNormalize with cumulative SB3 steps',id=restored['id'],start=restored['state']['start_timesteps'],end=restored['state']['num_timesteps'])
        # An additional short run exercises asynchronous evaluation and best selection.
        auto=post('/runs',{**cfg,'name':'验收 · 自动评估','resume_checkpoint':cp,'total_timesteps':256,
                          'eval_freq':128,'eval_episodes':1,'eval_steps':64})
        auto=wait('runs',auto['id'],lambda j:j['status']=='completed')
        assert auto['state'].get('best_checkpoint') and auto['state'].get('evaluation_id')
        passed('Independent automatic evaluation and best checkpoint',id=auto['id'],best=auto['state']['best_checkpoint'])
        pretrained_resume=post('/runs',{**cfg,'name':'验收 · 预训练 CPU 微调','resume_checkpoint':'pretrained','total_timesteps':128})
        pretrained_resume=wait('runs',pretrained_resume['id'],lambda j:j['status']=='completed')
        assert pretrained_resume['state']['start_timesteps']==8050000
        passed('Resume upstream NumPy 2 pretrained policy and exact reward RMS on CPU',id=pretrained_resume['id'],start=pretrained_resume['state']['start_timesteps'],end=pretrained_resume['state']['num_timesteps'])
        if args.parallel:
            for n_envs in (4,8):
                p=post('/runs',{**cfg,'name':f'验收 · {n_envs} 进程','n_envs':n_envs,'vec_backend':'subproc',
                              'n_steps':32,'batch_size':64,'total_timesteps':n_envs*32,'threads':1,'n_epochs':1})
                p=wait('runs',p['id'],lambda j:j['status']=='completed',timeout=360)
                assert p['state']['num_timesteps']==n_envs*32
                passed(f'{n_envs}-worker spawn vector env',id=p['id'],steps=p['state']['num_timesteps'])
    report['finished_at']=time.strftime('%Y-%m-%dT%H:%M:%S')
    destination=ROOT/'docs/acceptance-results.json'
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('ALL ACCEPTANCE CHECKS PASSED',destination,flush=True)


if __name__=='__main__':
    main()
