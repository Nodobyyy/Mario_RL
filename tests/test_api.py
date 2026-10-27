import json
import uuid
import pytest
from fastapi.testclient import TestClient
from backend.app import storage as s
from backend.app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(s, 'DATA', tmp_path)
    with TestClient(app) as client:
        yield client


def make_job():
    job_id = uuid.uuid4().hex[:16]
    with s.db() as c:
        c.execute('INSERT INTO jobs(id,kind,name,status,created_at,updated_at,config) VALUES(?,?,?,?,?,?,?)',
                  (job_id,'train','test','running',s.now(),s.now(),'{}'))
    return job_id


def test_control_and_snapshot_survive_refresh(client):
    job_id = make_job()
    s.metric(job_id, {'num_timesteps':128,'start_timesteps':0,'target_timesteps':1024})
    assert client.post(f'/api/runs/{job_id}/pause',json={}).json()['status']=='pausing'
    assert client.post(f'/api/runs/{job_id}/resume',json={}).json()['status']=='running'
    assert client.get(f'/api/runs/{job_id}').json()['state']['num_timesteps']==128
    with client.websocket_connect(f'/ws/runs/{job_id}/metrics') as ws:
        message=ws.receive_json()
        assert message['job_id']==job_id
        assert message['state']['num_timesteps']==128
    assert client.post(f'/api/runs/{job_id}/stop',json={}).json()['status']=='stopping'
    assert client.post(f'/api/runs/{job_id}/resume',json={}).status_code==409
    assert len(client.get(f'/api/runs/{job_id}/metrics').json())==1


def test_trajectory_full_fidelity_and_episode_filter(client):
    import numpy as np
    job_id=make_job()
    path=s.DATA/'trajectory.npz'
    np.savez_compressed(path,points=np.array([[1,12,80,2,.5,0],[2,13,81,4,-1.25,1]]))
    with s.db() as c:
        result=c.execute('''INSERT INTO episodes(job_id,kind,worker_id,started_at,ended_at,steps,reward,max_x,final_x,death,success,reason,trajectory)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)''',(job_id,'train',0,s.now(),s.now(),2,-.75,13,13,1,0,'death',str(path)))
        episode_id=result.lastrowid
    trajectory=client.get(f'/api/episodes/{episode_id}/trajectory').json()
    assert trajectory['coordinate_system']=='game_world'
    assert len(trajectory['points'])==2
    assert trajectory['points'][-1]==dict(step=2,x_pos=13,y_pos=81,action=4,reward=-1.25,done=True)
    assert client.get(f'/api/runs/{job_id}/episodes?success=true').json()['total']==0
    assert client.get(f'/api/runs/{job_id}/episodes?success=false').json()['total']==1
    assert client.get('/api/episodes/9999').status_code==404


def test_validation_and_browser_origin(client):
    assert client.post('/api/runs',json={'n_steps':32,'n_envs':1,'batch_size':64}).status_code==422
    assert client.post('/api/runs',json={'n_envs':2}).status_code==422
    assert client.post('/api/evaluations',json={'episodes':0}).status_code==422
    assert client.post('/api/runs',json={},headers={'Origin':'https://untrusted.example'}).status_code==403
    assert client.get('/api/runs/no-such-run').status_code==404


def test_numpy_emulator_scalars_are_persisted_as_json(client):
    import numpy as np
    job_id=make_job()
    s.metric(job_id, {'num_timesteps':np.int64(128), 'current':{'x_pos':np.int32(523), 'flag_get':np.bool_(False)}, 'mean_reward':np.float32(.25)})
    state=client.get(f'/api/runs/{job_id}').json()['state']
    assert state['current']=={'x_pos':523,'flag_get':False}
    assert state['num_timesteps']==128


def test_global_progress_includes_all_parallel_workers(client):
    from types import SimpleNamespace
    from backend.training.callbacks import DashboardCallback
    job_id=make_job()
    cb=DashboardCallback(job_id, {'total_timesteps':1024, 'learning_rate':.00025})
    cb.model=SimpleNamespace(lr_schedule=lambda progress:.00025,
                            _current_progress_remaining=.5,
                            logger=SimpleNamespace(name_to_value={}), device=SimpleNamespace(type='cpu'))
    cb.num_timesteps=128
    cb.locals={'infos':[{'x_pos':100,'y_pos':80,'flag_get':False},
                       {'x_pos':2400,'y_pos':110,'flag_get':False}]}
    cb.report()
    state=client.get(f'/api/runs/{job_id}').json()['state']
    assert state['max_x']==2400
    assert state['current']['x_pos']==100


def test_preview_rate_is_live_config_with_valid_bounds(client):
    from backend.app.schemas import RunConfig
    assert RunConfig().preview_fps==30
    assert RunConfig(preview_fps=60).preview_fps==60
    job_id=make_job()
    for fps in (60,0,30):
        response=client.post(f'/api/runs/{job_id}/preview',json={'fps':fps})
        assert response.status_code==200
        assert response.json()['config']['preview_fps']==fps
        assert response.json()['status']=='running'
    assert client.post(f'/api/runs/{job_id}/preview',json={'fps':61}).status_code==422
    assert client.post(f'/api/runs/{job_id}/preview',json={'fps':-1}).status_code==422
    s.patch_job(job_id,status='completed')
    assert client.post(f'/api/runs/{job_id}/preview',json={'fps':15}).status_code==409


def test_frame_publisher_picks_up_changes_without_restarting(client):
    import time
    import numpy as np
    from backend.training.frames import FramePublisher
    job_id=make_job()
    publisher=FramePublisher(job_id,'training_preview',fps=8)
    def wait_for(predicate):
        deadline=time.monotonic()+4
        while time.monotonic()<deadline:
            if predicate():return
            time.sleep(.02)
        raise AssertionError('Preview config/frame did not arrive')
    try:
        client.post(f'/api/runs/{job_id}/preview',json={'fps':60})
        wait_for(lambda:publisher.fps==60)
        publisher.submit(np.zeros((240,256,3),dtype=np.uint8),{'step':1})
        wait_for(lambda:publisher.path.exists())
        assert json.loads(publisher.path.read_text())['state']['step']==1
        client.post(f'/api/runs/{job_id}/preview',json={'fps':0})
        wait_for(lambda:publisher.fps==0)
        assert not publisher.due()
        client.post(f'/api/runs/{job_id}/preview',json={'fps':30})
        wait_for(lambda:publisher.fps==30)
        assert publisher.due()
    finally:
        publisher.close()
