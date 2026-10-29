import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from . import storage as s
from .hardware import hardware
from .processes import command, launch, reconcile
from .schemas import EvalConfig, PreviewConfig, RunConfig
from backend.training.algorithms import checkpoint_algorithm, checkpoint_details


@asynccontextmanager
async def lifespan(app):
    s.init()
    s.register_pretrained()
    reconcile()
    yield
    # Workers deliberately survive a browser/API restart; controls are persisted.


app = FastAPI(title='Mario RL Dashboard', version='1.0.0', lifespan=lifespan)


@app.middleware('http')
async def same_origin(request, call_next):
    origin = request.headers.get('origin')
    if request.method not in ('GET', 'HEAD', 'OPTIONS') and origin:
        from urllib.parse import urlsplit
        parsed = urlsplit(origin)
        if parsed.netloc not in (request.headers.get('host'), '127.0.0.1:5173', 'localhost:5173'):
            return Response('Origin not allowed', status_code=403)
    return await call_next(request)


def require_job(job_id, kind=None):
    reconcile()
    job = s.get_job(job_id)
    if not job or (kind and job['kind'] != kind):
        raise HTTPException(404, '任务不存在')
    return job


def require_checkpoint(cp_id):
    cp = s.checkpoint(cp_id)
    if not cp or not Path(cp['path']).is_file():
        raise HTTPException(404, 'Checkpoint 不存在')
    return cp


def ready():
    hw = hardware()
    if hw['missing'] or hw['torch_error'] or not hw['upstream_ready']:
        raise HTTPException(503, {'message': '请先运行安装脚本及 doctor.py', 'system': hw})


@app.get('/api/system/status')
def system_status():
    return {**hardware(), 'timestamp': s.now(), 'data_dir': str(s.DATA), 'algorithms': ['ppo', 'a2c']}


@app.get('/api/runs')
def runs():
    reconcile()
    with s.db() as c:
        return [s.decode(x) for x in c.execute("SELECT * FROM jobs WHERE kind='train' ORDER BY created_at DESC")]


@app.post('/api/runs', status_code=201)
def create_run(config: RunConfig):
    ready()
    if config.resume_checkpoint:
        cp = require_checkpoint(config.resume_checkpoint)
        if checkpoint_algorithm(cp['path']) != config.algorithm:
            raise HTTPException(422, '起始权重的算法与所选训练算法不一致，请选择相同算法的权重或从零训练')
    try:
        return launch('train', config.model_dump())
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get('/api/runs/{job_id}')
def run(job_id: str):
    return require_job(job_id, 'train')


@app.post('/api/runs/{job_id}/preview')
def set_preview(job_id: str, config: PreviewConfig):
    require_job(job_id, 'train')
    with s.db() as c:
        row = c.execute('SELECT status,config FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row['status'] not in ('starting', 'running', 'pausing', 'paused'):
            raise HTTPException(409, '只有活动训练任务可以调整预览帧率')
        settings = json.loads(row['config'])
        settings['preview_fps'] = config.fps
        c.execute('UPDATE jobs SET config=?,updated_at=? WHERE id=?',
                  (json.dumps(settings, ensure_ascii=False), s.now(), job_id))
    return s.get_job(job_id)


@app.post('/api/runs/{job_id}/{action}')
def run_command(job_id: str, action: str):
    require_job(job_id, 'train')
    try:
        return command(job_id, action)
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get('/api/runs/{job_id}/metrics')
def metrics(job_id: str, limit: int = Query(2000, ge=1, le=10000), after: int = Query(0, ge=0)):
    require_job(job_id)
    with s.db() as c:
        rows = c.execute('SELECT * FROM metrics WHERE job_id=? AND id>? ORDER BY id DESC LIMIT ?',
                         (job_id, after, limit)).fetchall()
        return [dict(s.decode(x)['payload'], id=x['id']) for x in reversed(rows)]


@app.get('/api/runs/{job_id}/metrics/export')
def metrics_export(job_id: str):
    require_job(job_id)
    path = s.job_dir(job_id) / 'metrics.jsonl'
    if not path.exists():
        raise HTTPException(404, '尚无指标')
    return FileResponse(path, filename=f'{job_id}-metrics.jsonl')


@app.get('/api/runs/{job_id}/episodes')
def episodes(job_id: str, success: bool | None = None, episode_id: int | None = None,
             limit: int = Query(200, ge=1, le=1000), offset: int = Query(0, ge=0)):
    require_job(job_id)
    conditions, args = ['job_id=?'], [job_id]
    if success is not None:
        conditions.append('success=?')
        args.append(int(success))
    if episode_id is not None:
        conditions.append('id=?')
        args.append(episode_id)
    where = ' AND '.join(conditions)
    with s.db() as c:
        total = c.execute(f'SELECT count(*) FROM episodes WHERE {where}', args).fetchone()[0]
        rows = c.execute(f'SELECT * FROM episodes WHERE {where} ORDER BY id DESC LIMIT ? OFFSET ?',
                         (*args, limit, offset)).fetchall()
        return {'total': total, 'items': [dict(x) for x in rows], 'offset': offset}


@app.get('/api/episodes/{episode_id}')
def episode(episode_id: int):
    with s.db() as c:
        row = c.execute('SELECT * FROM episodes WHERE id=?', (episode_id,)).fetchone()
    if row is None:
        raise HTTPException(404, 'Episode 不存在')
    return dict(row)


@app.get('/api/episodes/{episode_id}/trajectory')
def trajectory(episode_id: int):
    import numpy as np
    item = episode(episode_id)
    with np.load(item['trajectory'], allow_pickle=False) as f:
        points = f['points']
    keys = ['step', 'x_pos', 'y_pos', 'action', 'reward', 'done']
    return dict(episode_id=episode_id, coordinate_system='game_world',
                axes={'x': '从关卡左侧起算的世界像素', 'y': '模拟器 y_pos：从底部向上'},
                points=[dict(zip(keys, [int(p[0]), int(p[1]), int(p[2]), int(p[3]), float(p[4]), bool(p[5])]))
                        for p in points])


@app.get('/api/checkpoints')
def checkpoints():
    s.register_pretrained()
    with s.db() as c:
        rows = [dict(x) for x in c.execute('SELECT * FROM checkpoints ORDER BY created_at DESC')]
    for row in rows:
        if Path(row['path']).is_file():
            row.update(checkpoint_details(row['path']))
        else:
            row.update(algorithm='ppo', world=1, stage=1)
    return rows


@app.get('/api/checkpoints/{cp_id}/download')
def checkpoint_download(cp_id: str):
    cp = require_checkpoint(cp_id)
    return FileResponse(cp['path'], filename=Path(cp['path']).name)


@app.post('/api/evaluations', status_code=201)
def create_evaluation(config: EvalConfig):
    ready()
    require_checkpoint(config.checkpoint_id)
    try:
        return launch('evaluation', config.model_dump())
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get('/api/evaluations')
def evaluations():
    reconcile()
    with s.db() as c:
        return [s.decode(x) for x in c.execute("SELECT * FROM jobs WHERE kind='evaluation' ORDER BY created_at DESC")]


@app.get('/api/evaluations/{job_id}')
def evaluation(job_id: str):
    return require_job(job_id, 'evaluation')


@app.post('/api/evaluations/{job_id}/stop')
def stop_evaluation(job_id: str):
    require_job(job_id, 'evaluation')
    try:
        return command(job_id, 'stop')
    except ValueError as e:
        raise HTTPException(409, str(e))


@app.get('/api/jobs/{job_id}/log')
def logs(job_id: str, lines: int = Query(100, ge=1, le=2000)):
    require_job(job_id)
    from collections import deque
    path = s.job_dir(job_id) / 'worker.log'
    if not path.exists():
        return {'text': ''}
    with path.open(encoding='utf-8', errors='replace') as f:
        return {'text': ''.join(deque(f, maxlen=lines))}


@app.get('/api/evaluations/{job_id}/video')
def video(job_id: str):
    job = require_job(job_id, 'evaluation')
    path = s.job_dir(job_id) / 'evaluation.mp4'
    if job['status'] not in ('completed', 'stopped') or not path.exists():
        raise HTTPException(404, '尚无已完成的视频')
    return FileResponse(path, media_type='video/mp4', filename=f'mario-{job_id}.mp4',
                        content_disposition_type='inline')


async def stream(ws, job_id, frames=False):
    from urllib.parse import urlsplit
    origin = ws.headers.get('origin')
    if origin and urlsplit(origin).netloc not in (ws.headers.get('host'), '127.0.0.1:5173', 'localhost:5173'):
        await ws.close(code=1008)
        return
    if not s.get_job(job_id):
        await ws.close(code=1008)
        return
    await ws.accept()
    last = None
    last_file = None
    frame_path = s.job_dir(job_id) / 'frame.json' if frames else None
    try:
        while True:
            if frames:
                try:
                    stat = frame_path.stat()
                    file_version = (stat.st_mtime_ns, stat.st_size)
                    if file_version != last_file:
                        msg = json.loads(frame_path.read_text(encoding='utf-8'))
                        if msg['timestamp'] != last:
                            await asyncio.wait_for(ws.send_json(msg), timeout=2)
                            last = msg['timestamp']
                        last_file = file_version
                except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                    pass
            else:
                reconcile()
                job = s.get_job(job_id)
                await ws.send_json(dict(type='metrics', job_id=job_id, timestamp=s.now(),
                                        status=job['status'], state=job['state'], error=job['error']))
            # Poll above the maximum capture rate; read/decode only changed files.
            await asyncio.sleep(1 / 120 if frames else 1)
    except (WebSocketDisconnect, RuntimeError, asyncio.TimeoutError, OSError):
        pass


@app.websocket('/ws/runs/{job_id}/metrics')
async def ws_metrics(ws: WebSocket, job_id: str):
    await stream(ws, job_id)


@app.websocket('/ws/runs/{job_id}/frames')
async def ws_frames(ws: WebSocket, job_id: str):
    await stream(ws, job_id, frames=True)


@app.websocket('/ws/evaluations/{job_id}/frames')
async def ws_eval_frames(ws: WebSocket, job_id: str):
    await stream(ws, job_id, frames=True)


DIST = s.ROOT / 'frontend' / 'dist'
if DIST.exists():
    app.mount('/', StaticFiles(directory=DIST, html=True), name='frontend')
else:
    @app.get('/', response_class=HTMLResponse)
    def index():
        return '<h1>Mario Lab API 已启动</h1><p>请构建前端后重启，或启动 frontend 的 Vite 开发服务器。</p><a href="/docs">API 文档</a>'
