import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
import psutil
from .storage import ROOT, db, decode, get_job, job_dir, now, patch_job

ACTIVE = ('starting', 'running', 'pausing', 'paused', 'stopping')
LOCK = threading.Lock()
CHILDREN = {}


def reconcile():
    for job_id, p in list(CHILDREN.items()):
        if p.poll() is not None:
            CHILDREN.pop(job_id, None)
    with db() as c:
        rows = c.execute("SELECT * FROM jobs WHERE status IN ('starting','running','pausing','paused','stopping')").fetchall()
    for row in rows:
        job = decode(row)
        # Allow the launcher to persist its PID before considering a job orphaned.
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(job['updated_at'])).total_seconds()
        if age < 10:
            continue
        try:
            p = psutil.Process(job['pid']) if job['pid'] else None
            alive = p is not None and job['id'] in p.cmdline() and p.is_running()
        except (psutil.Error, TypeError):
            alive = False
        if not alive:
            patch_job(job['id'], status='failed', error='训练进程已退出。请查看 worker.log，可从最近 checkpoint 恢复。')


def launch(kind, config, *, parent_run=None):
    with LOCK:
        reconcile()
        with db() as c:
            busy = c.execute("SELECT id FROM jobs WHERE kind=? AND status IN ('starting','running','pausing','paused','stopping')", (kind,)).fetchone()
            if busy:
                raise ValueError(f'已有 {kind} 任务运行中：{busy[0]}')
        job_id = uuid.uuid4().hex[:16]
        config = dict(config)
        if parent_run:
            config['parent_run'] = parent_run
        stamp = now()
        with db() as c:
            c.execute('INSERT INTO jobs(id,kind,name,status,created_at,updated_at,config) VALUES(?,?,?,?,?,?,?)',
                      (job_id, kind, config.get('name', '独立评估'), 'starting', stamp, stamp,
                       json.dumps(config, ensure_ascii=False)))
        directory = job_dir(job_id)
        (directory / 'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
        try:
            with (directory / 'worker.log').open('a', encoding='utf-8') as log:
                env = {**os.environ, 'PYTHONUNBUFFERED': '1', 'PYTHONIOENCODING': 'utf-8'}
                p = subprocess.Popen([sys.executable, '-m', 'backend.training.worker', kind, job_id],
                                     cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                     creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0,
                                     start_new_session=os.name != 'nt')
            CHILDREN[job_id] = p
            with db() as c:
                c.execute('UPDATE jobs SET pid=? WHERE id=?', (p.pid, job_id))
        except Exception as e:
            patch_job(job_id, status='failed', error=str(e))
            raise
        return get_job(job_id)


def command(job_id, action):
    with LOCK, db() as c:
        row = c.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        status, kind = row['status'], row['kind']
        allowed = {'pause': ('running',), 'resume': ('paused', 'pausing'),
                   'stop': ACTIVE}
        if action not in allowed or status not in allowed[action] or (kind != 'train' and action != 'stop'):
            raise ValueError(f'无法对 {status} 任务执行 {action}')
        target = {'pause': 'pausing', 'resume': 'running', 'stop': 'stopping'}[action]
        cmd = 'run' if action == 'resume' else action
        c.execute('UPDATE jobs SET command=?, status=?, updated_at=? WHERE id=?', (cmd, target, now(), job_id))
    return get_job(job_id)
