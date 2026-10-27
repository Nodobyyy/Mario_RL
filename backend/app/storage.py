"""Shared WAL database. Workers own their connections; no shared connection objects."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get('MARIO_DATA_DIR', ROOT / 'data')).resolve()
UPSTREAM = ROOT / 'upstream' / 'mario-machine-learning'


def now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    """Convert emulator NumPy scalars before committing JSON state."""
    import math
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'item'):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@contextmanager
def db():
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DATA / 'dashboard.sqlite3', timeout=30)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init():
    with db() as c:
        c.execute('PRAGMA journal_mode=WAL')
        c.executescript('''
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,
          status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          config TEXT NOT NULL, state TEXT NOT NULL DEFAULT '{}',
          command TEXT NOT NULL DEFAULT 'run', pid INTEGER, error TEXT);
        CREATE TABLE IF NOT EXISTS metrics (
          id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
          timestamp TEXT NOT NULL, payload TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS metrics_job ON metrics(job_id,id);
        CREATE TABLE IF NOT EXISTS episodes (
          id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL, kind TEXT NOT NULL,
          worker_id INTEGER NOT NULL, started_at TEXT NOT NULL, ended_at TEXT NOT NULL,
          steps INTEGER NOT NULL, reward REAL NOT NULL, max_x INTEGER NOT NULL,
          final_x INTEGER NOT NULL, death INTEGER NOT NULL, success INTEGER NOT NULL,
          reason TEXT NOT NULL, trajectory TEXT NOT NULL, checkpoint TEXT);
        CREATE INDEX IF NOT EXISTS episode_job ON episodes(job_id,id);
        CREATE TABLE IF NOT EXISTS checkpoints (
          id TEXT PRIMARY KEY, job_id TEXT, name TEXT NOT NULL, path TEXT UNIQUE NOT NULL,
          steps INTEGER NOT NULL, created_at TEXT NOT NULL, score REAL, best INTEGER DEFAULT 0);
        ''')


def job_dir(job_id):
    p = DATA / 'runs' / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def decode(row):
    if row is None:
        return None
    item = dict(row)
    for key in ('config', 'state', 'payload'):
        if key in item:
            item[key] = json.loads(item[key])
    return item


def get_job(job_id):
    with db() as c:
        return decode(c.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone())


def patch_job(job_id, *, status=None, state=None, error=None):
    with db() as c:
        row = c.execute('SELECT state FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            return
        values = {'updated_at': now()}
        if state is not None:
            merged = json.loads(row[0])
            merged.update(jsonable(state))
            values['state'] = json.dumps(merged, allow_nan=False)
        if status is not None:
            values['status'] = status
        if error is not None:
            values['error'] = error
        c.execute('UPDATE jobs SET ' + ','.join(f'{k}=?' for k in values) + ' WHERE id=?',
                  (*values.values(), job_id))


def metric(job_id, payload):
    payload = jsonable({'job_id': job_id, 'timestamp': now(), **payload})
    with db() as c:
        c.execute('INSERT INTO metrics(job_id,timestamp,payload) VALUES(?,?,?)',
                  (job_id, payload['timestamp'], json.dumps(payload, allow_nan=False)))
    # SQLite is authoritative. JSONL is a portable export of the same points.
    with (job_dir(job_id) / 'metrics.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(payload, allow_nan=False) + '\n')
    patch_job(job_id, state=payload)


def register_pretrained():
    p = UPSTREAM / 'pretrained' / 'ppo_mario_1-1' / 'best_eval.zip'
    if p.exists():
        from zipfile import ZipFile
        with ZipFile(p) as z:
            steps = int(json.loads(z.read('data'))['num_timesteps'])
        with db() as c:
            c.execute('INSERT OR IGNORE INTO checkpoints(id,name,path,steps,created_at,best) VALUES(?,?,?,?,?,1)',
                      ('pretrained', '上游 PPO · Mario 1-1', str(p), steps, now()))
            c.execute('UPDATE checkpoints SET steps=? WHERE id=?', (steps, 'pretrained'))


def checkpoint(cp_id):
    with db() as c:
        row = c.execute('SELECT * FROM checkpoints WHERE id=?', (cp_id,)).fetchone()
        return dict(row) if row else None
