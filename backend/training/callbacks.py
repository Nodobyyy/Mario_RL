import json
import math
import shutil
import time
import uuid
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from backend.app import storage as s
from backend.app.processes import launch


class DashboardCallback(BaseCallback):
    def __init__(self, job_id, config):
        super().__init__()
        self.job_id, self.cfg = job_id, config
        self.started = time.monotonic()
        self.paused = 0.0
        self.last_metric = self.last_check = 0.0
        self.last_checkpoint = self.last_eval = 0
        self.start_steps = 0
        self.stop_requested = False
        self.auto_eval = None
        self.best_score = None
        self.eval_state = {}
        self.initialized = False

    def _on_training_start(self):
        self.initialized = True
        self.start_steps = self.num_timesteps
        self.last_checkpoint = self.last_eval = self.num_timesteps
        s.patch_job(self.job_id, status='running', state={
            'num_timesteps': self.num_timesteps, 'start_timesteps': self.start_steps,
            'target_timesteps': self.start_steps + self.cfg['total_timesteps'],
            'device': str(self.model.device), 'learning_rate': self.cfg['learning_rate'],
            'mode': 'training_preview', 'algorithm': 'ppo'})
        self.save_checkpoint('initial')

    def save_checkpoint(self, name):
        cp_id = uuid.uuid4().hex[:16]
        folder = s.DATA / 'checkpoints' / self.job_id / cp_id
        staging = folder.with_name('.' + cp_id)
        staging.mkdir(parents=True, exist_ok=True)
        self.model.save(staging / 'model.zip')
        vecnorm = self.model.get_vec_normalize_env()
        if vecnorm:
            vecnorm.save(str(staging / 'vecnormalize.pkl'))
        config = s.job_dir(self.job_id) / 'train_config.json'
        if config.exists():
            shutil.copy2(config, staging / 'train_config.json')
        staging.rename(folder)
        with s.db() as c:
            c.execute('INSERT INTO checkpoints(id,job_id,name,path,steps,created_at) VALUES(?,?,?,?,?,?)',
                      (cp_id, self.job_id, f'{name} · {self.num_timesteps:,}', str(folder / 'model.zip'),
                       self.num_timesteps, s.now()))
        s.patch_job(self.job_id, state={'latest_checkpoint': cp_id})
        self.last_checkpoint = self.num_timesteps
        return cp_id

    def collect_evaluation(self):
        if not self.auto_eval:
            return
        from backend.app.processes import reconcile
        reconcile()
        job = s.get_job(self.auto_eval)
        if job['status'] in ('completed', 'stopped', 'failed'):
            if job['status'] == 'completed':
                summary = job['state']
                score = (summary.get('clear_rate', 0), -summary.get('median_remaining_distance', 3161),
                         -summary.get('average_remaining_distance', 3161), summary.get('average_return', 0))
                cp_id = job['config']['checkpoint_id']
                self.eval_state = {'eval_success_rate': summary.get('clear_rate'), 'evaluation_id': job['id']}
                if self.best_score is None or score > self.best_score:
                    self.best_score = score
                    with s.db() as c:
                        c.execute('UPDATE checkpoints SET best=0 WHERE job_id=?', (self.job_id,))
                        c.execute('UPDATE checkpoints SET best=1,score=? WHERE id=?', (score[0], cp_id))
                    self.eval_state['best_checkpoint'] = cp_id
                s.metric(self.job_id, {'num_timesteps': self.num_timesteps, **self.eval_state})
            else:
                print('Automatic evaluation failed/stopped:', job.get('error'))
            self.auto_eval = None

    def launch_evaluation(self, cp_id):
        from backend.app.schemas import EvalConfig
        cfg = EvalConfig(checkpoint_id=cp_id, episodes=self.cfg['eval_episodes'],
                         max_steps=self.cfg['eval_steps'], realtime=False, record_video=False,
                         deterministic=True, seed=self.cfg['seed'], threads=min(2, self.cfg['threads'])).model_dump()
        try:
            job = launch('evaluation', cfg, parent_run=self.job_id)
            self.auto_eval = job['id']
            self.last_eval = self.num_timesteps
        except ValueError:
            pass  # Another demonstration is using the evaluator; retry next rollout.

    def _on_rollout_start(self):
        self.collect_evaluation()
        job = s.get_job(self.job_id)
        if job['command'] == 'pause':
            paused_at = time.monotonic()
            self.report()
            s.patch_job(self.job_id, status='paused')
            while s.get_job(self.job_id)['command'] == 'pause':
                self.collect_evaluation()
                time.sleep(0.25)
            self.paused += time.monotonic() - paused_at
            job = s.get_job(self.job_id)
        if job['command'] == 'stop':
            self.stop_requested = True
            return
        s.patch_job(self.job_id, status='running')
        cp_id = None
        if self.num_timesteps - self.last_checkpoint >= self.cfg['checkpoint_freq']:
            cp_id = self.save_checkpoint('checkpoint')
        if (self.cfg['eval_freq'] and self.num_timesteps - self.last_eval >= self.cfg['eval_freq']
                and self.auto_eval is None):
            self.launch_evaluation(cp_id or self.save_checkpoint('evaluation'))
        self.report()

    def report(self):
        elapsed = max(0.001, time.monotonic() - self.started - self.paused)
        with s.db() as c:
            rows = c.execute("SELECT * FROM episodes WHERE job_id=? AND kind='train' AND reason NOT IN ('interrupted','reset') ORDER BY id DESC LIMIT 100", (self.job_id,)).fetchall()
            total = c.execute("SELECT count(*),max(max_x) FROM episodes WHERE job_id=? AND kind='train' AND reason NOT IN ('interrupted','reset')", (self.job_id,)).fetchone()
        metrics = dict(num_timesteps=self.num_timesteps, start_timesteps=self.start_steps,
                       target_timesteps=self.start_steps + self.cfg['total_timesteps'],
                       episodes=total[0], max_x=total[1] or 0, elapsed_seconds=elapsed,
                       fps=(self.num_timesteps - self.start_steps) / elapsed,
                       learning_rate=self.model.lr_schedule(self.model._current_progress_remaining),
                       mean_reward=sum(x['reward'] for x in rows) / len(rows) if rows else None,
                       mean_length=sum(x['steps'] for x in rows) / len(rows) if rows else None,
                       **self.eval_state)
        for key in ('train/policy_gradient_loss', 'train/value_loss', 'train/entropy_loss', 'train/approx_kl'):
            value = self.model.logger.name_to_value.get(key)
            if value is not None and math.isfinite(float(value)):
                metrics[key.split('/')[-1]] = float(value)
        if 'entropy_loss' in metrics:
            metrics['policy_entropy'] = -metrics['entropy_loss']
        if self.model.device.type == 'cuda':
            import torch
            metrics['gpu_memory_mb'] = torch.cuda.memory_allocated(self.model.device) / 1024**2
        infos = self.locals.get('infos') or []
        if infos:
            metrics['current'] = {k: infos[0].get(k) for k in ('x_pos', 'y_pos', 'flag_get')}
            metrics['max_x'] = max(metrics['max_x'], *(int(info.get('x_pos', 0)) for info in infos))
        old = s.get_job(self.job_id)['state'].get('max_x', 0)
        metrics['max_x'] = max(old, metrics['max_x'])
        s.metric(self.job_id, metrics)
        self.last_metric = time.monotonic()

    def _on_step(self):
        tick = time.monotonic()
        if tick - self.last_check >= 0.2:
            self.last_check = tick
            self.stop_requested = s.get_job(self.job_id)['command'] == 'stop'
        if tick - self.last_metric >= 1:
            self.collect_evaluation()
            self.report()
        return not self.stop_requested

    def _on_training_end(self):
        cp_id = self.save_checkpoint('stopped' if self.stop_requested else 'final')
        if self.cfg['eval_freq'] and not self.auto_eval and not self.stop_requested:
            self.launch_evaluation(cp_id)
        # Await an already launched evaluation so the best-model record is durable.
        while self.auto_eval:
            self.collect_evaluation()
            if self.auto_eval:
                job = s.get_job(self.job_id)
                if job['command'] == 'stop':
                    self.stop_requested = True
                    from backend.app.processes import command
                    try:
                        command(self.auto_eval, 'stop')
                    except ValueError:
                        pass
                time.sleep(0.25)
        self.report()
