import uuid
import gymnasium as gym
import numpy as np
from backend.app.storage import DATA, db, now


class Telemetry(gym.Wrapper):
    """Outside upstream WarpFrame/Monitor: logs shaped, unnormalized rewards."""
    def __init__(self, env, job_id, worker_id, runtime, publisher=None, kind='train', checkpoint=None):
        super().__init__(env)
        self.job_id, self.worker_id, self.runtime = job_id, worker_id, runtime
        self.publisher, self.kind, self.checkpoint = publisher, kind, checkpoint
        self.points = []
        self.started = now()
        self.x = self.max_x = 0
        self.total = 0.0
        self.current = {}

    def reset(self, **kwargs):
        if self.points:
            self.finish('reset', False, False)
        obs, info = self.env.reset(**kwargs)
        self.started = now()
        self.x = self.max_x = 0
        self.total = 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.x = self.runtime.extract_sanitized_x_position(info, previous_x_pos=self.x)
        info = dict(info)
        info['x_pos'] = self.x
        self.max_x = max(self.max_x, self.x)
        self.total += float(reward)
        done = bool(terminated or truncated)
        self.current = dict(step=len(self.points) + 1, x_pos=self.x, y_pos=int(info.get('y_pos', 0)),
                            action=int(action), reward=float(reward), done=done,
                            episode_reward=self.total, max_x=self.max_x)
        self.points.append([self.current[k] for k in ('step', 'x_pos', 'y_pos', 'action', 'reward', 'done')])
        if self.publisher and self.publisher.due():
            self.publisher.submit(self.runtime.render_rgb_frame(self.env), self.current)
        if done:
            success = bool(info.get('flag_get', False))
            reason = 'flag_get' if success else ('stall' if info.get('stall_truncated') else
                     ('death' if terminated else 'time_limit'))
            self.finish(reason, success, bool(terminated and not success))
        return obs, reward, terminated, truncated, info

    def finish(self, reason, success=False, death=False):
        if not self.points:
            return
        path = DATA / 'trajectories' / self.job_id / f'{self.worker_id}-{uuid.uuid4().hex}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        points = np.asarray(self.points, dtype=np.float64)
        np.savez_compressed(path, points=points)
        with db() as c:
            c.execute('''INSERT INTO episodes(job_id,kind,worker_id,started_at,ended_at,
                      steps,reward,max_x,final_x,death,success,reason,trajectory,checkpoint)
                      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
                      (self.job_id, self.kind, self.worker_id, self.started, now(), len(points),
                       self.total, self.max_x, self.x, int(death), int(success), reason,
                       str(path), self.checkpoint))
        self.points = []

    def close(self):
        self.finish('interrupted')
        return self.env.close()
