import base64
import json
import queue
import threading
import time
from backend.app.storage import get_job, job_dir, now


class FramePublisher:
    """Single-element mailbox: the sampling process never waits for JPEG or I/O."""
    def __init__(self, job_id, mode, fps=30):
        self.job_id, self.mode, self.fps = job_id, mode, fps
        self.path = job_dir(job_id) / 'frame.json'
        self.mailbox = queue.Queue(maxsize=1)
        self.last = 0.0
        self.closed = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def due(self):
        return self.fps > 0 and time.monotonic() - self.last >= 1 / self.fps

    def submit(self, frame, state):
        if not self.due():
            return
        self.last = time.monotonic()
        item = (frame.copy(), dict(state), now())
        try:
            self.mailbox.put_nowait(item)
        except queue.Full:
            try:
                self.mailbox.get_nowait()
                self.mailbox.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def _write(self):
        import cv2
        next_config_check = 0.0
        while not self.closed.is_set() or not self.mailbox.empty():
            # Controls are read on this I/O thread, never on the sampling path.
            tick = time.monotonic()
            if self.mode == 'training_preview' and tick >= next_config_check:
                try:
                    job = get_job(self.job_id)
                    if job:
                        self.fps = max(0, min(60, int(job['config'].get('preview_fps', self.fps))))
                except (OSError, ValueError):
                    pass
                next_config_check = tick + 0.5
            try:
                frame, state, captured = self.mailbox.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ok:
                    continue
                msg = dict(type='frame', job_id=self.job_id, timestamp=captured,
                           mode=self.mode, worker_id=0, state=state,
                           image='data:image/jpeg;base64,' + base64.b64encode(jpg).decode())
                temp = self.path.with_suffix('.tmp')
                temp.write_text(json.dumps(msg), encoding='utf-8')
                temp.replace(self.path)
            except OSError as e:
                self.error = str(e)  # Skip this frame; training must continue.

    def close(self):
        self.closed.set()
        self.thread.join(timeout=3)
