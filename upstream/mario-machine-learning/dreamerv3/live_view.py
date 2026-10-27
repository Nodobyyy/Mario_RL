import io
import json
import os
import glob
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
from PIL import Image

ROOT = os.path.dirname(__file__)
LOGDIR = os.path.join(ROOT, 'logs', 'mario_run3_rescue')
PORT = 8765

HTML = '''<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Mario Live View</title>
  <style>
    body { background:#0f1115; color:#e6edf3; font-family:Arial, sans-serif; margin:20px; }
    .row { display:flex; gap:24px; align-items:flex-start; }
    img { border:1px solid #30363d; background:#000; image-rendering:pixelated; width:512px; height:480px; }
    .card { background:#161b22; border:1px solid #30363d; padding:12px 14px; border-radius:8px; min-width:320px; }
    h2 { margin:0 0 8px 0; }
    .muted { color:#8b949e; }
  </style>
</head>
<body>
  <h2>Mario Live View</h2>
  <div class="muted">Auto-play latest episode (updates when a new episode file appears)</div>
  <div class="row" style="margin-top:12px;">
    <img id="frame" src="/frame.jpg" />
    <div class="card">
      <div><b>source:</b> <span id="src">-</span></div>
      <div><b>file:</b> <span id="file">-</span></div>
      <div><b>length:</b> <span id="len">-</span></div>
      <div><b>estimated max_x:</b> <span id="mx">-</span></div>
      <div><b>state:</b> <span id="state">-</span></div>
      <div><b>mtime:</b> <span id="mtime">-</span></div>
    </div>
  </div>
  <script>
    let epLen = 1;
    let idx = 0;

    async function refreshMeta(){
      const t = Date.now();
      try {
        const r = await fetch('/meta.json?t=' + t);
        const m = await r.json();
        epLen = Math.max(1, m.length || 1);
        document.getElementById('src').textContent = m.source;
        document.getElementById('file').textContent = m.file;
        document.getElementById('len').textContent = m.length;
        document.getElementById('mx').textContent = m.max_x;
        document.getElementById('state').textContent = m.state;
        document.getElementById('mtime').textContent = m.mtime;
      } catch(e) {}
    }

    function renderFrame(){
      const t = Date.now();
      document.getElementById('frame').src = '/frame.jpg?i=' + idx + '&t=' + t;
      idx = (idx + 1) % epLen;
    }

    refreshMeta();
    renderFrame();
    setInterval(refreshMeta, 1000);
    setInterval(renderFrame, 120);
  </script>
</body>
</html>'''


def latest_episode():
    files = glob.glob(os.path.join(LOGDIR, 'eval_eps', '*.npz')) + glob.glob(os.path.join(LOGDIR, 'train_eps', '*.npz'))
    if not files:
        return None
    f = max(files, key=os.path.getmtime)
    src = 'eval' if os.sep + 'eval_eps' + os.sep in f else 'train'
    d = np.load(f)
    frames = d['image']
    reward = d['reward']
    frame = frames[-1]
    pos_sum = float(sum(x for x in reward if 0 < x < 50))
    max_x = int(round(pos_sum / 0.05))
    length = int(len(reward))
    if reward.max() > 50:
        state = 'FLAG'
    elif length >= 1900:
        state = 'TIMER'
    else:
        state = 'DIED'
    return {
        'file': os.path.basename(f),
        'source': src,
        'length': length,
        'max_x': max_x,
        'state': state,
        'mtime': os.path.getmtime(f),
        'frame': frame,
        'frames': frames,
    }


def to_jpeg_bytes(arr):
    img = Image.fromarray(arr)
    img = img.resize((512, 480), resample=Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=90)
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, data, ctype='text/plain; charset=utf-8'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        return

    def do_GET(self):
        p = urlparse(self.path).path
        if p == '/':
            self._send(200, HTML.encode('utf-8'), 'text/html; charset=utf-8')
            return
        if p == '/meta.json':
            ep = latest_episode()
            if ep is None:
                payload = {'source':'-','file':'-','length':0,'max_x':0,'state':'NO_DATA','mtime':'-'}
            else:
                payload = {
                    'source': ep['source'],
                    'file': ep['file'],
                    'length': ep['length'],
                    'max_x': ep['max_x'],
                    'state': ep['state'],
                    'mtime': ep['mtime'],
                }
            self._send(200, json.dumps(payload).encode('utf-8'), 'application/json')
            return
        if p == '/frame.jpg':
            ep = latest_episode()
            if ep is None:
                arr = np.zeros((64, 64, 3), dtype=np.uint8)
            else:
                q = parse_qs(urlparse(self.path).query)
                i = 0
                if 'i' in q:
                    try:
                        i = int(q['i'][0])
                    except Exception:
                        i = 0
                frames = ep['frames']
                arr = frames[i % len(frames)]
            data = to_jpeg_bytes(arr)
            self._send(200, data, 'image/jpeg')
            return
        self._send(404, b'not found')


if __name__ == '__main__':
    srv = ThreadingHTTPServer(('127.0.0.1', PORT), Handler)
    print(f'LIVE VIEW: http://127.0.0.1:{PORT}')
    srv.serve_forever()
