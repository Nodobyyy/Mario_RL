import argparse
from pathlib import Path
import sys
import threading
import webbrowser
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    if not (ROOT/'frontend/dist/index.html').exists():
        print('Frontend not built. Run npm install and npm run build in frontend/.')
    if not args.no_browser:
        threading.Timer(2, lambda: webbrowser.open(f'http://127.0.0.1:{args.port}')).start()
    import uvicorn
    uvicorn.run('backend.app.main:app', host=args.host, port=args.port, log_level='info')


if __name__=='__main__':
    main()
