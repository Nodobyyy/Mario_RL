import sys
import traceback
from backend.app import storage as s


def main():
    kind, job_id = sys.argv[1:3]
    s.init()
    try:
        if kind == 'train':
            from .trainer import train
            train(job_id)
        elif kind == 'evaluation':
            from .evaluate import evaluate
            evaluate(job_id)
        else:
            raise ValueError(kind)
    except BaseException as e:
        traceback.print_exc()
        s.patch_job(job_id, status='failed', error=f'{type(e).__name__}: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
