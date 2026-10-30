"""Method configuration and JSONL record writer for external experiment runners."""
import argparse
import json
from pathlib import Path

METHODS = {
    'B0': {'policy': 'anygrasp_moveit', 'recovery': 'none', 'execution': 'synchronous'},
    'B1': {'policy': 'smolvla', 'recovery': 'none', 'execution': 'synchronous'},
    'B2': {'policy': 'smolvla', 'recovery': 'fixed', 'execution': 'synchronous'},
    'B3': {'policy': 'smolvla', 'recovery': 'rule', 'execution': 'synchronous'},
    'B4': {'policy': 'failure_aware_smolvla', 'recovery': 'learned', 'execution': 'synchronous'},
    'B5': {'policy': 'failure_aware_smolvla', 'recovery': 'learned', 'execution': 'naive_async'},
    'B6': {'policy': 'failure_aware_smolvla', 'recovery': 'learned', 'execution': 'aligned_async'},
    'B7': {'policy': 'failure_aware_smolvla', 'recovery': 'anygrasp_assisted', 'execution': 'aligned_async'},
}


def write_record(path, record):
    if record.get('method') not in METHODS or not record.get('episode_id'):
        raise ValueError('record needs a known method and episode_id')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS, required=True)
    args = parser.parse_args()
    print(json.dumps({'method': args.method, **METHODS[args.method]}, indent=2))


if __name__ == '__main__':
    main()
