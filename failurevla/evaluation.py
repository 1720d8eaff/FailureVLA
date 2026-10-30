"""Aggregate actual JSONL episode records; missing measurements stay null."""
import argparse
import json
from pathlib import Path


def summarize(rows):
    if not rows:
        raise ValueError('no episode records')
    def mean(key):
        values = [r[key] for r in rows if r.get(key) is not None]
        return sum(values)/len(values) if values else None
    tp = sum(r.get('failure_tp', 0) for r in rows)
    fp = sum(r.get('failure_fp', 0) for r in rows)
    fn = sum(r.get('failure_fn', 0) for r in rows)
    attempted = sum(r.get('recovery_attempts', 0) for r in rows)
    successes = sum(r.get('recovery_successes', 0) for r in rows)
    rejected = sum(r.get('stale_rejected', 0) for r in rows)
    received = sum(r.get('chunks_received', 0) for r in rows)
    return {'episodes': len(rows), 'task_success_rate': mean('task_success'), 'grasp_success_rate': mean('grasp_success'),
            'failure_precision': tp/(tp+fp) if tp+fp else None, 'failure_recall': tp/(tp+fn) if tp+fn else None,
            'recovery_success_rate': successes/attempted if attempted else None, 'recovery_attempts': attempted,
            'mean_inference_blocking_s': mean('inference_blocking_s'), 'mean_task_time_s': mean('task_time_s'),
            'stale_rejection_rate': rejected/received if received else None, 'human_intervention_rate': mean('human_intervention')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('log', type=Path)
    a = p.parse_args()
    rows = [json.loads(line) for line in a.log.read_text(encoding='utf-8').splitlines() if line.strip()]
    methods = sorted({r['method'] for r in rows})
    print(json.dumps({method: summarize([r for r in rows if r['method'] == method]) for method in methods}, indent=2))


if __name__ == '__main__':
    main()
