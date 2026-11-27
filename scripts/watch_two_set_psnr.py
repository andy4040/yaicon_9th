"""Evaluate atomically saved checkpoints without pausing the trainer."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--interval', type=int, default=5000)
    parser.add_argument('--start-step', type=int, default=40000)
    parser.add_argument('--end-step', type=int, default=60000)
    parser.add_argument('--poll-seconds', type=float, default=10)
    parser.add_argument('--frame-delay', type=float, default=.5)
    parser.add_argument('--keep-last-checkpoints', type=int, default=0,
                        help='Keep this many evaluated checkpoint files; 0 keeps all. Metrics/images remain.')
    args = parser.parse_args()
    if args.interval <= 0 or args.poll_seconds <= 0 or args.keep_last_checkpoints < 0:
        parser.error('interval and poll-seconds must be positive')
    output = args.run_dir / 'periodic_psnr'
    output.mkdir(parents=True, exist_ok=True)
    # Prevent duplicate watchers for this run; OS releases the lock on exit.
    import fcntl
    lock = (output / 'watcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.nice(10)
    last_signature = None
    latest = args.run_dir / 'checkpoints/latest.pt'
    while True:
        try:
            stat = latest.stat()
            signature = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
            if signature == last_signature:
                time.sleep(args.poll_seconds)
                continue
            snapshot = output / 'pending.pt'
            snapshot.unlink(missing_ok=True)
            # Trainer replaces latest.pt atomically, so this link remains a
            # complete, immutable checkpoint even when the next save occurs.
            os.link(latest, snapshot)
            payload = torch.load(snapshot, map_location='cpu', weights_only=True)
            step = int(payload['step'])
            del payload
            if step < args.start_step or step % args.interval:
                snapshot.unlink()
                last_signature = signature
                time.sleep(args.poll_seconds)
                continue
            destination = output / f'step_{step:06d}'
            destination.mkdir(exist_ok=True)
            checkpoint = destination / 'checkpoint.pt'
            snapshot.replace(checkpoint)
            metrics = destination / 'metrics.json'
            if not metrics.exists():
                print(f'EVALUATING step={step}', flush=True)
                with (destination / 'eval.log').open('w') as log:
                    subprocess.run([
                        sys.executable, '-m', 'glint.evaluate_two_set',
                        '--config', args.config, '--data-root', args.data_root,
                        '--checkpoint', str(checkpoint), '--output-dir', str(destination),
                        '--split', 'val', '--ratio', '0.5',
                        '--frame-delay', str(args.frame_delay),
                    ], stdout=log, stderr=subprocess.STDOUT, check=True)
            records = []
            for path in sorted(output.glob('step_*/metrics.json')):
                data = json.loads(path.read_text())
                records.append(dict(step=data['checkpoint_step'], psnr=data['mean_psnr'],
                                    num_frames=data['num_frames'], ratio=data['ratio']))
            temporary = output / 'history.tmp'
            temporary.write_text(json.dumps(records, indent=2))
            temporary.replace(output / 'history.json')
            print(json.dumps(records[-1]), flush=True)
            if args.keep_last_checkpoints:
                evaluated = sorted(p for p in output.glob('step_*/checkpoint.pt')
                                   if (p.parent/'metrics.json').exists())
                for old in evaluated[:-args.keep_last_checkpoints]:
                    if not old.resolve().is_relative_to(output.resolve()):
                        raise ValueError('Checkpoint cleanup escaped run directory')
                    old.unlink()
            last_signature = signature
            if step >= args.end_step:
                print('DONE', flush=True)
                break
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            print(f'RETRY: {error}', flush=True)
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
