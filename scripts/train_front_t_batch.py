"""Sequential fresh direct-beta runs; scene 3 is a historical result, never rerun.

Managed by supervisor. Completed scenes are skipped on restart; interrupted
scenes resume their own latest checkpoint. Model/training code is baseline 7a99838.
"""
from pathlib import Path
import argparse
import datetime
import fcntl
import json
import os
import signal
import subprocess
import sys
import time


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def validate_metric(metric, steps, views):
    if metric['checkpoint_step'] != steps or metric['num_frames'] != views:
        raise ValueError('Final metric step/view count does not match this experiment')
    if metric['transport_mode'] != 'direct_beta' or metric['split'] != 'val':
        raise ValueError('Unexpected evaluation mode/split')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/root/data/3d-front-t'))
    parser.add_argument('--output-root', type=Path, default=Path('results/front_t_direct_beta_baseline'))
    parser.add_argument('--preflight', type=Path, default=Path('experiments/front_t_preflight.json'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = (args.output_root/'batch.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inventory = json.loads(args.preflight.read_text())
    records = {'scene_3': dict(status='historical_result', step=60000, num_frames=25,
                              mean_psnr=35.53775317728046, transport_mode='direct_beta',
                              source='previous Scene3 run; reused by user instruction; NOT re-evaluated on this server')}
    status = dict(state='prepared', order=[1, 2, 4, 5], baseline_commit='7a99838',
                  max_steps=60000, results=records)
    children = []

    def stop_children():
        for process in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process in children:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

    def interrupted(signum, _frame):
        raise InterruptedError(f'Batch received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def save():
        status['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(args.output_root/'status.json', status)

    save()
    if args.dry_run:
        print(json.dumps(status), flush=True)
        return
    try:
        for scene in status['order']:
            name = f'scene_{scene}'
            views = inventory[name]['val_frames']
            data = args.data_root/name
            run = args.output_root/name
            run.mkdir(exist_ok=True)
            config = f'configs/exps/glint/3d-front-t/{name}.yaml'
            final = run/'periodic_psnr/step_060000/metrics.json'
            if final.exists():
                metric = json.loads(final.read_text())
                validate_metric(metric, 60000, views)
            else:
                train = [sys.executable, '-m', 'glint.train_two_set', '--config', config,
                         '--data-root', str(data), '--environment-ply', str(data/'envs/points3D.ply'),
                         '--output-dir', str(run), '--transport-mode', 'direct_beta',
                         '--max-steps', '60000', '--seed', '42', '--ratio', '0.5', '--save-every', '1000']
                latest = run/'checkpoints/latest.pt'
                if latest.exists():
                    train += ['--resume', str(latest)]
                watcher = [sys.executable, 'scripts/watch_two_set_psnr.py', '--run-dir', str(run),
                           '--config', config, '--data-root', str(data), '--start-step', '5000',
                           '--end-step', '60000', '--interval', '5000', '--frame-delay', '0.5',
                           '--keep-last-checkpoints', '1']
                status.update(state='training', current_scene=name, command=train)
                save()
                print(f'START {name}', flush=True)
                with (run/'console.log').open('a') as log, (run/'psnr_watcher.log').open('a') as eval_log:
                    trainer = subprocess.Popen(train, stdout=log, stderr=subprocess.STDOUT,
                                               stdin=subprocess.DEVNULL, start_new_session=True)
                    children.append(trainer)
                    evaluator = subprocess.Popen(watcher, stdout=eval_log, stderr=subprocess.STDOUT,
                                                 stdin=subprocess.DEVNULL, start_new_session=True)
                    children.append(evaluator)
                    status.update(train_pid=trainer.pid, watcher_pid=evaluator.pid)
                    save()
                    while trainer.poll() is None:
                        if evaluator.poll() not in (None, 0):
                            raise RuntimeError(f'{name} evaluation watcher failed')
                        time.sleep(5)
                    if trainer.returncode:
                        raise RuntimeError(f'{name} training exited with {trainer.returncode}')
                    status['state'] = 'final_evaluation'
                    save()
                    if evaluator.wait(timeout=900):
                        raise RuntimeError(f'{name} final evaluation failed')
                children.clear()
                metric = json.loads(final.read_text())
                validate_metric(metric, 60000, views)
            records[name] = dict(status='completed', step=60000, num_frames=views,
                                 mean_psnr=metric['mean_psnr'], transport_mode='direct_beta',
                                 metric_path=str(final), source='measured in this batch')
            print(json.dumps(records[name]), flush=True)
            save()
        status.update(state='completed', current_scene=None)
        status['mean_scene_psnr'] = sum(r['mean_psnr'] for r in records.values())/len(records)
        save()
    except BaseException as error:
        stop_children()
        status.update(state='stopped' if isinstance(error, InterruptedError) else 'failed', error=str(error))
        save()
        raise


if __name__ == '__main__':
    main()
