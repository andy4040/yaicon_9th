"""Portable Ref-DL3DV runner for the restored direct-beta two-set baseline.

This is a new launcher; model/trainer sources are preserved from yai5.
Use --dry-run to inspect every command without loading data or CUDA.
"""
from pathlib import Path
import argparse
import json
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--scenes', nargs='+', help='Full scene IDs; default: all eight')
    parser.add_argument('--resume', action='store_true', help='Resume each existing latest.pt')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    configs = ROOT / 'configs/exps/glint/ref-dl3dv'
    available = {p.stem for p in configs.glob('*.yaml')}
    scenes = args.scenes or sorted(available)
    unknown = set(scenes) - available
    if unknown:
        parser.error(f'Unknown scenes: {sorted(unknown)}')
    if len(set(scenes)) != len(scenes):
        parser.error('Duplicate scenes')
    data_root, output_root = args.data_root.resolve(), args.output_root.resolve()
    plans = []
    for scene in scenes:
        data, run = data_root / scene, output_root / scene
        checkpoint = run / 'checkpoints/latest.pt'
        config = configs / (scene + '.yaml')
        train = [sys.executable, '-m', 'glint.train_two_set', '--config', str(config),
                 '--data-root', str(data), '--output-dir', str(run),
                 '--transport-mode', 'direct_beta', '--ratio', '0.5',
                 '--max-steps', '60000', '--warmup-steps', '5000',
                 '--probation-steps', '5000', '--refine-stop', '30000',
                 '--normal-weight', '0.1', '--albedo-threshold', '0.05',
                 '--support-fraction', '0.6', '--min-views', '2',
                 '--selection-views', '24', '--save-every', '5000']
        if args.resume and checkpoint.exists():
            train += ['--resume', str(checkpoint)]
        elif not args.dry_run and run.exists() and any(run.iterdir()):
            parser.error(f'Output is not empty: {run}; use --resume or a new output root')
        evaluate = [sys.executable, '-m', 'glint.evaluate_two_set', '--config', str(config),
                    '--data-root', str(data), '--checkpoint', str(checkpoint),
                    '--output-dir', str(run / 'eval'), '--ratio', '0.5',
                    '--split', 'val', '--save-components']
        if not args.dry_run and not (data / 'sparse/0/points3D.ply').is_file():
            parser.error(f'Missing initialization point cloud: {data}/sparse/0/points3D.ply')
        plans.append((run, train, evaluate))
    for run, train, evaluate in plans:
        for command in (train, evaluate):
            print(shlex.join(command), flush=True)
        if args.dry_run:
            continue
        run.mkdir(parents=True, exist_ok=True)
        (run / 'launch_commands.json').write_text(
            json.dumps({'train': train, 'evaluate': evaluate}, indent=2), encoding='utf-8')
        subprocess.run(train, cwd=ROOT, check=True)
        subprocess.run(evaluate, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
