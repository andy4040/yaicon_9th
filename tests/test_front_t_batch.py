import json
from pathlib import Path
import sys

import pytest

from scripts import train_front_t_batch as batch


@pytest.mark.parametrize('bad_views', [False, True])
def test_batch_skips_scene3_and_only_advances_after_valid_metric(tmp_path, monkeypatch, bad_views):
    inventory = tmp_path/'inventory.json'
    inventory.write_text(json.dumps({f'scene_{n}': {'val_frames': 22+n} for n in (1, 2, 4, 5)}))
    output = tmp_path/'runs'
    calls = []
    class Process:
        returncode = 0
        pid = 123
        def __init__(self, command, **kwargs):
            calls.append(command)
            if 'scripts/watch_two_set_psnr.py' in command:
                run = Path(command[command.index('--run-dir')+1])
                number = int(run.name.split('_')[1])
                final = run/'periodic_psnr/step_060000/metrics.json'
                final.parent.mkdir(parents=True)
                final.write_text(json.dumps(dict(checkpoint_step=60000, num_frames=1 if bad_views else 22+number,
                    mean_psnr=30., transport_mode='direct_beta', split='val')))
        def poll(self): return 0
        def wait(self, timeout=None): return 0
    monkeypatch.setattr(batch.subprocess, 'Popen', Process)
    monkeypatch.setattr(batch.os, 'chdir', lambda _: None)
    monkeypatch.setattr(batch.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(sys, 'argv', ['batch', '--preflight', str(inventory), '--output-root', str(output)])
    if bad_views:
        with pytest.raises(ValueError, match='view count'):
            batch.main()
        assert len(calls) == 2
    else:
        batch.main()
        assert len(calls) == 8
        trains = calls[::2]
        assert [Path(cmd[cmd.index('--data-root')+1]).name for cmd in trains] == ['scene_1','scene_2','scene_4','scene_5']
        assert all(cmd[cmd.index('--transport-mode')+1] == 'direct_beta' for cmd in trains)
        assert all('--ray-origin-grad' not in cmd and '--plane-weight' not in cmd for cmd in trains)
        status = json.loads((output/'status.json').read_text())
        assert status['state'] == 'completed'
        assert status['results']['scene_3']['status'] == 'historical_result'
        assert status['results']['scene_3']['mean_psnr'] == 35.53775317728046
        batch.main()
        assert len(calls) == 8  # Completed scenes never launch again.
