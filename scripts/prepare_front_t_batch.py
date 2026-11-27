"""Verify and extract the four requested scene archives, then validate all paths."""
from pathlib import Path
import hashlib
import json
import tempfile
import zipfile


def main():
    import torch
    from glint.dataset import GlintDataset
    data_root = Path('/root/data/3d-front-t')
    data_root.mkdir(parents=True, exist_ok=True)
    report = {}
    for number in (1, 2, 4, 5):
        name = f'scene_{number}'
        archive = Path('/root')/f'{name}-only.zip'
        expected = archive.with_suffix('.sha256').read_text().strip()
        with archive.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'{name}: archive SHA256 mismatch')
        destination = data_root/name
        if not destination.exists():
            staging = Path(tempfile.mkdtemp(prefix=f'.{name}-', dir=data_root))
            with zipfile.ZipFile(archive) as source:
                for info in source.infolist():
                    path = Path(info.filename)
                    if path.is_absolute() or '..' in path.parts or path.parts[0] != name:
                        raise ValueError(f'Unsafe archive path: {info.filename}')
                source.extractall(staging)
            (staging/name).rename(destination)
            staging.rmdir()
        for file in ('intri.yml', 'extri.yml', 'sparse/0/points3D.ply', 'envs/points3D.ply'):
            if not (destination/file).is_file():
                raise FileNotFoundError(destination/file)
        item = dict(archive_sha256=actual, root=str(destination))
        for split in ('train', 'val'):
            dataset = GlintDataset.from_config(f'configs/exps/glint/3d-front-t/{name}.yaml',
                split=split, data_root=destination, ratio=.5, load_neighbors=False)
            for camera, frame in dataset.entries:
                for path in dataset._paths(camera, frame).values():
                    if isinstance(path, Path) and not path.is_file():
                        raise FileNotFoundError(path)
            for index in (0, len(dataset)//2, len(dataset)-1):
                sample = dataset[index]
                for value in (sample.rgb, sample.normal, sample.diffuse_albedo):
                    if value is None or not torch.isfinite(value).all():
                        raise ValueError(f'Invalid data in {name}/{split}/{index}')
            item[f'{split}_frames'] = len(dataset)
            item['image_shape'] = list(sample.rgb.shape)
        report[name] = item
        print(json.dumps({name: item}), flush=True)
    output = Path('experiments/front_t_preflight.json')
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
