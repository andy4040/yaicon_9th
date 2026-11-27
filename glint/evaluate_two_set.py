"""Evaluate a two-set checkpoint on GLINT's existing dataset/view splits."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .two_set_transport import render_two_set


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ratio", type=float)
    parser.add_argument("--split", choices=("train", "val", "all"), default="val")
    parser.add_argument("--save-components", action="store_true")
    parser.add_argument("--frame-delay", type=float, default=0.0,
                        help="Seconds to yield between views during background evaluation")
    args = parser.parse_args()
    if args.frame_delay < 0:
        parser.error("frame-delay must be non-negative")
    if args.ratio is not None and not 0 < args.ratio <= 1:
        parser.error("ratio must be in (0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("Evaluation requires CUDA, gsplat and the OptiX backend")

    import imageio.v3 as iio
    from .dataset import GlintDataset
    from .optix_backend import OptixSurfelTracer
    from .two_set_model import load_two_set

    overrides = {"load_neighbors": False}
    if args.data_root is not None:
        overrides["data_root"] = args.data_root
    if args.ratio is not None:
        overrides["ratio"] = args.ratio
    dataset = GlintDataset.from_config(args.config, split=args.split, **overrides)
    model, payload = load_two_set(args.checkpoint, device="cuda")
    model.eval().requires_grad_(False)
    tracer = OptixSurfelTracer().eval()
    config = payload.get("config", {})
    epsilon = float(config.get("ray_epsilon", 1e-4)) * float(config.get("scene_scale", 1.0))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames = []

    def save_image(path: Path, image: torch.Tensor) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = image.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
        iio.imwrite(path, pixels)

    with torch.no_grad():
        for index in range(len(dataset)):
            frame = dataset[index].to("cuda")
            if index == 0:
                # Exclude lazy CUDA compilation and initial BVH construction
                # from per-view timings. All measured views are synchronized.
                render_two_set(frame.camera, model, tracer, ray_epsilon=epsilon)
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = render_two_set(frame.camera, model, tracer, ray_epsilon=epsilon)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            prediction = output.render.permute(1, 2, 0).clamp(0, 1)
            difference = prediction - frame.rgb
            mse = float(difference.square().mean())
            psnr = -10.0 * math.log10(max(mse, 1e-12))
            record = dict(index=index, camera=frame.camera_name, frame=frame.frame_name,
                          width=frame.camera.image_width, height=frame.camera.image_height,
                          psnr=psnr, l1=float(difference.abs().mean()), render_seconds=seconds)
            frames.append(record)
            filename = f"{index:05d}.png"
            save_image(args.output_dir / "rgb" / filename, prediction)
            if args.save_components:
                save_image(args.output_dir / "gt" / filename, frame.rgb)
                for component in ("reflection", "transmission"):
                    trace = getattr(output, component + "_trace", None)
                    contribution = torch.zeros_like(prediction) if trace is None else (
                        output.optical_gate * getattr(output, component + "_weight") * trace.rgb)
                    save_image(args.output_dir / component / filename, contribution)
                save_image(args.output_dir / "ordinary" / filename, output.direct_rgb)
                save_image(args.output_dir / "opaque_hypothesis" / filename, output.opaque_rgb)
                save_image(args.output_dir / "optical" / filename, output.optical_rgb)
                save_image(args.output_dir / "optical_gate" / filename, output.optical_gate[..., 0])
            print(json.dumps(record), flush=True)
            if args.frame_delay:
                time.sleep(args.frame_delay)

    if not frames:
        raise ValueError("The selected dataset split contains no frames")
    total_seconds = sum(frame["render_seconds"] for frame in frames)
    report = dict(
        format="glint-two-set-v1", checkpoint=str(args.checkpoint), split=args.split,
        transport_mode=model.transport_mode,
        dataset_root=str(dataset.data_root), ratio=dataset.ratio,
        checkpoint_phase=payload.get("phase"), checkpoint_step=payload.get("step"),
        general_gaussians=len(model.general.get_xyz), optical_gaussians=len(model.optical.get_xyz),
        num_frames=len(frames), mean_psnr=sum(f["psnr"] for f in frames)/len(frames),
        mean_l1=sum(f["l1"] for f in frames)/len(frames),
        mean_render_seconds=total_seconds/len(frames), fps=len(frames)/total_seconds,
        timing_scope="synchronized renderer only; excludes first warmup, image I/O and metrics",
        frames=frames,
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}), flush=True)


if __name__ == "__main__":
    main()
