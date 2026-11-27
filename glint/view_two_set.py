"""Interactive Viser camera with native two-set optical rendering."""
import argparse
import math
import time
import traceback
import numpy as np
import torch
import viser
import viser.transforms as vtf
from .camera import GlintCamera
from .dataset import GlintDataset
from .two_set_model import load_two_set
from .two_set_transport import render_two_set
from .optix_backend import OptixSurfelTracer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--port', type=int, default=8088)
    args = parser.parse_args()
    model, payload = load_two_set(args.checkpoint, device='cuda')
    model.eval().requires_grad_(False)
    tracer = OptixSurfelTracer().eval()
    dataset = GlintDataset.from_config(args.config, split='val', data_root=args.data_root,
                                      ratio=.5, load_neighbors=False)
    first = dataset[0].camera
    server = viser.ViserServer(host='127.0.0.1', port=args.port)
    server.scene.set_global_visibility(False)
    server.gui.add_markdown(f'## GLINT two-set viewer\nCheckpoint **{payload["step"]:,}** steps\n\nDrag: orbit · wheel: zoom · right drag: pan.\n\nTraining continues separately.')
    mode = server.gui.add_dropdown('View', options=('Final render','General radiance','Optical coverage','Transmission','Reflection'))
    width = server.gui.add_slider('Render width', min=240, max=960, step=120, initial_value=480)
    preset = server.gui.add_dropdown('Camera preset', options=tuple(dataset.camera_names))
    reset = server.gui.add_button('Go to preset')
    status = server.gui.add_markdown('Waiting for browser camera...')
    center = model.general.get_xyz.detach().median(dim=0).values.cpu().numpy()

    def pose(camera):
        position = camera.camera_center.numpy().astype(float)
        distance = max(float(np.linalg.norm(center-position)), 1.0)
        return position, position+camera.R.numpy().T[:, 2]*distance, -camera.R.numpy().T[:, 1]

    position, target, up = pose(first)
    server.initial_camera.position = position
    server.initial_camera.look_at = target
    server.initial_camera.up = up
    server.initial_camera.fov = 2*math.atan(first.image_height/(2*float(first.K[1,1])))
    server.scene.set_up_direction(up)
    revision = [0]
    @mode.on_update
    @width.on_update
    def changed(_):
        revision[0] += 1

    @reset.on_click
    def go(event):
        camera = dataset.cameras[preset.value]
        position, target, up = pose(camera)
        for client in server.get_clients().values():
            with client.atomic():
                client.camera.position = position
                client.camera.look_at = target
                client.camera.up_direction = up
                client.camera.fov = 2*math.atan(camera.image_height/(2*float(camera.K[1,1])))
        revision[0] += 1

    signatures = {}
    cfg = payload['config']
    epsilon = cfg.get('ray_epsilon', 1e-4)*cfg.get('scene_scale', 1.)
    print(f'VIEWER_READY step={payload["step"]} port={args.port}', flush=True)
    while True:
        for client_id, client in list(server.get_clients().items()):
            try:
                c = client.camera
                signature = (c.update_timestamp, revision[0])
                if signatures.get(client_id) == signature:
                    continue
                w = int(width.value)
                h = max(16, round(w/max(c.aspect, .1)))
                h = min(h, 960)
                focal = h/(2*math.tan(c.fov/2))
                rotation = vtf.SO3(c.wxyz).as_matrix() @ np.diag([1.,-1.,-1.])
                R = torch.tensor(rotation.T, device='cuda', dtype=torch.float32)
                position = torch.tensor(c.position, device='cuda', dtype=torch.float32)
                camera = GlintCamera(K=torch.tensor([[focal,0,w/2],[0,focal,h/2],[0,0,1]],device='cuda'),
                                     R=R,T=-R@position,image_width=w,image_height=h)
                started = time.perf_counter()
                with torch.no_grad():
                    output = render_two_set(camera, model, tracer, ray_epsilon=epsilon)
                    image = output.render.permute(1,2,0)
                    if mode.value == 'General radiance': image = output.direct_rgb
                    elif mode.value == 'Optical coverage': image = output.optical_gate.expand(-1,-1,3)
                    elif mode.value == 'Transmission' and output.transmission_trace is not None:
                        image = output.optical_gate*output.transmission_weight*output.transmission_trace.rgb
                    elif mode.value == 'Reflection' and output.reflection_trace is not None:
                        image = output.optical_gate*output.reflection_weight*output.reflection_trace.rgb
                    pixels = (image.clamp(0,1)*255).byte().cpu().numpy()
                client.scene.set_background_image(pixels, format='jpeg', jpeg_quality=90)
                status.content = f'{w} × {h} · {(time.perf_counter()-started)*1000:.0f} ms/frame'
                signatures[client_id] = signature
            except Exception:
                traceback.print_exc()
                time.sleep(.5)
        time.sleep(.15)  # Throttle to leave GPU capacity for ongoing training.


if __name__ == '__main__':
    main()
