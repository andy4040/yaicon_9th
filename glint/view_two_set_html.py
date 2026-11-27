"""Plain HTML camera viewer backed by GLINT; no Viser dependency."""
import argparse
import io
import json
import math
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from PIL import Image
import torch
from .camera import GlintCamera
from .dataset import GlintDataset
from .two_set_model import load_two_set
from .two_set_transport import render_two_set
from .optix_backend import OptixSurfelTracer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--port', type=int, default=8089)
    args = p.parse_args()
    model, payload = load_two_set(args.checkpoint, device='cuda')
    model.eval().requires_grad_(False)
    tracer = OptixSurfelTracer().eval()
    dataset = GlintDataset.from_config(args.config, split='val', data_root=args.data_root,
                                      ratio=.5, load_neighbors=False)
    center = model.general.get_xyz.detach().median(dim=0).values.cpu().numpy()
    presets = []
    for name, cam in dataset.cameras.items():
        pos = cam.camera_center.numpy()
        distance = max(float(np.linalg.norm(center-pos)), 1.)
        presets.append(dict(name=name, position=pos.tolist(),
                            target=(pos+cam.R.numpy().T[:,2]*distance).tolist(),
                            up=(-cam.R.numpy().T[:,1]).tolist(),
                            fov=2*math.atan(cam.image_height/(2*float(cam.K[1,1])))))
    cfg = payload['config']
    epsilon = cfg.get('ray_epsilon', 1e-4)*cfg.get('scene_scale', 1.)
    html = Path(__file__).with_name('web_viewer').joinpath('index.html').read_bytes()
    scene_label = Path(args.data_root).name.replace('_', ' ').title()
    html = html.replace(b'Scene 3', scene_label.encode('utf-8'))

    class Handler(BaseHTTPRequestHandler):
        def reply(self, body, kind, code=200, extra=None):
            self.send_response(code)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            for k,v in (extra or {}).items(): self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in ('/', '/index.html'):
                self.reply(html, 'text/html; charset=utf-8')
            elif path == '/config':
                self.reply(json.dumps(dict(step=payload['step'],presets=presets,
                    general=len(model.general.get_xyz),optical=len(model.optical.get_xyz))).encode(), 'application/json')
            else: self.reply(b'Not found','text/plain',404)

        def do_POST(self):
            if self.path != '/render':
                self.reply(b'Not found','text/plain',404); return
            try:
                size = int(self.headers.get('Content-Length',0))
                if not 0 < size < 8192: raise ValueError('Invalid request size')
                data = json.loads(self.rfile.read(size))
                def vector(key):
                    v = np.asarray(data[key],dtype=np.float64)
                    if v.shape != (3,) or not np.isfinite(v).all(): raise ValueError('Invalid camera')
                    return v
                def unit(v):
                    n = np.linalg.norm(v)
                    if n < 1e-8: raise ValueError('Degenerate camera')
                    return v/n
                pos, target, up = vector('position'),vector('target'),vector('up')
                forward = unit(target-pos)
                right = unit(np.cross(forward,up))
                down = np.cross(forward,right)
                w,h = int(data['width']),int(data['height'])
                fov = float(data['fov'])
                if not (64 <= w <= 960 and 64 <= h <= 960 and .05 < fov < 3):
                    raise ValueError('Invalid image dimensions/FOV')
                R = torch.tensor(np.stack((right,down,forward)),device='cuda',dtype=torch.float32)
                focal = h/(2*math.tan(fov/2))
                camera = GlintCamera(K=torch.tensor([[focal,0,w/2],[0,focal,h/2],[0,0,1]],device='cuda'),
                    R=R,T=-R@torch.tensor(pos,device='cuda',dtype=torch.float32),image_width=w,image_height=h)
                started = time.perf_counter()
                with torch.no_grad():
                    result = render_two_set(camera,model,tracer,ray_epsilon=epsilon)
                    mode = data.get('mode','final')
                    image = result.render.permute(1,2,0)
                    if mode == 'general': image = result.direct_rgb
                    elif mode == 'optical': image = result.optical_gate.expand(-1,-1,3)
                    elif mode in ('transmission','reflection'):
                        trace = getattr(result,mode+'_trace')
                        image = torch.zeros_like(image) if trace is None else result.optical_gate*getattr(result,mode+'_weight')*trace.rgb
                    pixels = (image.clamp(0,1)*255).byte().cpu().numpy()
                buffer = io.BytesIO()
                Image.fromarray(pixels).save(buffer,format='JPEG',quality=92)
                self.reply(buffer.getvalue(),'image/jpeg',extra={'X-Render-Ms':round((time.perf_counter()-started)*1000)})
            except (ValueError,KeyError,TypeError) as error:
                self.reply(str(error).encode(),'text/plain',400)
            except Exception as error:
                import traceback
                traceback.print_exc()
                self.reply(str(error).encode(),'text/plain',500)

        def log_message(self, *args): pass

    print(f'HTML_VIEWER_READY step={payload["step"]} port={args.port}',flush=True)
    HTTPServer(('127.0.0.1',args.port),Handler).serve_forever()


if __name__ == '__main__': main()
