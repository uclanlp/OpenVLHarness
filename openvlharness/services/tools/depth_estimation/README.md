# DA3 depth and camera server

One DA3 process provides `POST /v1/depth` for a single image and `POST /v1/pose`
for an ordered sequence of at least two images. Both wrappers must point at
that same process for the paper setup.

```bash
python -m venv /path/to/envs/da3
source /path/to/envs/da3/bin/activate
bash openvlharness/services/tools/depth_estimation/install.sh
CUDA_VISIBLE_DEVICES=0 bash openvlharness/services/tools/depth_estimation/run_depth_estimation_server.sh
```

The installer pins the tested DA3 source commit and Torch 2.10.0/Torchvision
0.25.0 CUDA 12.8 wheels. It installs upstream DA3 dependencies, NumPy below 2
and the HTTP/plotting dependencies. Gaussian-splat rendering (`gsplat`) and the
Gradio application extra are not needed by these depth/pose endpoints.
The fresh-install recipe has not been rerun as part of the documentation audit.

The equivalent installed-package command and health check are:

```bash
python -m uvicorn openvlharness.services.tools.depth_estimation.server:app --host 127.0.0.1 --port 8087
curl --fail http://127.0.0.1:8087/health
```

| Raw-process variable | Default |
| --- | --- |
| `DEPTH_ESTIMATION_MODEL_ID` | `depth-anything/DA3NESTED-GIANT-LARGE` |
| `DEPTH_ESTIMATION_DEVICE` | `auto` |
| `DEPTH_ESTIMATION_HOST` / `DEPTH_ESTIMATION_PORT` | `127.0.0.1` / `8087` |
| `DEPTH_ESTIMATION_COLORMAP` | `inferno` |

`DEPTH_ESTIMATION_MODEL_ID` may identify a locally downloaded checkpoint
directory. The checked-in default supplies depth in meters. Preserve its
checkpoint for comparisons; substituting another DA3 variant can change units.
Raw depth is base64 float32 plus its shape; intrinsics and world-to-camera
extrinsics retain image order. Pose uses the first supplied image as reference
when the router creates camera-motion evidence.

Install Matplotlib here for inferno depth rendering. The raw colormap helper
falls back to grayscale if its plotting import fails. Install Matplotlib in
the router environment as well: camera bird's-eye rendering runs there, not in
this DA3 process. Without router Matplotlib, numeric results remain available
but the camera visualization is absent.

The stage-1 environment had an incompatible inherited xFormers extension
warning, while depth and pose inference completed successfully through the
available implementation. That observation is not a dependency compatibility
claim. Keep accelerator packages compatible with the selected Torch/Python
versions. [Observed versions and revisions](../README.md) record the baseline.
