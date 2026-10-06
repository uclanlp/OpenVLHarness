# SAM3 raw server

Create a dedicated Python 3.10 environment and run the installer from the
source checkout:

```bash
python -m venv /path/to/envs/sam3
source /path/to/envs/sam3/bin/activate
bash src/openvlharness/services/tools/sam3_segment/install.sh
```

Alternatively, `environment-sam3.yml` creates a minimal conda environment;
activate it before running `install.sh`. This replaces the old full-machine
conda export. The installer pins the tested SAM3 source commit and Torch
2.10.0/Torchvision 0.25.0 CUDA 12.8 wheels. It resolves upstream dependencies,
including NumPy below 2; it does not reproduce every inherited package in the
observed stage-1 environment. See [deployment notes](../README.md).

Obtain access to the upstream `facebook/sam3` checkpoint and authenticate
Hugging Face if its download requires authentication. The raw loader calls
upstream `build_sam3_image_model()` and uses its Hugging Face cache. It has no
checkpoint-revision environment option. Stage 1 used the cached checkpoint
revision recorded in the deployment notes, with offline resolution enabled.
Verify the cached revision before claiming an identical checkpoint on a new host.

```bash
CUDA_VISIBLE_DEVICES=0 bash src/openvlharness/services/tools/sam3_segment/run_sam3_server.sh
```

The equivalent installed-package entrypoint is:

```bash
python -m uvicorn openvlharness.services.tools.sam3_segment.server:app --host 127.0.0.1 --port 8086
curl --fail http://127.0.0.1:8086/health
```

`SAM3_HOST` and `SAM3_PORT` configure the shell launcher. Configure the router's
`services.tools.sam3_segment.server_url` separately; the paper YAML expands
`SAM3_SERVER_URL` there. Raw requests use multipart `POST /v1/segment` with
`prompt` and `image` fields.

Paper behavior retains disabled presence weighting, IoM overlap threshold
0.3, aligned left-to-right box/score/RLE ordering and numbered overlays.
The model and processor are shared within the raw process. Use the router's
SAM3 gate for the paper deployment; do not assume independent routers provide
one global gate. The SAM3 overlay renderer requires Matplotlib, OpenCV and
pycocotools in this environment.
