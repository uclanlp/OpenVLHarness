# GroundingDINO extension

GroundingDINO is an optional alternative implementation of
`Visual_Grounding_Tool`. It is not part of the SAM3 paper baseline and was not
included in the stage-1 GPU validation.

In a separate model environment, install a compatible Torch/Torchvision pair,
Transformers with `AutoModelForZeroShotObjectDetection`, FastAPI, Uvicorn,
python-multipart, Pillow, and the OpenVLHarness package. For example, from the
source checkout after activating that environment:

```bash
python -m pip install . torch torchvision transformers fastapi uvicorn python-multipart
CUDA_VISIBLE_DEVICES=0 bash openvlharness/services/tools/grounding_dino/run_grounding_dino_server.sh
```

This extension recipe is not a tested version lock. The module entrypoint is:

```bash
python -m uvicorn openvlharness.services.tools.grounding_dino.server:app --host 127.0.0.1 --port 8084
curl --fail http://127.0.0.1:8084/health
```

`GROUNDING_DINO_MODEL_ID` defaults to `IDEA-Research/grounding-dino-base`;
`GROUNDING_DINO_DEVICE` defaults to `auto`. The shell launcher reads
`GROUNDING_DINO_HOST` and `GROUNDING_DINO_PORT`.

In a copy of an agent configuration, add `extensions` to
`capabilities.registries`, select `grounding.backend: grounding_dino`, and add
`services.tools.grounding_dino.server_url` pointing at this raw server.
These are configuration fragments, not a standalone configuration:

```yaml
capabilities:
  registries: [paper, extensions]
  tools:
    grounding:
      backend: grounding_dino
services:
  tools:
    grounding_dino:
      server_url: http://127.0.0.1:8084
```

The public tool remains `Visual_Grounding_Tool`; models cannot choose a raw
backend or override thresholds. The wrapper fixes box/text thresholds at
0.4/0.3. Canonical results preserve boxes, labels and scores, while masks and
mask count are null. The stored binding uses the `segment_` prefix and the
same box/score field names as SAM3 without claiming segmentation output.
