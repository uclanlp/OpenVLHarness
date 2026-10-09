# Third-party notices

OpenVLHarness is released under the MIT License (see [LICENSE](LICENSE)). The
repository also contains or depends on the following third-party work.

## Code

- **VLMEvalKit** (Copyright 2023 VLMEvalKit Authors, Apache License 2.0).
  `evaluation_kit/` is adapted from
  [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and modified by the
  OpenVLHarness authors; files derived from it start with a comment saying so.
  `openvlharness/media.py` (image resize and base64 helpers) is also adapted
  from VLMEvalKit. License text:
  [LICENSES/VLMEvalKit-Apache-2.0.txt](LICENSES/VLMEvalKit-Apache-2.0.txt).
- **SAM 3 visualization helpers** (Copyright Meta Platforms, Inc. and
  affiliates, SAM License). `openvlharness/services/tools/sam3_segment/viz/`
  is adapted from `sam3/agent/helpers/` in
  [SAM 3](https://github.com/facebookresearch/sam3) and is distributed under
  the SAM License, not the MIT License. License text:
  [LICENSES/SAM-LICENSE.txt](LICENSES/SAM-LICENSE.txt).

## Models and services used by the default tool backends

Model weights and hosted services are not distributed with this repository.
Each is downloaded or called at run time and is governed by its own license or
terms of service, which you should review before use or redistribution.

- **SAM 3** (`facebook/sam3`, Meta) for visual grounding: SAM License, gated
  access.
- **Depth Anything 3** (`depth-anything/DA3NESTED-GIANT-LARGE`, the default
  depth and camera trajectory checkpoint): CC BY-NC 4.0, non-commercial use
  only.
- **PaddleOCR-VL** (`PaddlePaddle/PaddleOCR-VL-1.5`) for OCR: Apache License
  2.0.
- **Qwen3-VL-8B-Instruct** (`Qwen/Qwen3-VL-8B-Instruct`, Alibaba) as the
  default main and auxiliary model: Apache License 2.0.
- **GroundingDINO** (`IDEA-Research/grounding-dino-base`) for the optional
  GroundingDINO grounding backend: Apache License 2.0.
- **Other registered main models** (the Qwen3-VL family, OpenAI GPT models and
  Kimi K3), each under its provider's license or terms of service.
- **Serper** search, Lens and scrape APIs for the search tools, under the
  Serper terms of service.

## Datasets

Benchmark data is downloaded from the original sources or from the
[OpenVLHarness evaluation dataset repository](https://huggingface.co/datasets/uclanlp/OpenVLHarness-Evaluation-Datasets)
on first use. Every benchmark keeps the license of its original authors; see
the dataset card for per-split sources.
