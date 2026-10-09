# Third-party notices

OpenVLHarness is released under the MIT License (see [LICENSE](LICENSE)). The
repository also contains or depends on the following third-party work.

## Code

- **VLMEvalKit** (OpenCompass, Apache License 2.0). The benchmark loaders and
  evaluators under `evaluation_kit/` are derived from
  [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) and keep its
  structure, dataset names and evaluation utilities. The Apache License 2.0
  text is available at <https://www.apache.org/licenses/LICENSE-2.0>.
  Modifications are noted in the affected files' docstrings and in the git
  history of this repository.

## Models and services used by the default tool backends

These are not distributed with this repository. Each is downloaded or called
at run time and is governed by its own license or terms of service, which you
should review before use or redistribution.

- **SAM 3** (Meta) for visual grounding, under Meta's SAM License.
- **Depth Anything 3** for depth estimation and camera trajectory.
- **PaddleOCR-VL** (PaddlePaddle) for OCR.
- **Qwen3-VL-8B-Instruct** (Alibaba) as the default main and auxiliary model.
- **Other registered main models** (the Qwen3-VL family, OpenAI GPT models and
  Kimi K3), each under its provider's license or terms of service.
- **Serper** search, Lens and scrape APIs for the search tools, under the
  Serper terms of service.

## Datasets

Benchmark data is downloaded from the original sources or from the
[OpenVLHarness evaluation dataset repository](https://huggingface.co/datasets/uclanlp/OpenVLHarness-Evaluation-Datasets)
on first use. Every benchmark keeps the license of its original authors; see
the dataset card for per-split sources.
