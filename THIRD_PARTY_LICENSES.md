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
- **MM-BrowseComp** (the MM-BrowseComp authors, MIT License).
  `evaluation_kit/dataset/utils/mmbrowsecomp_judge.py` (judge prompt, API retry
  loop and response parser) is adapted from `src/eval.py` in
  [MM-BrowseComp](https://github.com/MMBrowseComp/MM-BrowseComp). The upstream
  repository declares the MIT License in its README but ships no license file;
  the MIT text naming the MM-BrowseComp authors as copyright holders is in
  [LICENSES/MM-BrowseComp-MIT.txt](LICENSES/MM-BrowseComp-MIT.txt).
- **Vision-DeepResearch** (Copyright (c) 2026 Wenxuan Huang, MIT License). The
  judge prompt in `evaluation_kit/dataset/utils/judge_vdr.py` and the auxiliary
  judge prompts in `evaluation_kit/dataset/browsecomp_vl.py` and
  `evaluation_kit/dataset/visbrowse_bench.py` are adapted from
  `rllm/rllm/rewards/deepresearch_reward.py` in
  [Vision-DeepResearch](https://github.com/Osilly/Vision-DeepResearch). License
  text:
  [LICENSES/Vision-DeepResearch-MIT.txt](LICENSES/Vision-DeepResearch-MIT.txt).
- **Humanity's Last Exam** (Copyright (c) 2025 centerforaisafety, MIT License).
  The judge prompt in `evaluation_kit/dataset/utils/hle_judge.py` is adapted
  from `hle_eval/run_judge_results.py` in
  [HLE](https://github.com/centerforaisafety/hle). License text:
  [LICENSES/HLE-MIT.txt](LICENSES/HLE-MIT.txt).

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

## Models, services and data used by ToolSmith

As above, none of these is distributed with this repository.

- **BGE-M3** (`BAAI/bge-m3`, BAAI), the sentence encoder for tool retrieval and
  for clustering failures in the pipeline: MIT License.
- **CLIP ViT-B/32** (OpenAI's weights, which open_clip downloads from
  `timm/vit_base_patch32_clip_224.openai`) for the packaged `image_similarity`
  tool: Apache License 2.0 on its Hugging Face card. OpenAI's model card,
  reproduced there, says any deployed use of the model, commercial or not, is
  out of scope.
- **BioCLIP** (`imageomics/bioclip`, Imageomics Institute) for the packaged
  `species_id` tool: MIT License.
- **Wikidata** (the Wikidata API and query service, Wikimedia Foundation),
  called by the packaged tools through `wikidata_query`: structured data under
  CC0 1.0, and use subject to the Wikimedia Terms of Use and User-Agent policy.
- **Nominatim** (`nominatim.openstreetmap.org`, OpenStreetMap Foundation),
  called through `gazetteer_lookup`: OpenStreetMap data under the Open Database
  License (© OpenStreetMap contributors, attribution required). The Nominatim
  Usage Policy allows at most one request per second and asks for a User-Agent
  that identifies the application.
- **MusicBrainz** (MetaBrainz Foundation), called by `media_catalog`: core data
  under CC0 and supplementary data under CC BY-NC-SA 3.0. The API allows on
  average one request per second from an IP address.
- **TMDB** (The Movie Database), called by `media_catalog` with your
  `TMDB_API_KEY`, under the TMDB API Terms of Use: free for non-commercial use
  that credits TMDB as the source of the data, with a license from TMDB needed
  for commercial use. This product uses the TMDB API but is not endorsed or
  certified by TMDB.
- **OpenAI API** for the pipeline's generator, critic and analyst
  (`gpt-5-2025-08-07` in `configs/toolsmith/pipeline.yaml`), under OpenAI's
  terms of use and usage policies.
- **ToolSmith training pools**
  ([uclanlp/ToolSmith-Datasets](https://huggingface.co/datasets/uclanlp/ToolSmith-Datasets):
  `FVQA_train`, `Infoseek_GDM_val` and `EncyclopedicVQA_val`). The packaging is
  under the MIT License; the questions, answers and images keep their original
  licenses: the FVQA and InfoSeek annotations are Apache-2.0, EncyclopedicVQA
  follows the terms of its release, and the images follow the terms of their
  sources (OVEN, iNaturalist 2021, Google Landmarks v2 and the FVQA image
  sources). See the dataset card.

## Datasets

Benchmark data is downloaded from the original sources or from the
[OpenVLHarness evaluation dataset repository](https://huggingface.co/datasets/uclanlp/OpenVLHarness-Evaluation-Datasets)
on first use. Every benchmark keeps the license of its original authors; see
the dataset card for per-split sources.
