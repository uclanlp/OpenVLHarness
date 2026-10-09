# PaddleOCR-VL service

The router's OCR wrapper preprocesses images (EXIF transpose, RGB and PNG)
and calls a dedicated vLLM `/v1/chat/completions` endpoint. This deployment
uses the Hugging Face `PaddlePaddle/PaddleOCR-VL-1.5` model through vLLM;
installing the PaddlePaddle runtime or a separate `paddleocr` CLI is unnecessary.

From the source checkout, create a separate environment:

```bash
python -m venv /path/to/envs/ocr
source /path/to/envs/ocr/bin/activate
bash openvlharness/services/tools/ocr/install.sh
CUDA_VISIBLE_DEVICES=0 bash openvlharness/services/tools/ocr/run_ocr_server.sh
```

The installer selects the tested vLLM 0.18.1 and Transformers 4.57.3. The
launcher uses the tested operational settings: 16384-token model length and
batch-token cap, GPU memory fraction 0.45, eager mode, prefix caching off, and
multimodal processor cache off. Extra vLLM options can be appended to the
launcher command. The installer has not yet been verified from a clean
environment.

| Launcher variable | Meaning |
| --- | --- |
| `OCR_MODEL_ID` | Served name; defaults to `PaddlePaddle/PaddleOCR-VL-1.5` |
| `OCR_MODEL_PATH` | Optional local checkpoint directory; defaults to `OCR_MODEL_ID` |
| `OCR_HOST` / `OCR_PORT` | Listen address; defaults to `127.0.0.1` / `8085` |

The module invocation is `python -m vllm.entrypoints.openai.api_server` with
the options in `run_ocr_server.sh`. Weights download on first use unless a local
checkpoint/cache is supplied. Use `HF_HOME` and vLLM cache directories on local
disk when possible. Wait for readiness, then check:

```bash
curl --fail http://127.0.0.1:8085/health
curl --fail http://127.0.0.1:8085/v1/models
```

Configure `services.tools.ocr.server_url` as the raw base URL; the paper YAML
uses `OCR_SERVER_URL`. If you change the served identity, set `OCR_MODEL_ID`
on the router host too. The wrapper's fixed request is prompt `OCR:`,
temperature 0, max_tokens 2048, repetition_penalty 1.05 and timeout 600 seconds;
these values are fixed in the wrapper.

The returned dictionary has `text` and `model_id`; it has no token boxes,
confidence values or device field. The capability stores the text in
`ocr_result` for local computation. See [router launch and tested checkpoint
revision](../README.md).
