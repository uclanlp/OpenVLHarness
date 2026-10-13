# Search and webpage services

The router hosts the Serper wrappers; they contact external search/scrape APIs
and a configured auxiliary language model. The official paper configuration
uses Qwen3-VL-8B-Instruct for **multimodal** search summarization independently
of the main GPT or Qwen backbone.

Configure the auxiliary connection in `services.models.auxiliary` and select
it with each search wrapper's `model_service`. The paper YAML expands
`AUXILIARY_MODEL_BASE_URL` and `AUXILIARY_MODEL_API_KEY`. An alternative model
requires an explicit supported profile and connection. Search uses its own
prompt and context, not the orchestrator conversation.

Fixed paper settings are summarization on, multimodal mode on, 10 retrieved
results, at most 3 retained images/results, 4096 summary tokens, temperature 0,
top_p 1 and thinking disabled for Qwen. Text-to-image retrieval uses
`Text_Search_Tool(search_type="images")`; reverse search uses `Image_Search_Tool`.
Webpage extraction has a separate 16384-token cap and accepts at most five
URLs. These settings are fixed in code, not task arguments or environment
settings.

## Credentials and reverse-image upload

Set `SERPER_API_KEY` in the router process, or point `SERPER_API_KEY_FILE` at an
explicit external secret file. Restart the router when rotating a cached key.
Webpage visits use Serper scraping.

Reverse image search needs a URL that Serper Lens can fetch for the query
image. Configure one provider in the router environment:

| Provider | Environment settings |
| --- | --- |
| Local (no cloud account) | `IMAGE_UPLOAD_PROVIDER=local`, `IMAGE_UPLOAD_PUBLIC_BASE_URL`; optional `IMAGE_UPLOAD_LOCAL_DIR` |
| S3 | `IMAGE_UPLOAD_PROVIDER=s3`, `AWS_S3_BUCKET_NAME`, `AWS_S3_REGION`; credentials via `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or the provider credential chain |
| GCS | `IMAGE_UPLOAD_PROVIDER=gcs`, `GCS_BUCKET_NAME`, `GOOGLE_APPLICATION_CREDENTIALS`; optional `GCS_PROJECT_ID` |

The local provider writes query images to `IMAGE_UPLOAD_LOCAL_DIR` (default: a
directory under the system temporary folder) and serves them with
`python -m openvlharness.services.image_host --port 8111`. Publish that port at a
public address, for example with a tunnel such as
`cloudflared tunnel --url http://127.0.0.1:8111` (no account required), and set
`IMAGE_UPLOAD_PUBLIC_BASE_URL` to the printed URL. Images are reachable only
while the host and the tunnel run, so restart the router if the tunnel URL
changes. This is the quickest way to run; use S3 or GCS for long or highly
concurrent runs, for networks that do not allow tunnels, and when the uploaded
images must stay available afterwards. The paper results used S3.

The router extra includes boto3. Install `google-cloud-storage` separately
when using GCS. Cloud objects use content hashes under `tool-uploads/`; signed
URLs last one hour. Configure appropriate bucket access for the deployment.
Text-only web search does not need image-upload credentials.

Results keep the returned text and image payloads; an image-search response
with text only has null `matches`. The capability registers kept images and
rewrites webpage/image references in model evidence. Live search contents,
image availability and signed URLs can change between runs; use captured
responses for exact behavior comparisons.

See [router setup](../README.md) for launch and status commands.
