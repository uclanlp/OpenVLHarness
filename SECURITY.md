# Security

## Reporting a vulnerability

Report vulnerabilities through GitHub private vulnerability reporting: open the
repository's **Security** tab and choose **Report a vulnerability**. Please do not
open public issues for security problems.

## What runs where

- **Generated Python.** The coding tool runs model-written Python as your user, in a
  subprocess on the agent host, with network and filesystem access. The execution
  timeout is not a sandbox, and child processes that detach from the worker's
  process group can outlive it.
- **Tool router and GPU workers.** These services are unauthenticated. The router
  binds to `127.0.0.1` by default (`services.server.host`), as do the SAM 3, OCR and
  Depth Anything 3 workers. Keep them on loopback or a private network: anyone who
  can reach them can use your GPUs, model endpoints and Serper key. The agent
  executes Python returned by the router at `TOOL_SERVER_URL`, so point it only at a
  router you run.
- **Run directories.** Evaluation reads pickle caches from the run directory, so
  evaluate only run directories you trust.
- **Environment files.** `openvlharness-eval` loads `./.env` from the working
  directory, so run it only from directories you trust.

## Data that leaves your machine

- The main model endpoint receives question text, images and tool evidence. The
  auxiliary model receives search results, webpage text and the images being searched.
- Serper receives search queries and the URLs of visited webpages, which it fetches.
- Reverse image search uploads query images to your S3 or GCS bucket as
  `tool-uploads/<sha256>.<ext>` and sends Serper a signed URL valid for one hour.
  The uploaded objects are not deleted; add a lifecycle rule to the bucket.
- Search tools download result images from their source sites, and the agent
  downloads HTTP(S) images given in the question or in tool arguments.
