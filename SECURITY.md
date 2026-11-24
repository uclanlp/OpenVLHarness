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
- **ToolSmith catalogs.** Generated tools are Python written by a language model. The
  tool router runs the catalog in a separate worker process, the pipeline (and the agent
  when its router lists no catalog) start one to read tool descriptions (without network
  when isolation is on), and the pipeline's Gate 1 runs each candidate once before
  admitting it. With isolation on (the default: `toolsmith.worker.isolation` for the
  router and the agent, `isolation` in the pipeline configuration), these processes run
  in Linux user, mount, PID and IPC namespaces under your user and see, read-only, the
  interpreter, every directory on the router's `sys.path` and `PYTHONPATH`, the catalog,
  the model caches, the files that granted variables name (a `*_FILE` key file, say),
  certificate bundles, a few settings from `/etc` and the host's `/sys`; with
  `write_files` and the local image host, the worker can also write the upload directory
  that host serves. They see no other host files or processes. The worker gets no
  credential-like environment variables that `toolsmith.worker.environment` does not
  grant, and Gate 1 runs with no credentials, no file writes and, with isolation on, no
  network. The router's worker shares the host's network unless
  `toolsmith.worker.network` is false, so a tool can reach the router and other services
  on the host's loopback. With isolation off, catalog code runs as your user, with your
  files and network, in every one of these processes. Run catalogs you did not write or
  review in a container or a virtual machine; see [Running generated tools
  safely](toolsmith/README.md#running-generated-tools-safely).
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
- With `openvlharness-eval`, the judge endpoint in `--judge-config` receives each
  question, its reference answer and the model's response. The paper uses GPT-4o
  for search and deep-research judging.
- The ToolSmith pipeline sends its generator, critic and analyst endpoint (the
  OpenAI API in `configs/toolsmith/pipeline.yaml`) each failed question with the
  agent's prediction, the reference answer, the tools the agent tried and their
  errors, the catalog's tool descriptions and the code of each candidate tool. It
  sends no images.
- The packaged ToolSmith tools send entity names, place names and titles to
  Wikidata, OpenStreetMap Nominatim, MusicBrainz and TMDB (with your
  `TMDB_API_KEY`). `visual_matches` uploads the query image as reverse image search
  does and sends its URL to Serper's Google Lens.
