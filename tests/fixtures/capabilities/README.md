These compact, synthetic service replies were passed through the unchanged old
release at commit `b98e5a540c4867eab20e07166c274e7ca8b8b9fe`. The fixture records
the resulting evidence, image/URL registries, typed binding values, descriptors,
environment messages, argument resolution metadata, and detection sidecars.
Source file SHA-256 hashes and capture method are in `old_release.json`.

Golden capture used the old memory orchestrator and processors and the old
coding postprocess/executor, with network access denied. The small PNGs and
numerical arrays are synthetic. SAM3 RLE strings are opaque fixture payloads;
the test checks preservation and does not claim to decode masks or run models.
Tests import only the new installed package and do not regenerate expectations.

`public_schemas.json` is the approved nine-tool handoff snapshot, copied exactly.
OCR text storage and the DINO backend adapter are tested as explicit new
contracts rather than claimed old/new equality. Execution timeout tests exercise
Python loops in both main and worker threads, not blocking native calls.
