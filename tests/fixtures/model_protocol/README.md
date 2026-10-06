# Native model protocol fixtures

`old_release.json` contains offline outputs captured from immutable old release
`b98e5a540c4867eab20e07166c274e7ca8b8b9fe`. Its provenance records SHA-256 hashes
of the source files. The generator executed unchanged selected method ASTs and
the old native HTTP helpers with `requests.post` mocked; it did not contact a
model or import the refactored implementation.

Inputs cover both protocols' argument normalization, IDs, text/image encoding,
processed function outputs, and full request bodies. The tiny PNG is synthetic;
fixture URLs and API keys are deliberately nonfunctional. The generation script
is an external experiment artifact, not a test dependency. Checked-in tests use
only the installed `openvlharness` package and these static fixtures.

The fixtures preserve the old protocols' different malformed-argument handling,
image-result placement, and top-level schema sanitization. They do not assert
that those policies are interchangeable.

`auxiliary_old_release.json` independently captures the old `LLMEngine`'s SDK
arguments for greedy text and multimodal requests, including existing-system
message handling, task token budgets, and disabled Qwen thinking. Its generator
executes the unchanged old class with a mocked OpenAI constructor. Auxiliary
tests also exercise the actual new text search, reverse image search, webpage
extraction, and router coding paths against an in-process SDK/HTTP response.
