# Memory regression fixture

`memory_old_release.json` contains deterministic outputs produced by the original
memory code at commit `b98e5a540c4867eab20e07166c274e7ca8b8b9fe`. The provenance
object records SHA256s of the original image, URL, binding, and memory-orchestrator
modules. It is a behavior fixture, not a new implementation's expected-output
snapshot.

Inputs are three locally generated 2-by-2 colored PNGs, invented URLs, a small
float32 depth matrix, and synthetic query/argument strings. Download calls were
replaced with fixed byte or failure responses when collecting lazy-resolution
outputs. No network, credential, original runtime installation, or dataset file
is required to run the checked-in tests.

The `capability_boundary_reference_only` section records the old missing-image
argument and plural-image rewrite contract. The generic memory store resolves
individual references; capability boundary tests must check those argument
rewrites and ordering separately.
