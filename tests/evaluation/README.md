# Retained evaluator regression tests

These tests moved from the retired agent test directory. They cover run identity
and credential redaction, failed-row retries, atomic per-example records/resume,
HTML selection, full trajectories, input snapshots, and detection sidecars.
They import only retained evaluation/reporting code and the new profile resolver.
No old agent, memory, or tool infrastructure is needed.

Run from the research checkout with its evaluation dependencies installed:

```sh
python -m pytest tests/evaluation
```

Inference/reporting tests require the retained evaluator's dependencies,
including pandas, torch, validators, openpyxl, and xlsxwriter. They skip if the
retained checkout package or these optional dependencies are absent, allowing a
client-only wheel to run its own tests independently. No `sys.path` injection or
`sys.modules` package stubs are used. The synchronous scheduler is patched only
for each relevant test and restored afterward; reporting remains real.
