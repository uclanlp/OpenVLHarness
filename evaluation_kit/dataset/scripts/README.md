# Dataset preparation helpers

These commands prepare and maintain the evaluation datasets. Run them from the
repository checkout with the evaluator dependencies installed;
`python -m evaluation_kit.dataset.scripts.<module> --help` lists each command's
inputs and options. Download the source datasets separately.

Set `LMUData` to your local dataset/output directory, or pass the command's explicit
output-directory argument. FSCD preparation requires source paths:

```bash
python -m evaluation_kit.dataset.scripts.prepare_fscd147_box \
  --fscd-root "$FSCD147_ROOT" --desc-json "$FSC147_DESCRIPTIONS" \
  --output-dir "$LMUData"

python -m evaluation_kit.dataset.scripts.prepare_fscdlvis_box \
  --fscd-root "$FSCD_LVIS_ROOT" --lvis-taxonomy "$LVIS_TAXONOMY" \
  --output-dir "$LMUData"
```

`FSCD147_ROOT` and `FSCD_LVIS_ROOT` point to the unpacked FSCD datasets.
`FSC147_DESCRIPTIONS` is the downloaded `FSC-147-D.json`; `LVIS_TAXONOMY` is the
LVIS `lvis_v1_val.json` category file. `prepare_fscdlvis_count` accepts the same
source arguments as `prepare_fscdlvis_box`.

| Commands | Purpose |
| --- | --- |
| `prepare_fsc147`, `prepare_fscd147_box`, `prepare_fscdlvis_box`, `prepare_fscdlvis_count` | Counting and box-grounding TSV preparation |
| `prepare_odinw`, `prepare_odinw_qwen25`, `prepare_odinw_qwen3` | ODinW dataset preparation and prompt variants |
| `prepare_lvis_federated`, `prepare_lvis_positive_only` | LVIS grounding subsets |
| `build_hf_release`, `backup_eval_datasets` | Release/export and backup utilities; inspect destinations before running |
| `patch_grounding_prompts` | Update local grounding TSV prompts; pass `--dry-run` to inspect first |

For prompt maintenance, supply `--lmu-data` or set `LMUData`. Service launch examples are documented separately under
[tool services](../../../openvlharness/services/tools/README.md).
