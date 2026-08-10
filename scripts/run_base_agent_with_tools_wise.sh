#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Usage:
#   ./scripts/run_base_agent_with_tools_wise.sh [model_preset]
# Examples:
#   CONFIG=<agent.yaml> ./scripts/run_base_agent_with_tools_wise.sh      # required: a config with an image tool
#   ./scripts/run_base_agent_with_tools_wise.sh v5-oh                   # openvl-harness aligned
#   ./scripts/run_base_agent_with_tools_wise.sh gpt5                    # gpt-5 baseline for comparison
#   NUM_EXAMPLES=20 ./scripts/run_base_agent_with_tools_wise.sh         # smoke
#
# Dataset: WISE-Verified (arXiv 2503.07265, Niu et al. 2025)
#   1000 knowledge-intensive text-to-image prompts across 6 domains
#   (CULTURE / TIME / SPACE / BIOLOGY / PHYSICS / CHEMISTRY). Each prompt
#   requires world-knowledge inference (not stated in the prompt surface)
#   to produce a semantically correct image. No GT reference image; the
#   LLM judge sees (PROMPT, EXPLANATION, generated_image) and emits a
#   binary Score in {0, 1} per the paper's protocol. See
#   dataset/utils/judge_wise.py for the byte-exact judge prompt.
#
# Data source: a WISE repo checkout at $LMUData/WISE/data_verified/
#   Override via env var WISE_DATA_DIR to point to a different checkout.
#
# Paper compare: Qwen-Image-Agent (arXiv 2606.26907) Table 2 reports:
#   Qwen-Image-2.0 base: 0.7954 overall
#   Qwen-Image-Agent  : 0.9020 overall (+10.7 points via agentic search)
#
# Judge: only judge_wise (paper protocol, binary). No override.
#
# Presets (v5 nightly family, all use BaseAgentWithToolsAndCode):
#   v5              -> configs/rui-testing/base_agent_with_tools_v5_rui.yaml       (umbrella, DEFAULT)
#   v5-under        -> configs/rui-testing/base_agent_with_tools_v5_under_rui.yaml (understanding-only; NOT useful for gen)
#   v5-oh           -> configs/rui-testing/base_agent_with_tools_v5_oh_rui.yaml    (openvl-harness aligned)
#   gpt5            -> configs/rui-testing/base_agent_with_tools-gpt5_rui.yaml     (gpt-5 comparison)

# ⛔ No usable default: WISE is a GENERATION benchmark, and no config shipped in
# this repo can produce an image (configs/base_agent_with_tools.yaml restricts
# tool_list to Crop/Visual_Grounding/OCR/Python/Zoom, and agent/base.py drops every
# tool outside that list -- including Image_Generation_Tool). Defaulting to it would
# start happily and then score every sample zero for a missing image, which is worse
# than refusing to run. Pass CONFIG=<yaml> with an image tool, or a v5* preset if you
# have the local configs/rui-testing/ tree.
MODEL_PRESET="${1:-}"
DATASET="${DATASET:-WISE}"
# Judge config: prefer an in-repo default so the advertised bare invocation runs.
if [[ -z "${JUDGE_CONFIG:-}" ]]; then
  if [[ -f configs/rui-testing/judge_rui.yaml ]]; then
    JUDGE_CONFIG=configs/rui-testing/judge_rui.yaml
  else
    JUDGE_CONFIG=configs/judge.yaml
  fi
fi
NUM_PROC="${NUM_PROC:-4}"
SAVE_HTML="${SAVE_HTML:-10}"
SAVE_RAW_TRAJECTORY="${SAVE_RAW_TRAJECTORY:-1}"  # default ON; override with SAVE_RAW_TRAJECTORY= bash ... to disable
NUM_EXAMPLES="${NUM_EXAMPLES:-}"
MODE="${MODE:-all}"
SAMPLE_FILTER="${SAMPLE_FILTER:-all}"
SAVE_IMAGES="${SAVE_IMAGES:-files}"                # REQUIRED for WISE evaluate() to resolve generated images
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-default}"

if [[ -n "${CONFIG:-}" ]]; then
  AGENT_CONFIG="$CONFIG"
else
  case "$MODEL_PRESET" in
    v5|tools-v5|v5-all|v5_all)
      AGENT_CONFIG="configs/rui-testing/base_agent_with_tools_v5_rui.yaml"
      ;;
    v5-under|v5_under|tools-v5-under)
      AGENT_CONFIG="configs/rui-testing/base_agent_with_tools_v5_under_rui.yaml"
      ;;
    v5-under-qwen|v5_under_qwen|tools-v5-under-qwen)
      # v5-under with official Qwen3-VL-Instruct sampling (temp0.7/top_p0.8/
      # top_k20/rep1.0/presence1.5) — comparable to cf v8_instruct history.
      AGENT_CONFIG="configs/rui-testing/base_agent_with_tools_v5_under_qwen_rui.yaml"
      ;;
    v5-oh|v5_oh|tools-v5-oh)
      AGENT_CONFIG="configs/rui-testing/base_agent_with_tools_v5_oh_rui.yaml"
      ;;
    gpt5|gpt-5)
      AGENT_CONFIG="configs/rui-testing/base_agent_with_tools-gpt5_rui.yaml"
      ;;
    "")
      echo "ERROR: no agent config given." >&2
      echo "  WISE is a text-to-image benchmark: the agent must be able to PRODUCE an" >&2
      echo "  image, otherwise evaluate() finds none and scores every sample 0. That is" >&2
      echo "  why there is no default here." >&2
      echo "  Pass CONFIG=<agent.yaml> whose tool_list includes Image_Generation_Tool" >&2
      echo "  (or Image_Edit_Tool), or a v5* preset if you have configs/rui-testing/." >&2
      exit 1
      ;;
    *)
      echo "Unknown model preset: $MODEL_PRESET"
      echo "Supported presets: v5 | v5-under | v5-under-qwen | v5-oh | gpt5"
      echo "(these live in a local configs/rui-testing/ tree that is NOT in this repo;"
      echo " without it, pass CONFIG=<yaml> instead)"
      echo "(no default: WISE needs a generation-capable config -- see the note above)"
      echo "Or pass CONFIG=<path-to-yaml> to use a custom config."
      exit 1
      ;;
  esac
fi

# Fail fast with a clear message: the preset paths above point at a local
# configs/rui-testing/ tree that is not part of this repo, so a bare preset run
# would otherwise die later with a confusing error.
for _cfg_var in AGENT_CONFIG JUDGE_CONFIG; do
  _cfg="${!_cfg_var}"
  if [[ ! -f "$_cfg" ]]; then
    echo "ERROR: $_cfg_var not found: $_cfg" >&2
    echo "Pass CONFIG=<agent.yaml> / JUDGE_CONFIG=<judge.yaml> pointing at configs that exist" >&2
    echo "in your checkout, e.g. CONFIG=configs/base_agent_with_tools.yaml" >&2
    exit 1
  fi
done

# ⭐ Existence is not enough: the config must actually be able to generate. base.py
# filters self.tools down to tool_list, so a config without an image tool reaches
# evaluate() with nothing to score and reports 0 for every sample.
#
# ⛔ Check tool_list itself, not the whole file. A bare `grep Image_Generation_Tool`
# over the yaml has three failure modes, all of which bit an earlier revision:
#   - a mention in a COMMENT passes the guard;
#   - a config with NO tool_list at all fails it, even though base.py then keeps every
#     registered tool and the config IS image-capable;
#   - a commented-out tool name inside tool_list passes it.
_toolchk=$(python3 - "$AGENT_CONFIG" <<'PYCHK'
import sys, yaml
try:
    with open(sys.argv[1]) as f:
        cfg = yaml.safe_load(f) or {}
except Exception as e:
    print(f"unreadable: {e}"); raise SystemExit(0)
def find(node):
    if isinstance(node, dict):
        if "tool_list" in node:
            return node["tool_list"]
        for v in node.values():
            r = find(v)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = find(v)
            if r is not None:
                return r
    return None
tl = find(cfg)
if tl is None:
    print("no_tool_list")          # base.py keeps all tools -> image-capable
elif any(t in ("Image_Generation_Tool", "Image_Edit_Tool") for t in tl):
    print("ok")
else:
    print("missing:" + ",".join(map(str, tl)))
PYCHK
)
case "$_toolchk" in
  ok|no_tool_list) : ;;
  unreadable*)
    echo "WARNING: could not parse $AGENT_CONFIG to check tool_list ($_toolchk)" >&2 ;;
  missing:*)
    echo "ERROR: $AGENT_CONFIG has no image-producing tool in tool_list." >&2
    echo "  tool_list = ${_toolchk#missing:}" >&2
    echo "  WISE scores a missing generated image as 0, so this run would report ~0.00" >&2
    echo "  accuracy for reasons that have nothing to do with the model." >&2
    echo "  Add Image_Generation_Tool (or Image_Edit_Tool), and make sure the tool server" >&2
    echo "  exposes it." >&2
    exit 1 ;;
esac

# WISE only supports judge_wise (no override).
if [[ -n "${MMDR_JUDGE_PROMPT:-}" && "${MMDR_JUDGE_PROMPT}" != "judge_wise" ]]; then
  echo "WARNING: MMDR_JUDGE_PROMPT=${MMDR_JUDGE_PROMPT} overridden — WISE supports only judge_wise." >&2
fi
export MMDR_JUDGE_PROMPT="judge_wise"

# evaluate() resolves the generated image off disk, so anything other than
# "files" guarantees every sample scores 0. Warning is not enough for a mode
# that reaches evaluate() -- fail before burning a whole run.
if [[ "$SAVE_IMAGES" != "files" ]]; then
  if [[ "$MODE" == "all" || "$MODE" == "eval" ]]; then
    echo "ERROR: SAVE_IMAGES=${SAVE_IMAGES} but MODE=${MODE} runs evaluate(),"  >&2
    echo "       which resolves generated images from disk — every sample would score 0." >&2
    echo "       Set SAVE_IMAGES=files (or use MODE=infer)." >&2
    exit 1
  fi
  echo "WARNING: SAVE_IMAGES=${SAVE_IMAGES} — evaluate() would not resolve generated images." >&2
fi

# Verify WISE data is reachable. WISE runs from the cached $LMUData/WISE.tsv when
# present (load_data reads it directly); only the FIRST build needs merge.json.
# Source dir defaults under $LMUData (portable across checkouts) — override via
# WISE_DATA_DIR. No machine-specific absolute paths.
_LMU="${LMUData:-$HOME/LMUData}"
_WISE_DIR="${WISE_DATA_DIR:-$_LMU/WISE/data_verified}"
_WISE_TSV="$_LMU/WISE.tsv"
# ⛔ Do NOT exit when both caches are absent: WISEDataset._resolve_wise_data_dir()
# downloads merge.json from PKU-YuanGroup/WISE on a first clean run, and bailing
# out here would make that documented auto-download unreachable. Inform only.
if [[ ! -f "$_WISE_TSV" && ! -f "$_WISE_DIR/merge.json" ]]; then
  echo "NOTE: no cached $_WISE_TSV and no $_WISE_DIR/merge.json — the loader will" >&2
  echo "      auto-download from https://github.com/PKU-YuanGroup/WISE. Set WISE_DATA_DIR" >&2
  echo "      to an existing data_verified/ checkout to skip the download." >&2
fi

echo "Running BaseAgentWithToolsAndCode on WISE-Verified"
echo "  config:        $AGENT_CONFIG"
echo "  judge-config:  $JUDGE_CONFIG"
echo "  judge-prompt:  judge_wise (only supported)"
echo "  dataset:       $DATASET"
echo "  wise-data-dir: $_WISE_DIR"
echo "  num-proc:      $NUM_PROC"
echo "  save-html:     $SAVE_HTML"
echo "  save-images:   $SAVE_IMAGES (must be 'files' for WISE)"
echo "  num-examples:  ${NUM_EXAMPLES:-all}"
echo "  mode:          $MODE"

python run_benchmark.py \
  --config "$AGENT_CONFIG" \
  --judge-config "$JUDGE_CONFIG" \
  --data "$DATASET" \
  --num-proc "$NUM_PROC" \
  --incremental-json \
  --save-html "$SAVE_HTML" --save-images "$SAVE_IMAGES" \
  $( [[ "$SAVE_RAW_TRAJECTORY" == "1" ]] && echo --save-raw-trajectory ) \
  ${NUM_EXAMPLES:+--num-examples "$NUM_EXAMPLES"} \
  ${SHUFFLE:+--shuffle} \
  --seed "$SEED" --run-tag "$RUN_TAG" \
  ${RESUME:+--resume} ${RUN_DIR:+--run-dir "$RUN_DIR"} \
  --sample-filter "$SAMPLE_FILTER" --mode "$MODE"
