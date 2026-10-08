"""Parser contract tests for the generation-benchmark judges.

These benchmarks are used to make decisions about our own models, so the failure
mode that matters is not a crash -- it is a judge response that is malformed,
truncated or empty being silently recorded as a real score. Every case below
exists because that class of bug was found in review:

  * a syntactically valid but INCOMPLETE JSON object was accepted as a verdict,
    with missing score keys defaulting to 0, so ``{"rationale": "truncated"}``
    became four genuine zeros and the caller never retried;
  * parse-failure branches truncated the judge text to 500 chars, discarding
    exactly the tail needed to see that the completion had been cut off;
  * ``<Evaluation>...</evaluation>`` could never parse (XML tags are
    case-sensitive), so a documented recovery path silently always failed;
  * ``&copy;`` was treated as a legal XML entity, which it is not for
    ``xml.etree.ElementTree``.

The contract each judge parser must satisfy:
  1. a well-formed response parses, with the scores it actually contains;
  2. an incomplete or garbage response is reported as a PARSE FAILURE (so the
     caller retries) -- never as a zero-score verdict;
  3. the raw judge text is preserved untruncated on both paths.
"""

import importlib.util
import json
import pathlib

# The judge modules are deliberately dependency-free (re / json / typing only), but
# importing them through the package would execute openvlharness_eval.dataset.__init__,
# which pulls in pandas, validators and the rest of the eval stack. Loading them by
# path keeps this contract test runnable anywhere -- including a bare checkout -- so
# it actually gets run instead of being skipped for missing deps.
_UTILS = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _UTILS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_knowgen = _load("judge_knowgen")
_factip = _load("judge_factip")
_wise = _load("judge_wise")
_t2i = _load("judge_t2i_factualbench")
sg = _load("judge_searchgen")

parse_judge_knowgen_judgement = _knowgen.parse_judge_knowgen_judgement
parse_judge_factip_judgement = _factip.parse_judge_factip_judgement
parse_judge_wise_judgement = _wise.parse_judge_wise_judgement
parse_concept_rating = _t2i.parse_concept_rating
parse_task_score = _t2i.parse_task_score
parse_integration_score = _t2i.parse_integration_score
metric_for_tier = _t2i.metric_for_tier


# --------------------------------------------------------------------------
# KnowGen: 4 dimension scores + NA flag
# --------------------------------------------------------------------------

def test_knowgen_full_object_parses():
    out = parse_judge_knowgen_judgement(json.dumps({
        "rationale": "ok", "faithfulness": 1, "visual_correctness": 0.5,
        "text_accuracy": 0, "aesthetics": 1, "text_accuracy_na": False,
    }))
    assert out["verdict"], "a complete object must be accepted"
    assert out["faithfulness"] == 1.0
    assert out["visual_correctness"] == 0.5


def test_knowgen_partial_object_is_a_parse_failure():
    """The regression: this used to be cached as four genuine zeros."""
    out = parse_judge_knowgen_judgement('{"rationale": "truncated"}')
    assert not out["verdict"], (
        "an object missing every score key must be a parse failure, otherwise "
        "the caller sees a verdict, skips its retry and records zeros"
    )


def test_knowgen_one_missing_score_is_still_a_parse_failure():
    out = parse_judge_knowgen_judgement(json.dumps({
        "rationale": "ok", "faithfulness": 1, "visual_correctness": 1,
        "text_accuracy": 1,          # aesthetics missing
    }))
    assert not out["verdict"]


# --------------------------------------------------------------------------
# FactIP: 5 integer scores, weighted
# --------------------------------------------------------------------------

def test_factip_full_object_parses():
    out = parse_judge_factip_judgement(json.dumps({
        "rationale": "ok", "clarity": 8, "content_quality": 7,
        "aesthetics": 6, "text_relevance_ip": 9, "overall_score": 8,
    }))
    assert out["verdict"]
    assert out["clarity"] == 8


def test_factip_partial_object_is_a_parse_failure():
    out = parse_judge_factip_judgement('{"rationale": "cut off here"}')
    assert not out["verdict"]


# --------------------------------------------------------------------------
# WISE: binary score, and raw must survive intact
# --------------------------------------------------------------------------

def test_wise_binary_score_parses():
    assert parse_judge_wise_judgement("Score: 1")["score"] == 1.0
    assert parse_judge_wise_judgement("Score: 0")["score"] == 0.0


def test_wise_unparseable_reports_no_score():
    out = parse_judge_wise_judgement("I cannot evaluate this image.")
    assert out["score"] is None, "no score means the caller must retry"


def test_wise_keeps_the_whole_response():
    long_text = "x" * 4000 + "\nScore: 1"
    assert parse_judge_wise_judgement(long_text)["raw"] == long_text.strip()


def test_wise_keeps_the_whole_response_on_failure():
    """Truncating here removes the tail that shows the completion was cut off."""
    long_text = "y" * 4000
    assert parse_judge_wise_judgement(long_text)["raw"] == long_text


# --------------------------------------------------------------------------
# T2I-FactualBench: parsed_ok drives the retry, raw is never truncated
# --------------------------------------------------------------------------

def test_t2i_parsers_report_parsed_ok():
    assert parse_concept_rating("Total Rating: 3")["parsed_ok"] is True
    assert parse_task_score("Task Score: 1")["parsed_ok"] is True
    assert parse_integration_score("Integration Score: 0")["parsed_ok"] is True


def test_t2i_parse_failure_is_flagged_not_scored_zero():
    for parse in (parse_concept_rating, parse_task_score, parse_integration_score):
        out = parse("the judge rambled and never emitted a score")
        assert out["parsed_ok"] is False, (
            f"{parse.__name__} must flag the failure; a score of 0 with "
            "parsed_ok=True would be indistinguishable from a real zero"
        )


def test_t2i_parse_failure_keeps_full_raw():
    long_text = "z" * 4000
    for parse in (parse_concept_rating, parse_task_score, parse_integration_score):
        assert parse(long_text)["raw"] == long_text


def test_t2i_concept_mean_keeps_half_steps():
    """Ratings 2 and 3 average to 2.5; int(round(...)) used to make it 2.

    That moved the normalized concept contribution from 0.625 to 0.5 on every
    such sample.
    """
    assert metric_for_tier("SKCM", 2.5, None, None) == 0.625


# --------------------------------------------------------------------------
# SearchGen: XML recovery paths that used to be dead code
# --------------------------------------------------------------------------

def test_searchgen_normalises_mismatched_root_casing():
    """``<Evaluation>...</evaluation>`` is not well-formed XML on its own."""
    import xml.etree.ElementTree as ET

    fixed = sg._normalise_root_tag_case("<Evaluation><a>1</a></evaluation>")
    ET.fromstring(fixed)  # must not raise


def test_searchgen_leaves_correct_casing_alone():
    frag = "<evaluation><a>1</a></evaluation>"
    assert sg._normalise_root_tag_case(frag) == frag


def test_searchgen_escapes_non_predefined_entities():
    """ElementTree knows only amp/lt/gt/quot/apos -- not HTML entities."""
    repair = sg._repair_bare_xml_ampersands_in_fragment
    assert repair("A &copy; B") == "A &amp;copy; B"
    assert repair("A &nbsp; B") == "A &amp;nbsp; B"
    assert repair("A & B") == "A &amp; B"


def test_searchgen_keeps_legal_references():
    repair = sg._repair_bare_xml_ampersands_in_fragment
    for legal in ("A &amp; B", "A &lt; B", "A &gt; B", "A &quot; B",
                  "A &apos; B", "A &#169; B", "A &#x00A9; B"):
        assert repair(legal) == legal, f"{legal} is valid XML and must survive"

# --------------------------------------------------------------------------
# Out-of-protocol scores must be parse failures, not clamped into range
# --------------------------------------------------------------------------

def test_knowgen_rejects_scores_outside_the_allowed_set():
    """faithfulness: 9 used to be clipped to 1, and 0.7 accepted as-is.

    Both make the verdict look valid, so the caller stops retrying and the
    benchmark records a score the judge never gave.
    """
    for bad in (9, -1, 0.7, "high"):
        out = parse_judge_knowgen_judgement(json.dumps({
            "rationale": "x", "faithfulness": bad, "visual_correctness": 1,
            "text_accuracy": 1, "aesthetics": 1, "text_accuracy_na": False,
        }))
        assert not out["verdict"], f"faithfulness={bad!r} must be a parse failure"


def test_knowgen_accepts_exactly_the_three_legal_values():
    for good in (0, 0.5, 1):
        out = parse_judge_knowgen_judgement(json.dumps({
            "rationale": "x", "faithfulness": good, "visual_correctness": good,
            "text_accuracy": good, "aesthetics": good, "text_accuracy_na": False,
        }))
        assert out["verdict"], f"{good} is legal and must be accepted"
        assert out["faithfulness"] == float(good)


def test_factip_rejects_non_integer_and_out_of_range_scores():
    """7.6 used to round to 8 and 100 to clamp to 10."""
    for bad in (7.6, 100, -3, "eight"):
        out = parse_judge_factip_judgement(json.dumps({
            "rationale": "x", "clarity": bad, "content_quality": 5,
            "aesthetics": 5, "text_relevance_ip": 5, "overall_score": 5,
        }))
        assert not out["verdict"], f"clarity={bad!r} must be a parse failure"


def test_factip_accepts_integers_in_range():
    for good in (0, 5, 10, 8.0):     # 8.0 is an integer value, just typed as float
        out = parse_judge_factip_judgement(json.dumps({
            "rationale": "x", "clarity": good, "content_quality": 5,
            "aesthetics": 5, "text_relevance_ip": 5, "overall_score": 5,
        }))
        assert out["verdict"], f"clarity={good!r} is legal and must be accepted"
