"""ToolSmith pipeline phases with stub models and synthetic rollouts, offline."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="the pipeline needs scikit-learn")
from openvlharness.toolsmith.pipeline import Pipeline  # noqa: E402
from openvlharness.toolsmith.pipeline.generation import describe_tools, source_of  # noqa: E402
from openvlharness.toolsmith.pipeline.mining import Gap, critique, representatives  # noqa: E402
from openvlharness.toolsmith.pipeline.reuse import closest  # noqa: E402
from openvlharness.toolsmith.pipeline.sandbox import dummy_arguments  # noqa: E402
from openvlharness.toolsmith.pipeline.verification import review, run_once, scan  # noqa: E402

TOOL = '''import re

from openvlharness.toolsmith import GeneratedTool


class RiverLength(GeneratedTool):
    def __init__(self):
        super().__init__(tool_name="{name}", tool_description="{description}", input_schema={{
            "type": "object", "properties": {{"river": {{"type": "string"}}}}, "required": ["river"]}})

    def execute(self, river=""):
        {body}
'''


def tool(name="river_length_lookup", description="Report the length of a river named in the question.",
         body='return {"river": river, "words": len(re.findall(r"\\\\w+", river))}'):
    return TOOL.format(name=name, description=description, body=body)


class StubModel:
    """Answers each pipeline prompt from its system prompt; records every request."""

    def __init__(self, config):
        self.config, self.requests = config, []

    def generate(self, request, system_prompt=None, max_tokens=None):
        self.requests.append(request)
        if system_prompt.startswith("You are an analyst"):
            return "LABEL: NEW_TOOL_NEEDED  WHY: no tool reads river lengths from an atlas"
        if system_prompt.startswith("You classify"):
            return "ANSWER_TYPE: FACTUAL_LOOKUP\nANSWER_LENGTH: short_phrase\nEXTRACTION_HINT: the length in km"
        if system_prompt.startswith("You are a Python tool author"):
            return json.dumps({"name": "river_length_lookup", "code": tool()})
        return json.dumps({"parametric_usage": 4, "description_abstractness": 4, "edge_case_handling": 3,
                           "task_alignment": 5, "critique": "fine"})


class GateTests(unittest.TestCase):
    def test_scan_accepts_clean_tools_and_rejects_dynamic_execution_and_io(self):
        self.assertEqual(scan(tool()), [])
        for body, issue in (('return eval(river)', "eval is not allowed"),
                            ('return open(river).read()', "open is not allowed"),
                            ('return river.__class__', "attribute __class__ is not allowed"),
                            ('return __builtins__', "__builtins__ is not allowed"),
                            ('return self.call_tool("river_length_lookup", river=river)', "execute calls the tool itself")):
            with self.subTest(body=body):
                self.assertIn(issue, scan(tool(body=body)))
        self.assertIn("import os is not allowed", scan("import os\n" + tool()))
        self.assertIn("import from subprocess is not allowed", scan("from subprocess import run\n" + tool()))
        self.assertIn("expected one GeneratedTool subclass, found 0", scan("import re\n"))
        self.assertTrue(scan("def broken(:\n")[0].startswith("syntax error"))

    def test_sandbox_runs_candidates_once_with_dummy_input_and_limits(self):
        metadata, issue = run_once(tool(), 30)
        self.assertIsNone(issue)
        self.assertEqual(metadata["tool_name"], "river_length_lookup")
        self.assertEqual(metadata["input_schema"]["required"], ["river"])
        for body, expected in (('raise ValueError("bad input")', "ValueError: bad input"),
                               ('return object()', "not JSON serializable"),
                               ('stream = __builtins__["open"]("x", "w"); stream.write("y"); return stream.flush()',
                                "File too large"),
                               ('return self.call_tool("Text_Search_Tool", query=river)',
                                "ToolCallError: Text_Search_Tool: tools are unavailable during the sanity run")):
            with self.subTest(body=body):
                self.assertIn(expected, run_once(tool(body=body), 30)[1])
        self.assertEqual(run_once(tool(body="while True: pass"), 2)[1], "execution exceeded 2 seconds")

    def test_dummy_arguments_cover_required_parameters_and_images(self):
        arguments = dummy_arguments({"properties": {"image_id": {"type": "string"}, "year": {"type": "integer"},
                                                    "labels": {"type": "array"}, "note": {"type": "string"}},
                                     "required": ["image_id", "year", "labels"]})
        self.assertEqual(sorted(arguments), ["image_base64", "labels", "year"])
        self.assertTrue(arguments["image_base64"].startswith("data:image/png;base64,"))
        self.assertEqual((arguments["year"], arguments["labels"]), (1, []))

    def test_review_applies_mean_and_minimum_thresholds(self):
        class Critic:
            def __init__(self, *replies):
                self.replies = list(replies)

            def generate(self, request, system_prompt=None, max_tokens=None):
                return self.replies.pop(0)
        metadata = {"tool_name": "t", "tool_description": "d", "input_schema": {}}
        scores = {"parametric_usage": 5, "description_abstractness": 5, "edge_case_handling": 0, "task_alignment": 5}
        self.assertFalse(review(Critic(json.dumps(scores)), "q", metadata, "src", mean=3.0, minimum=1)[0])
        passed, feedback, _ = review(Critic("not json", json.dumps({**scores, "edge_case_handling": 2})),
                                     "q", metadata, "src", mean=3.0, minimum=1)
        self.assertTrue(passed)
        self.assertIn('"edge_case_handling": 2.0', feedback)
        self.assertEqual(review(Critic("no", "still no"), "q", metadata, "src", mean=3.0, minimum=1)[1],
                         "the critic reply could not be parsed")


class PackagedToolTests(unittest.TestCase):
    HELPERS = {"wikidata_query", "gazetteer_lookup", "unit_compute", "media_catalog", "image_similarity",
               "species_id", "visual_matches"}

    def test_packaged_generated_tools_and_examples_pass_the_static_gate(self):
        package = Path(__file__).resolve().parents[1] / "openvlharness" / "toolsmith"
        paths = [path for directory in ("tools", "pipeline/examples") for path in (package / directory).glob("[!_]*.py")]
        self.assertTrue(self.HELPERS <= {path.stem for path in paths})
        for path in paths:
            if path.stem not in self.HELPERS:
                with self.subTest(tool=path.stem):
                    self.assertEqual(scan(path.read_text()), [])


class PhaseTests(unittest.TestCase):
    def test_reuse_finds_the_closest_description(self):
        descriptions = {"river": "Report the length of a river.", "bird": "Identify a bird species from an image."}
        name, similarity = closest("How long is the Danube river?", descriptions)
        self.assertEqual(name, "river")
        self.assertGreater(similarity, 0)
        self.assertEqual(closest("", descriptions), (None, 0.0))

    def test_generator_replies_yield_source_from_json_fences_or_python_blocks(self):
        self.assertEqual(source_of(json.dumps({"code": "x = 1"})), "x = 1\n")
        self.assertEqual(source_of('```json\n{"code": "x = 2"}\n```'), "x = 2\n")
        self.assertEqual(source_of("Here:\n```python\nx = 3\n```"), "x = 3\n")
        self.assertEqual(source_of('The tool:\n{"name": "t", "code": "def f():\\n    return {}"}'), "def f():\n    return {}\n")
        self.assertIsNone(source_of("no code"))

    def test_generated_tools_may_call_the_base_tools_the_generator_is_shown(self):
        from openvlharness.toolsmith.pipeline.generation import BASE_RESULTS
        from openvlharness.toolsmith.worker import CALLABLE_BASE_TOOLS

        self.assertEqual(set(BASE_RESULTS), set(CALLABLE_BASE_TOOLS))

    def test_tool_listing_uses_router_argument_names(self):
        base = {"Image_Search_Tool": {"description": "Reverse image search.",
                                      "parameters": {"properties": {"image_id": {"type": "string"}}}}}
        self.assertEqual(describe_tools(base, {}).split(": ", 1)[0], "- Image_Search_Tool(image_base64)")

    def test_self_critique_labels_and_reasons_are_parsed(self):
        class Analyst:
            def generate(self, request, system_prompt=None, max_tokens=None):
                return "LABEL: SEARCH_NOISE WHY: the search worked but returned the wrong river"
        gap = critique(Gap("d:1", "q", "p", "g", False), Analyst())
        self.assertEqual((gap.label, gap.reason), ("SEARCH_NOISE", "the search worked but returned the wrong river"))

    def test_representatives_pick_one_member_per_cluster(self):
        class Encoder:
            def __init__(self, name):
                pass

            def encode(self, texts, normalize_embeddings, convert_to_numpy):
                vectors = np.array([[text.count("river"), text.count("bird"), 0.1] for text in texts], dtype=float)
                return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        gaps = [Gap(f"d:{i}", question, "p", "g", False) for i, question in
                enumerate(["river river", "long river", "bird bird", "small bird"])]
        with patch.dict(sys.modules, {"sentence_transformers": types.SimpleNamespace(SentenceTransformer=Encoder)}):
            chosen = representatives(gaps, "stub", 2)
        self.assertEqual(sorted("river" in gap.question for gap in chosen), [False, True])
        self.assertEqual(representatives(gaps[:1], "stub", 2), gaps[:1])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        self.root = Path(tempfile.mkdtemp())
        rollout = self.root / "FVQA_train"
        (rollout / "full_trajectories").mkdir(parents=True)
        pd.DataFrame([
            {"index": "a", "question": "How long is the river in this map?", "answer": "2850 km",
             "prediction": "1000 km", "image_path": "a.png", "hit": 0},
            {"index": "b", "question": "Which bird is this?", "answer": "robin", "prediction": "robin",
             "image_path": "b.png", "hit": 1}]).to_excel(rollout / "Qwen3-VL-8B-Instruct_judge_judge_vdr.xlsx", index=False)
        (rollout / "full_trajectories" / "a.json").write_text(json.dumps({"messages": [
            {"role": "assistant", "tool_calls": [{"function": {"name": "Text_Search_Tool"}}]},
            {"role": "tool", "content": '{"ok": false, "error": {"message": "no results"}}'}]}))
        self.rollout = str(rollout)
        self.catalog = self.root / "catalog"
        self.catalog.mkdir()
        self.config = {"pipeline_config_version": "paper_v1", "clusters": 4, "workers": 2,
                       "models": dict.fromkeys(("generator", "critic", "analyst"), {"alias": "stub"}),
                       "agent": {"agent_name": "MainAgent", "agent_config_version": "paper_v1",
                                 "agent": {"model_service": "orchestrator"},
                                 "memory": {"modules": [{"name": "image"}]},
                                 "capabilities": {"registries": ["paper"], "tools": {"zoom": {"backend": "zoom_in_image_by_bbox"}}},
                                 "services": {"client": {"base_url": "http://router.invalid"},
                                              "server": {}, "tools": {"zoom_in_image_by_bbox": {}},
                                              "models": {"orchestrator": {"alias": "qwen3-vl-8b-instruct",
                                                                          "base": "http://model.invalid/v1"}}},
                                 "toolsmith": {"catalog": str(self.catalog)}}}

    def pipeline(self):
        with patch("openvlharness.toolsmith.pipeline.AuxiliaryModel", StubModel):
            return Pipeline(self.config, self.root / "work")

    def test_failed_rollouts_become_admitted_tools_and_reruns_resume(self):
        pipeline = self.pipeline()
        self.assertEqual(pipeline.run([self.rollout]), {"admitted": 1})
        self.assertTrue((self.catalog / "river_length_lookup.py").exists())
        gap = json.loads((self.root / "work" / "gaps.jsonl").read_text())
        self.assertEqual((gap["source"], gap["tried_tools"], gap["errors"], gap["image"]),
                         ("FVQA_train:a", ["Text_Search_Tool"], ["no results"], True))
        decision = json.loads((self.root / "work" / "decisions.jsonl").read_text())
        self.assertEqual((decision["decision"], decision["tool"], decision["attempts"]), ("admitted", "river_length_lookup", []))
        request = pipeline.models["generator"].requests[0]
        self.assertIn("Declare image_id in input_schema", request)
        self.assertIn("no tool reads river lengths", request)
        self.assertIn("- Zoom_In_Tool(image_base64, coords, padding)", pipeline.generator.system)
        self.assertEqual(self.pipeline().run([self.rollout]), {})

    def test_existing_tools_are_reused_and_rejected_candidates_get_feedback(self):
        (self.catalog / "atlas_river_length.py").write_text(tool(
            name="atlas_river_length", description="Report how long the river in this map is, using an atlas."))
        self.assertEqual(self.pipeline().run([self.rollout]), {"reused": 1})
        (self.catalog / "atlas_river_length.py").unlink()
        (self.root / "work" / "decisions.jsonl").unlink()
        pipeline = self.pipeline()
        pipeline.generator._source = lambda request: tool(body="return eval(river)")
        self.assertEqual(pipeline.run([self.rollout]), {"rejected": 1})
        decision = json.loads((self.root / "work" / "decisions.jsonl").read_text())
        self.assertEqual(decision["attempts"], ["Gate 1: eval is not allowed"] * 3)
        (self.root / "work" / "decisions.jsonl").unlink()
        pipeline.generator._source = lambda request: tool(name="../escape")
        self.assertEqual(pipeline.run([self.rollout]), {"rejected": 1})
        decision = json.loads((self.root / "work" / "decisions.jsonl").read_text())
        self.assertEqual(decision["attempts"][0], "Gate 1: tool_name '../escape' must be snake_case")
        self.assertFalse((self.root / "escape.py").exists())

    def test_a_catalog_file_is_never_overwritten_by_a_tool_with_another_public_name(self):
        existing = tool(name="phrase_translator", description="Translate a phrase into French.")
        path = self.catalog / "river_length_lookup.py"
        path.write_text(existing)
        pipeline = self.pipeline()
        self.assertEqual(pipeline.run([self.rollout]), {"rejected": 1})
        decision = json.loads((self.root / "work" / "decisions.jsonl").read_text())
        self.assertEqual(decision["attempts"],
                         ["Gate 1: a tool named river_length_lookup already exists; choose another name"] * 3)
        self.assertEqual(path.read_text(), existing)
        metadata = {"tool_name": "river_length_lookup", "tool_description": "Report the length of a river."}
        self.assertEqual(pipeline.admit(metadata, tool(), {}),
                         {"decision": "duplicate", "tool": "river_length_lookup", "file": "river_length_lookup.py"})
        self.assertEqual(path.read_text(), existing)
        self.assertNotIn("river_length_lookup", pipeline.descriptions)

    def test_a_failing_model_call_leaves_the_gap_undecided(self):
        pipeline = self.pipeline()

        def unavailable(request):
            raise TimeoutError("model timed out")
        pipeline.generator._source = unavailable
        self.assertEqual(pipeline.run([self.rollout]), {"failed": 1})
        self.assertFalse((self.root / "work" / "decisions.jsonl").exists())
        self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})

    def test_settings_and_catalog_are_validated(self):
        self.config["clusterz"] = 3
        with self.assertRaisesRegex(ValueError, "Unknown pipeline settings"):
            self.pipeline()
        del self.config["clusterz"]
        del self.config["agent"]["toolsmith"]["catalog"]
        with self.assertRaisesRegex(ValueError, "Set toolsmith.catalog"):
            self.pipeline()
