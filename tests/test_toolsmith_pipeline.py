"""ToolSmith pipeline phases with stub models and synthetic rollouts, offline."""
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="the pipeline needs scikit-learn")
from openvlharness.toolsmith.isolation import available  # noqa: E402
from openvlharness.toolsmith.pipeline import Pipeline  # noqa: E402
from openvlharness.toolsmith.pipeline.generation import describe_tools, source_of  # noqa: E402
from openvlharness.toolsmith.pipeline.mining import Gap, critique, representatives  # noqa: E402
from openvlharness.toolsmith.pipeline.reuse import closest  # noqa: E402
from openvlharness.toolsmith.pipeline.sandbox import dummy_arguments  # noqa: E402
from openvlharness.toolsmith.pipeline.verification import review, run_once, scan  # noqa: E402

# Gate 1 runs confined where this host allows it, as the pipeline runs it; elsewhere without isolation.
ISOLATION = available()[0]

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
        for header, body, issue in (
                ("import json\n", "return json.codecs.open(river).read()", "json.codecs is not allowed"),
                ("from re import __builtins__ as b\n", 'return b["open"](river).read()', "re does not export __builtins__"),
                ("from json import *\n", "return river", "json does not export *"),
                ("import json.decoder\n", "return river", "import json.decoder is not allowed"),
                ("import datetime\n", "return datetime.sys.modules", "datetime.sys is not allowed"),
                ("import json as j\n", "return j.scanner", "j.scanner is not allowed"),
                ("import json\n", "alias = json\n        return alias.loads(river)", "module json may only be used"),
                ("", "return (x for x in []).gi_frame.f_globals", "attribute gi_frame is not allowed"),
                ("", 'return {}.get("__builtins__")', "reaches hidden attributes"),
                ("", 'return "{0.__class__}".format(self)', "reaches hidden attributes"),
                ("", 'return help("os")', "help is not allowed")):
            with self.subTest(body=body):
                self.assertIn(issue, "; ".join(scan(header + tool(body=body))))
        clean = tool(body='return json.dumps({"day": (datetime.datetime(2020, 1, 1) + timedelta(days=1)).isoformat(), '
                          '"words": re.findall(r"\\w+", river), "helper": self._words(river)})')
        clean = "import datetime\nimport json\nfrom datetime import timedelta\n" + clean.replace(
            "    def execute", "    def _words(self, text):\n        return text.split()\n\n    def execute")
        self.assertEqual(scan(clean), [])
        self.assertIn("import from subprocess is not allowed", scan("from subprocess import run\n" + tool()))
        self.assertIn("expected one GeneratedTool subclass, found 0", scan("import re\n"))
        self.assertTrue(scan("def broken(:\n")[0].startswith("syntax error"))

    def test_sandbox_runs_candidates_once_with_dummy_input_and_limits(self):
        metadata, issue = run_once(tool(), 30, isolation=ISOLATION)
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
                self.assertIn(expected, run_once(tool(body=body), 30, isolation=ISOLATION)[1])
        self.assertEqual(run_once(tool(body="while True: pass"), 2, isolation=ISOLATION)[1], "execution exceeded 2 seconds")
        listed = tool(description="Rivers").replace('tool_description="Rivers"', 'tool_description=["Rivers"]')
        self.assertIn("tool_description must be strings", run_once(listed, 30, isolation=ISOLATION)[1])
        nullable = tool().replace('{"type": "string"}', '{"type": ["string", "null"]}')
        self.assertEqual(run_once(nullable, 30, isolation=ISOLATION)[0]["input_schema"]["properties"]["river"]["type"], ["string", "null"])

    def test_gate_one_keeps_only_the_end_of_a_chatty_candidates_output(self):
        from openvlharness.toolsmith.pipeline import verification

        chatty = tool(body='print("x" * 2_000_000)\n        print("y" * 2_000_000)\n        return {"river": river}')
        with patch.object(verification, "OUTPUT_BYTES", 1 << 16):
            self.assertEqual(run_once(chatty, 60, isolation=ISOLATION)[0]["tool_name"], "river_length_lookup")
            endless = tool(body='while True:\n            print("x" * 100_000)')
            self.assertEqual(run_once(endless, 3, isolation=ISOLATION)[1], "execution exceeded 3 seconds")
        with patch.object(verification, "OUTPUT_BYTES", 64):  # shorter than the metadata line itself
            self.assertIn("no metadata line within its last 64 bytes", run_once(tool(), 30, isolation=ISOLATION)[1])

    def test_dummy_arguments_cover_required_parameters_and_images(self):
        arguments = dummy_arguments({"properties": {"image_id": {"type": "string"}, "year": {"type": "integer"},
                                                    "labels": {"type": "array"}, "note": {"type": "string"}},
                                     "required": ["image_id", "year", "labels"]})
        self.assertEqual(sorted(arguments), ["image_base64", "labels", "year"])
        self.assertTrue(arguments["image_base64"].startswith("data:image/png;base64,"))
        self.assertEqual((arguments["year"], arguments["labels"]), (1, []))
        unions = dummy_arguments({"properties": {"name": {"type": ["string", "null"]},
                                                 "count": {"type": ["null", "integer"]},
                                                 "nothing": {"type": ["null"]}, "odd": {"type": ["mystery"]}},
                                  "required": ["name", "count", "nothing", "odd"]})
        self.assertEqual(unions, {"name": "test", "count": 1, "nothing": None, "odd": "test"})

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


    def test_examples_skip_failed_page_fetches(self):
        from openvlharness.toolsmith.catalog import load_catalog
        from openvlharness.toolsmith.pipeline.generation import EXAMPLES

        def block(url, summary):
            return (f"The useful information in {url} for user goal g as follows: \n\n"
                    f"Evidence in page: \ne\n\nSummary: \n{summary}\n\n")
        failed = block("https://a.example/page", "FETCH FAILED.\n\nReason: HTTP 403")
        good = block("https://b.example/page", "Emma Watson\nshe plays Hermione")

        class Client:
            def __init__(self, page):
                self.page = page

            def execute(self, tool_name, args, *, tool_call_id):
                result = ("Results: https://a.example/page https://b.example/page" if tool_name == "Text_Search_Tool"
                          else self.page)
                return {"ok": True, "result": result}
        for page, actress, stat in ((failed + "\n\n=======\n\n" + good, "Emma Watson", "Emma Watson"),
                                    (failed, {"error": "the cast pages do not name the performer"},
                                     {"error": "approximate total winnings not found on "
                                               "https://liquipedia.net/dota2/Miracle-"})):
            tools = load_catalog(EXAMPLES, client=Client(page))
            with self.subTest(page=page[:60]):
                self.assertEqual(tools["actress_by_character"].execute(character="Hermione Granger"), actress)
                self.assertEqual(tools["esports_player_stats"].execute(
                    player="Miracle-", game="dota2", statistic="approximate total winnings"), stat)


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

    def test_tool_activity_reads_the_failures_the_runtime_records(self):
        from openvlharness.toolsmith.pipeline.mining import _tool_activity

        chat = [{"role": "assistant", "tool_calls": [{"function": {"name": "Text_Search_Tool"}},
                                                     {"function": {"name": "Camera_Trajectory_Tool"}}]},
                {"role": "tool", "content": "ok"},
                {"role": "tool", "content": "tool_execution_error: Camera_Trajectory_Tool requires 2 images."},
                {"role": "user", "content": "Tool `Text_Search_Tool` (tool_call_id=c1) returned:\nresult: ok"}]
        self.assertEqual(_tool_activity(chat), (["Text_Search_Tool", "Camera_Trajectory_Tool"],
                                                ["tool_execution_error: Camera_Trajectory_Tool requires 2 images."]))
        responses = [{"type": "function_call", "call_id": "c1", "name": "Image_Search_Tool", "arguments": "{}"},
                     {"type": "function_call_output", "call_id": "c1",
                      "output": "invalid_tool_arguments: Unsupported task arguments: query."},
                     {"type": "function_call", "call_id": "c2", "name": "OCR_Tool", "arguments": "{}"},
                     {"type": "function_call_output", "call_id": "c2", "output": [
                         {"type": "input_text", "text": "Tool `OCR_Tool` (tool_call_id=c2) returned: text"}]}]
        self.assertEqual(_tool_activity(responses), (["Image_Search_Tool", "OCR_Tool"],
                                                     ["invalid_tool_arguments: Unsupported task arguments: query."]))
        envelope = [{"role": "tool", "content": '{"ok": false, "error": {"message": "no results"}}'}]
        self.assertEqual(_tool_activity(envelope), ([], ["no results"]))
        failures = [{"role": "tool", "content": f"tool_execution_error: failure {n}"} for n in range(5)]
        self.assertEqual(_tool_activity(failures)[1], [f"tool_execution_error: failure {n}" for n in (2, 3, 4)])

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
        self.config = {"pipeline_config_version": "paper_v1", "clusters": 4, "workers": 2, "isolation": ISOLATION,
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

    def judged_rollout(self, name, question, hit=0):
        import pandas as pd

        rollout = self.root / name
        (rollout / "full_trajectories").mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"index": "c", "question": question, "answer": "Austria", "prediction": "Germany",
                       "image_path": "c.png", "hit": hit}]).to_excel(
            rollout / "Qwen3-VL-8B-Instruct_judge_judge_vdr.xlsx", index=False)
        return str(rollout)

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

    def test_judged_predictions_are_read_in_every_prediction_format(self):
        pytest.importorskip("validators", reason="the evaluator's loader needs the eval extra")
        import pandas as pd
        from evaluation_kit.smp.file import dump

        from openvlharness.toolsmith.pipeline.mining import judged_predictions

        rollout = Path(self.rollout)
        rows = pd.read_excel(rollout / "Qwen3-VL-8B-Instruct_judge_judge_vdr.xlsx")
        (rollout / "Qwen3-VL-8B-Instruct_judge_judge_vdr.xlsx").unlink()
        for extension in ("tsv", "json"):
            with self.subTest(format=extension):
                for stale in rollout.glob("*_judge*"):
                    stale.unlink()
                dump(rows, str(rollout / f"Qwen3-VL-8B-Instruct_judge_judge_vdr.{extension}"))
                dump(pd.DataFrame([{"Overall": 0.5}]), str(rollout / f"Qwen3-VL-8B-Instruct_acc_judge_vdr.{extension}"))
                self.assertEqual(judged_predictions(rollout).name, f"Qwen3-VL-8B-Instruct_judge_judge_vdr.{extension}")
                for name in ("work", "catalog"):
                    shutil.rmtree(self.root / name, ignore_errors=True)
                self.catalog.mkdir()
                self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})
                gap = json.loads((self.root / "work" / "gaps.jsonl").read_text())
                self.assertEqual((gap["source"], gap["errors"]), ("FVQA_train:a", ["no results"]))
        for stale in rollout.glob("*_judge*"):
            stale.unlink()
        with self.assertRaisesRegex(ValueError, "no judged predictions"):
            judged_predictions(rollout)

    def test_the_gap_cache_is_reused_only_for_the_rollouts_and_settings_it_was_mined_from(self):
        self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})
        record = json.loads((self.root / "work" / "mining.json").read_text())
        self.assertEqual([(Path(entry["directory"]).name, entry["predictions"]) for entry in record["rollouts"]],
                         [("FVQA_train", "Qwen3-VL-8B-Instruct_judge_judge_vdr.xlsx")])
        self.assertEqual((record["encoder"], record["clusters"], record["analyst"]), ("BAAI/bge-m3", 4, {"alias": "stub"}))
        self.assertEqual(self.pipeline().run([self.rollout]), {})
        infoseek = self.judged_rollout("Infoseek_GDM_val", "Which country is this castle in?")
        for change, message in (({}, "rollout .*Infoseek_GDM_val is new"), ({"clusters": 3}, "clusters was 4, now 3"),
                                ({"encoder": "BAAI/bge-base-en-v1.5"}, "encoder was 'BAAI/bge-m3'")):
            with self.subTest(change=change):
                rollouts = [self.rollout, infoseek] if not change else [self.rollout]
                with patch.dict(self.config, change), self.assertRaisesRegex(ValueError, message + ".*--rebuild-gaps"):
                    self.pipeline().run(rollouts)
        trajectory = Path(self.rollout) / "full_trajectories" / "a.json"
        recorded = trajectory.read_text()
        for change in (lambda: trajectory.write_text(json.dumps({"messages": [
                {"role": "assistant", "tool_calls": [{"function": {"name": "OCR_Tool"}}]}]})), trajectory.unlink):
            change()
            with self.assertRaisesRegex(ValueError, "the trajectories of failed queries in .*FVQA_train changed"):
                self.pipeline().run([self.rollout])
        trajectory.write_text(recorded)
        self.assertEqual(self.pipeline().run([self.rollout]), {})
        self.judged_rollout("FVQA_train", "How long is the river in this map?")
        with self.assertRaisesRegex(ValueError, "the judged predictions of .*FVQA_train changed"):
            self.pipeline().run([self.rollout])
        counts = self.pipeline().run([self.rollout, infoseek], rebuild_gaps=True)
        self.assertEqual(sum(counts.values()), 2)
        sources = [json.loads(line)["source"] for line in (self.root / "work" / "gaps.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(sources), ["FVQA_train:c", "Infoseek_GDM_val:c"])
        self.assertEqual(self.pipeline().run([infoseek, self.rollout]), {})
        Path(self.rollout, "full_trajectories", "c.json").write_text(json.dumps({"messages": []}))  # was missing
        with self.assertRaisesRegex(ValueError, "the trajectories of failed queries in .*FVQA_train changed"):
            self.pipeline().run([self.rollout, infoseek])
        Path(self.rollout, "full_trajectories", "c.json").unlink()
        (self.root / "work" / "mining.json").unlink()
        with self.assertRaisesRegex(ValueError, "no mining.json"):
            self.pipeline().run([self.rollout, infoseek])

    def test_a_rebuilt_gap_is_decided_again_when_its_inputs_changed(self):
        self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})
        decision = json.loads((self.root / "work" / "decisions.jsonl").read_text())
        self.assertEqual(len(decision["gap_sha256"]), 64)
        self.assertEqual(self.pipeline().run([self.rollout], rebuild_gaps=True), {})  # same inputs: reused
        import pandas as pd
        pd.DataFrame([{"index": "a", "question": "How long is the Danube?", "answer": "2850 km", "prediction": "1000 km",
                       "image_path": "a.png", "hit": 0}]).to_excel(
            Path(self.rollout, "Qwen3-VL-8B-Instruct_judge_judge_vdr.xlsx"), index=False)
        for change in ("question", "reason"):
            with self.subTest(change=change):
                why = "LABEL: NEW_TOOL_NEEDED  WHY: no atlas lists the Danube" if change == "reason" else None
                original = StubModel.generate

                def generate(model, request, system_prompt=None, max_tokens=None):
                    if why and system_prompt.startswith("You are an analyst"):
                        return why
                    return original(model, request, system_prompt, max_tokens)
                with patch.object(StubModel, "generate", generate):
                    counts = self.pipeline().run([self.rollout], rebuild_gaps=True)
                self.assertEqual(sum(counts.values()), 1)  # the same source, decided again
                self.assertEqual(self.pipeline().run([self.rollout]), {})

    def test_the_command_line_rebuilds_gaps_only_when_asked(self):
        from openvlharness.toolsmith.pipeline import __main__ as cli

        calls = []

        class Recorder:
            def __init__(self, config, work_dir):
                pass

            def run(self, rollouts, *, rebuild_gaps=False):
                calls.append((rollouts, rebuild_gaps))
                if not rebuild_gaps:
                    raise ValueError("gaps.jsonl was mined from other inputs")
                return {"admitted": 2}
        config = self.root / "pipeline.yaml"
        config.write_text("pipeline_config_version: paper_v1\n")
        argv = ["pipeline", "--config", str(config), "--rollouts", self.rollout, "--work-dir", str(self.root / "w")]
        with patch.object(cli, "Pipeline", Recorder), patch.object(sys, "argv", argv), \
                self.assertRaisesRegex(SystemExit, "error: gaps.jsonl was mined from other inputs"):
            cli.main()
        with patch.object(cli, "Pipeline", Recorder), patch.object(sys, "argv", argv + ["--rebuild-gaps"]), \
                patch("builtins.print") as printed:
            cli.main()
        self.assertEqual(calls, [([self.rollout], False), ([self.rollout], True)])
        printed.assert_called_once_with("admitted: 2")

    def test_a_failing_model_call_leaves_the_gap_undecided(self):
        pipeline = self.pipeline()

        def unavailable(request):
            raise TimeoutError("model timed out")
        pipeline.generator._source = unavailable
        self.assertEqual(pipeline.run([self.rollout]), {"failed": 1})
        self.assertFalse((self.root / "work" / "decisions.jsonl").exists())
        self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})

    def test_the_pipeline_reads_catalog_metadata_without_importing_it(self):
        (self.catalog / "marker.py").write_text("import sys\n" + tool(
            name="atlas_marker", description="Translate a phrase into French.").replace(
            "class RiverLength", "sys.toolsmith_pipeline_marker = True\n\n\nclass RiverLength"))
        pipeline = self.pipeline()
        self.assertFalse(hasattr(sys, "toolsmith_pipeline_marker"))
        self.assertEqual(pipeline.descriptions["atlas_marker"], "Translate a phrase into French.")
        self.assertIn("- atlas_marker(river): Translate a phrase into French.", pipeline.generator.system)

    def test_settings_and_catalog_are_validated(self):
        self.config["clusterz"] = 3
        with self.assertRaisesRegex(ValueError, "Unknown pipeline settings"):
            self.pipeline()
        del self.config["clusterz"]
        for key, value, message in (("timeout_seconds", None, "timeout_seconds must be a number"),
                                    ("timeout_seconds", 0, "timeout_seconds must be a positive number"),
                                    ("workers", 0, "workers must be an integer of at least 1"),
                                    ("clusters", 2.5, "clusters must be an integer of at least 1"),
                                    ("retries", True, "retries must be an integer of at least 0"),
                                    ("reuse_threshold", "0.4", "reuse_threshold must be a number"),
                                    ("critic_mean", float("nan"), "critic_mean must be a number"),
                                    ("encoder", None, "encoder must name a sentence-transformers model"),
                                    ("isolation", "yes", "isolation must be true or false")):
            with self.subTest(key=key, value=value), patch.dict(self.config, {key: value}), \
                    self.assertRaisesRegex(ValueError, message):
                self.pipeline()
        del self.config["agent"]["toolsmith"]["catalog"]
        with self.assertRaisesRegex(ValueError, "Set toolsmith.catalog"):
            self.pipeline()
