"""ToolSmith pipeline phases with stub models and synthetic rollouts, offline."""
import json
import os
import re
import shutil
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("sklearn", reason="the pipeline needs scikit-learn")
from openvlharness.services.models import IncompleteReply  # noqa: E402
from openvlharness.toolsmith.isolation import available  # noqa: E402
from openvlharness.toolsmith.pipeline import Pipeline  # noqa: E402
from openvlharness.toolsmith.pipeline.generation import describe_tools, source_of  # noqa: E402
from openvlharness.toolsmith.pipeline.mining import (Gap, UnparsedCritique, critique, parse_critique,  # noqa: E402
                                                     representatives)
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

    # A role whose reply a test empties, as a reasoning model's reply is when its budget runs out.
    empty = None
    # A role whose reply stops at its token limit after text that still parses.
    truncated = None

    def generate(self, request, system_prompt=None, max_tokens=None, require_complete=False):
        self.requests.append(request)
        if self.truncated and system_prompt.startswith(self.truncated) and require_complete:
            raise IncompleteReply("the reply stopped at its token limit (length); raise its max_tokens")
        if self.empty and system_prompt.startswith(self.empty):
            reply = ""
        elif system_prompt.startswith("You are an analyst"):
            reply = "LABEL: NEW_TOOL_NEEDED  WHY: no tool reads river lengths from an atlas"
        elif system_prompt.startswith("You classify"):
            reply = "ANSWER_TYPE: FACTUAL_LOOKUP\nANSWER_LENGTH: short_phrase\nEXTRACTION_HINT: the length in km"
        elif system_prompt.startswith("You are a Python tool author"):
            reply = json.dumps({"name": "river_length_lookup", "code": tool()})
        else:
            reply = json.dumps({"parametric_usage": 4, "description_abstractness": 4, "edge_case_handling": 3,
                                "task_alignment": 5, "critique": "fine"})
        if require_complete and not reply.strip():
            raise IncompleteReply("the reply holds no text")
        return reply


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
                ("", 'return {}.get("__builtins__")', "may name a hidden attribute"),
                ("", 'return "{0.__class__}".format(self)', "may name a hidden attribute"),
                ("", 'return {}.get("_" + "_builtins__")', "may name a hidden attribute"),
                ("", "match self.execute:\n            case object(__func__=f):\n                return f",
                 "match statements are not allowed"),
                ("", 'return ("{0." + "_" + "_class" + "_" + "_}").format(self)', "format is allowed only on a string"),
                ("", 'return str.format_map("{x}", {"x": river})', "format_map is allowed only on a string"),
                ("", 'return help("os")', "help is not allowed")):
            with self.subTest(body=body):
                self.assertIn(issue, "; ".join(scan(header + tool(body=body))))
        clean = tool(body='return json.dumps({"day": (datetime.datetime(2020, 1, 1) + timedelta(days=1)).isoformat(), '
                          '"words": re.findall(r"\\w+", river), "helper": self._words(river)})')
        clean = "import datetime\nimport json\nfrom datetime import timedelta\n" + clean.replace(
            "    def execute", "    def _words(self, text):\n        return text.split()\n\n    def execute")
        self.assertEqual(scan(clean), [])
        self.assertEqual(scan(tool(body='return "{} km, {river}".format(len(river), river=river)')), [])
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
        typed = tool().replace('{"type": "string"}', '{"type": "str"}')
        self.assertIn("not a valid JSON Schema", run_once(typed, 30, isolation=ISOLATION)[1])
        nullable = tool().replace('{"type": "string"}', '{"type": ["string", "null"]}')
        self.assertEqual(run_once(nullable, 30, isolation=ISOLATION)[0]["input_schema"]["properties"]["river"]["type"], ["string", "null"])

    def test_gate_one_feedback_is_the_candidates_own_error_the_same_on_every_run(self):
        # As the generator should read it: no loader wrapper, no temporary directory.
        failing = "PATTERNS = compile_patterns()  # a helper the candidate forgot\n" + tool()
        patterned = tool().replace('{"type": "string"}', '{"type": "string", "pattern": "^[a-z]+$"}', 1)
        for source, issue in ((failing, "execution failed: NameError: name 'compile_patterns' is not defined"),
                              (patterned, "execution failed: ValueError: candidate.py: input_schema uses pattern, "
                                          "which generated tools may not use: such keywords can make argument "
                                          "validation fetch documents or take exponential time"),
                              ("import re\n", "execution failed: ValueError: candidate.py must define exactly one "
                                              "GeneratedTool subclass"),
                              (tool().replace('"type": "object", "properties": {"river": {"type": "string"}}, '
                                              '"required": ["river"]', '"type": "array"'),
                               'execution failed: ValueError: candidate.py: input_schema must have "type": "object", '
                               'since the router passes the arguments to execute() as named arguments (it has '
                               '"array")')):
            with self.subTest(issue=issue):
                self.assertEqual([run_once(source, 30, isolation=ISOLATION)[1] for _ in range(2)], [issue] * 2)

    def test_gate_one_says_when_a_candidate_exits_at_import(self):
        for code, issue in (("0", "execution failed: candidate.py exits at import (SystemExit: 0)"),
                            ("", "execution failed: candidate.py exits at import (SystemExit)"),
                            ("3", "execution failed: SystemExit: 3")):
            source = f"import sys\nsys.exit({code})\n" + tool()
            with self.subTest(code=code):
                self.assertEqual(run_once(source, 30, isolation=ISOLATION), (None, issue))

    def test_gate_one_reads_a_tuple_schema_as_the_agent_does(self):
        source = (tool(body='assert isinstance(river, int), type(river).__name__; return {"river": river}')
                  .replace('{"type": "string"}', '{"type": ("integer", "null")}', 1)
                  .replace('["river"]', '("river",)', 1))
        metadata, issue = run_once(source, 30, isolation=ISOLATION)
        self.assertEqual((issue, metadata["input_schema"]["required"],
                          metadata["input_schema"]["properties"]["river"]["type"]), (None, ["river"], ["integer", "null"]))

    def test_gate_one_stops_a_candidate_whose_processes_together_hold_too_much_memory(self):
        from openvlharness.toolsmith.pipeline import verification

        # Three children of 100 MiB, each far under the per-process cap; the total is lowered to 256 MiB.
        body = ("import os\n        children = []\n        for _ in range(3):\n"
                "            read, write = os.pipe()\n            pid = os.fork()\n            if pid == 0:\n"
                "                block = bytearray(100 << 20)\n                for offset in range(0, len(block), 4096):\n"
                "                    block[offset] = 1\n                os.write(write, b'x')\n"
                "                __import__('time').sleep(20)\n                os._exit(0)\n"
                "            os.close(write)\n            os.read(read, 1)\n            children.append(pid)\n"
                "        __import__('time').sleep(20)\n        return river")
        started = time.monotonic()
        with patch.object(verification, "MEMORY_BYTES", 256 << 20):
            issue = run_once(tool(body=body), 60, isolation=ISOLATION)[1]
        self.assertRegex(issue, r"^execution used [\d.]+ GiB of memory in all its processes, more than 0\.25 GiB$")
        self.assertLess(time.monotonic() - started, 15)

    def test_an_interrupted_gate_one_run_is_killed(self):
        import subprocess

        from openvlharness.toolsmith.pipeline import verification

        # The run has a session of its own, so the terminal's Ctrl-C reaches only the pipeline.
        started, real = [], subprocess.Popen

        def recording(*args, **kwargs):
            started.append(real(*args, **kwargs))
            return started[-1]

        def interrupted(process, timeout, meter):
            time.sleep(1)
            raise KeyboardInterrupt
        self.addCleanup(lambda: [process.kill() for process in started if process.returncode is None])
        with patch.object(subprocess, "Popen", recording), patch.object(verification, "_wait", interrupted), \
                self.assertRaises(KeyboardInterrupt):
            run_once(tool(body="__import__('time').sleep(60)\n        return river"), 60, isolation=ISOLATION)
        self.assertEqual(len(started), 1)
        self.assertIsNotNone(started[0].returncode)  # killed and reaped, not left running

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

            def generate(self, request, system_prompt=None, max_tokens=None, require_complete=False):
                return self.replies.pop(0)
        metadata = {"tool_name": "t", "tool_description": "d", "input_schema": {}}
        scores = {"parametric_usage": 5, "description_abstractness": 5, "edge_case_handling": 0, "task_alignment": 5}
        self.assertFalse(review(Critic(json.dumps(scores)), "q", metadata, "src", mean=3.0, minimum=1)[0])
        passed, feedback, _ = review(Critic("not json", json.dumps({**scores, "edge_case_handling": 2})),
                                     "q", metadata, "src", mean=3.0, minimum=1)
        self.assertTrue(passed)
        self.assertIn('"edge_case_handling": 2.0', feedback)
        with self.assertRaisesRegex(RuntimeError, "held no valid scores"):  # the critic failed, not the candidate
            review(Critic("no", "still no"), "q", metadata, "src", mean=3.0, minimum=1)
        # A score off the 0-5 integer scale is no verdict: asked again, then the critic failed.
        axes = list(scores)
        for bad in ([10, 1, 1, 1], [5, 5, 5, -1], [4.5, 4, 4, 4], ["5", 5, 5, 5], [True, 5, 5, 5], [5, 5, 5, float("inf")]):
            with self.subTest(scores=bad), self.assertRaisesRegex(RuntimeError, "held no valid scores"):
                review(Critic(json.dumps(dict(zip(axes, bad))), json.dumps(dict(zip(axes, bad)))), "q", metadata,
                       "src", mean=3.0, minimum=1)
        passed, _, retried = review(Critic(json.dumps(dict(zip(axes, [10, 1, 1, 1]))),
                                           json.dumps(dict(zip(axes, [5, 4, 4, 4.0])))), "q", metadata, "src",
                                    mean=3.0, minimum=1)
        self.assertEqual((passed, list(retried.values())), (True, [5.0, 4.0, 4.0, 4.0]))


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
            def generate(self, request, system_prompt=None, max_tokens=None, require_complete=False):
                return "LABEL: SEARCH_NOISE WHY: the search worked but returned the wrong river"
        gap = critique(Gap("d:1", "q", "p", "g", False), Analyst(), 12800)
        self.assertEqual((gap.label, gap.reason), ("SEARCH_NOISE", "the search worked but returned the wrong river"))

    def test_a_self_critique_label_is_read_from_its_field_its_own_line_or_the_prompts_bullet_only(self):
        reason = "no tool reads river lengths"
        for reply, label in ((f"LABEL: NEW_TOOL_NEEDED  WHY: {reason}", "NEW_TOOL_NEEDED"),
                             (f"**LABEL:** NEW_TOOL_NEEDED  **WHY:** {reason}", "NEW_TOOL_NEEDED"),
                             (f"LABEL: new_tool_needed  WHY: {reason}", "NEW_TOOL_NEEDED"),
                             (f"Label: NEW_TOOL_NEEDED\nWhy: {reason}", "NEW_TOOL_NEEDED"),
                             (f"Label - new tool needed. Why: {reason}", "NEW_TOOL_NEEDED"),
                             (f"LABEL:\nNEW_TOOL_NEEDED\nWHY: {reason}", "NEW_TOOL_NEEDED"),
                             (f"Reply: LABEL: [ARGUMENT_MISTAKE] WHY: {reason}", "ARGUMENT_MISTAKE"),
                             (f"**Label**: `SEARCH_NOISE` — {reason}", "SEARCH_NOISE"),
                             (f"LABEL: SEARCH_NOISE WHY: {reason}", "SEARCH_NOISE"),
                             (f"LABEL: NEW_TOOL_NEEDED ({reason})", "NEW_TOOL_NEEDED"),
                             (f"Label = <ambiguous>; {reason}", "AMBIGUOUS"),
                             (f"NEW_TOOL_NEEDED\nWHY: {reason}", "NEW_TOOL_NEEDED"),
                             (f"- **NEW_TOOL_NEEDED**\nWHY: {reason}", "NEW_TOOL_NEEDED"),
                             (f"1. ARGUMENT_MISTAKE.\nWHY: {reason}", "ARGUMENT_MISTAKE"),
                             # The prompt's own bullet form, as the reply's first line and its only label line.
                             (f"NEW_TOOL_NEEDED: {reason}", "NEW_TOOL_NEEDED"),
                             (f"- NEW_TOOL_NEEDED: {reason}", "NEW_TOOL_NEEDED"),
                             (f"**ARGUMENT_MISTAKE**: {reason}", "ARGUMENT_MISTAKE"),
                             (f"**SEARCH_NOISE:** {reason}", "SEARCH_NOISE"),
                             (f"\n  * SEARCH_NOISE: {reason}\nThe pages were about another river.", "SEARCH_NOISE"),
                             (f"AMBIGUOUS: {reason}", "AMBIGUOUS"),
                             (f"NEW_TOOL_NEEDED: capability X is missing\nWHY: {reason}", "NEW_TOOL_NEEDED"),
                             # One preamble line that names no label, and Markdown headings for WHY.
                             (f"Here is my verdict:\nNEW_TOOL_NEEDED: {reason}", "NEW_TOOL_NEEDED"),
                             (f"Here is my answer.\n- SEARCH_NOISE: {reason}", "SEARCH_NOISE"),
                             (f"LABEL: NEW_TOOL_NEEDED\n### Why\n{reason}", "NEW_TOOL_NEEDED"),
                             (f"## Verdict\nARGUMENT_MISTAKE\n### Why\n{reason}", "ARGUMENT_MISTAKE"),
                             (f"NEW_TOOL_NEEDED\n**Why:** {reason}", "NEW_TOOL_NEEDED"),
                             (f"**Label:** AMBIGUOUS\n**Why**:\n{reason}", "AMBIGUOUS"),
                             # A short answer prefix before the field, and the label as a token before free text.
                             (f"Final answer: LABEL: NEW_TOOL_NEEDED  WHY: {reason}", "NEW_TOOL_NEEDED"),
                             (f"answer: label = SEARCH_NOISE\nWHY: {reason}", "SEARCH_NOISE"),
                             (f"OUTPUT: LABEL: ambiguous  WHY: {reason}", "AMBIGUOUS"),
                             (f"LABEL: NEW_TOOL_NEEDED because {reason}\nWHY: {reason}", "NEW_TOOL_NEEDED"),
                             (f"LABEL: SEARCH_NOISE (not NEW_TOOL_NEEDED)\nWHY: {reason}", "SEARCH_NOISE"),
                             (f"LABEL: NEW_TOOL_NEEDED  WHY: {reason}", "NEW_TOOL_NEEDED"),
                             # One JSON object with a label key.
                             (f'{{"label": "NEW_TOOL_NEEDED", "why": "{reason}"}}', "NEW_TOOL_NEEDED"),
                             (f'```json\n{{"LABEL": "search_noise", "Reason": "{reason}"}}\n```', "SEARCH_NOISE")):
            with self.subTest(reply=reply):
                self.assertEqual(parse_critique(reply), (label, reason))
        # "label:" or "label-" inside a sentence is prose, not a field.
        for reply, reason in (("LABEL: AMBIGUOUS\nWHY: a label: NEW_TOOL_NEEDED would not fit",
                               "a label: NEW_TOOL_NEEDED would not fit"),
                              ("NEW_TOOL_NEEDED\nWHY: a label-reading tool is missing", "a label-reading tool is missing"),
                              ("NEW_TOOL_NEEDED: no tool reads the label: it is in Japanese",
                               "no tool reads the label: it is in Japanese")):
            with self.subTest(reply=reply):
                self.assertEqual(parse_critique(reply)[1], reason)
        for reply in ("LABEL: NEW_TOOL_NEEDED because no tool reads river lengths",
                      "LABEL: NEW_TOOL_NEEDED since no tool reads river lengths",
                      "LABEL: NEW_TOOL_NEEDED as no tool reads river lengths",
                      "LABEL: NEW_TOOL_NEEDED due to no tool reading river lengths"):
            with self.subTest(reply=reply):
                self.assertEqual(parse_critique(reply), ("NEW_TOOL_NEEDED", reply.split(" ", 2)[2]))
        # Each reply of the three reviews' probes that is not one label in a field, on a line of its own or
        # in the prompt's bullet form.
        for reply in (
                "The agent failed.", "I cannot tell what went wrong.", "LABEL: OTHER WHY: something",
                "Search noise is unlikely; the agent called the wrong tool.",
                # Negations and other prose that names a label.
                "This is not a NEW_TOOL_NEEDED case: the search worked but the agent misread the page.",
                "This is not NEW_TOOL_NEEDED: the search found the right page and the agent misread it.",
                "I would not call it NEW_TOOL_NEEDED; the agent passed the wrong river name.",
                "I would not call this NEW_TOOL_NEEDED; the agent passed the wrong image.",
                "NEW_TOOL_NEEDED does not apply. The search returned noise.",
                "The failure is NEW_TOOL_NEEDED.\nWHY: no tool reads river lengths",
                "Verdict: NEW_TOOL_NEEDED is wrong; ARGUMENT_MISTAKE is right.",
                "This is not SEARCH_NOISE; it is NEW_TOOL_NEEDED because no tool reads tables.",
                "Search noise is not the issue. NEW_TOOL_NEEDED.\nWHY: no tool reads tables",
                "Label → NEW_TOOL_NEEDED (no tool reads river lengths)",
                "I would pick the label: SEARCH_NOISE.",
                # The bullet form anywhere but alone on the first line, or not quite it.
                "Intro one.\nIntro two.\nNEW_TOOL_NEEDED: no tool reads river lengths",
                "This is not SEARCH_NOISE:\nNEW_TOOL_NEEDED: no tool reads river lengths",
                "Here is my verdict:\nNEW_TOOL_NEEDED: no tool reads river lengths\nSEARCH_NOISE",
                "NEW_TOOL_NEEDED : no tool reads river lengths", "NEW_TOOL_NEEDED_V2: no tool reads river lengths",
                "NEW_TOOL_NEEDED: no tool reads river lengths\nSEARCH_NOISE",
                "NEW_TOOL_NEEDED\nSEARCH_NOISE: the results were off-topic",
                "- NEW_TOOL_NEEDED: capability X is missing\n- NEW_TOOL_NEEDED: no tool reads river lengths",
                # The word, not the label in capitals.
                "Ambiguous? No. The tools worked and returned the wrong river.",
                "Ambiguous at first sight, but no: the agent lacked a tool for river lengths.\nNEW_TOOL_NEEDED fits.",
                "Ambiguous? No: the question is clear.\nThe agent lacked a currency tool, so NEW_TOOL_NEEDED.",
                "Ambiguous wording aside, a converter was missing.\nWHY: no unit converter",
                "ambiguous", "new_tool_needed", "Ambiguous: the question has two readings",
                # A LABEL that is no label, or that names another one.
                "LABEL: TOOL_MISSING  WHY: unlike SEARCH_NOISE, the needed capability does not exist.",
                "Label: not NEW_TOOL_NEEDED (the arguments were wrong)",
                "LABEL: Search noise is unlikely; the agent called the wrong tool.",
                "LABEL: AMBIGUOUS? Hard to say.", "LABEL: NEW_TOOL_NEEDED_V2", "LABEL: NEW_TOOL_NEEDED/SEARCH_NOISE",
                # Prose that goes on about the label after it, not a reason.
                "LABEL: NEW_TOOL_NEEDED is not it", "LABEL: NEW_TOOL_NEEDED does not apply",
                "LABEL: SEARCH_NOISE is unlikely here", "LABEL: NEW_TOOL_NEEDED isn't right",
                "LABEL: NEW_TOOL_NEEDED seems wrong", "LABEL: NEW_TOOL_NEEDED would not help",
                "LABEL: NEW_TOOL_NEEDED was my first guess, but no", "LABEL: AMBIGUOUS not really",
                "LABEL: new tool needed because no tool reads river lengths",
                # A field after any other prefix is mid-sentence.
                "Verdict: LABEL: NEW_TOOL_NEEDED", "My final answer is label: NEW_TOOL_NEEDED",
                # JSON whose label is not exactly one label, or that has none.
                '{"label": "NEW_TOOL_NEEDED because no tool reads river lengths"}', '{"label": ["NEW_TOOL_NEEDED"]}',
                '{"label": "TOOL_MISSING", "why": "unlike SEARCH_NOISE"}', '{"verdict": "NEW_TOOL_NEEDED"}',
                '{"label": "NEW_TOOL_NEEDED", "Label": "SEARCH_NOISE"}', 'Here it is: {"label": "NEW_TOOL_NEEDED"}',
                # Several labels.
                "Label: NEW_TOOL_NEEDED? No - SEARCH_NOISE. WHY: results were off-topic",
                "LABEL: SEARCH_NOISE\nLABEL: NEW_TOOL_NEEDED",
                "NEW_TOOL_NEEDED\nSEARCH_NOISE",
                "- NEW_TOOL_NEEDED: capability X is missing\n- SEARCH_NOISE: tools worked, bad results",
                "- NEW_TOOL_NEEDED: capability X is missing\n- ARGUMENT_MISTAKE: right tool, wrong usage\n"
                "- SEARCH_NOISE: tools worked, bad results\n- AMBIGUOUS: question itself unanswerable"):
            with self.subTest(reply=reply), self.assertRaises(UnparsedCritique):
                parse_critique(reply)

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
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
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

    def decisions(self):
        path = self.root / "work" / "decisions.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]

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

    def test_an_admitted_module_appears_whole_or_not_at_all(self):
        from openvlharness.toolsmith.catalog import catalog_modules

        pipeline, source = self.pipeline(), tool()
        metadata = {"tool_name": "river_length_lookup", "tool_description": "Report the length of a river."}
        seen, real_fsync = [], os.fsync

        def fsync(descriptor):  # the moment the old code left an empty or partial module in the catalog
            seen.append([(path.name, path.read_text()) for path in catalog_modules(self.catalog)])
            real_fsync(descriptor)
        with patch("os.fsync", fsync):
            self.assertEqual(pipeline.admit(metadata, source, {})["decision"], "admitted")
        self.assertTrue(seen)
        self.assertNotIn("river_length_lookup.py", [name for name, _ in seen[0]])
        self.assertEqual((self.catalog / "river_length_lookup.py").read_text(), source)
        self.assertEqual(sorted(path.name for path in self.catalog.iterdir() if path.name.startswith(".")), [])
        # A run stopped while writing leaves neither the module nor its temporary file.
        stopped = {**metadata, "tool_name": "lake_depth_lookup", "tool_description": "Report a lake's depth."}

        def interrupted(descriptor):
            raise KeyboardInterrupt
        with patch("os.fsync", interrupted), self.assertRaises(KeyboardInterrupt):
            pipeline.admit(stopped, tool(name="lake_depth_lookup"), {})
        self.assertFalse((self.catalog / "lake_depth_lookup.py").exists())
        self.assertEqual(sorted(path.name for path in self.catalog.iterdir() if path.name.startswith(".")), [])

    def test_an_admitted_module_gets_the_mode_of_a_normally_created_file(self):
        metadata = {"tool_name": "river_length_lookup", "tool_description": "Report the length of a river."}
        for mask, mode in ((0o022, 0o644), (0o027, 0o640)):
            previous = os.umask(mask)
            try:
                path = self.catalog / "river_length_lookup.py"
                path.unlink(missing_ok=True)
                self.assertEqual(self.pipeline().admit(metadata, tool(), {})["decision"], "admitted")
            finally:
                os.umask(previous)
            with self.subTest(umask=oct(mask)):
                self.assertEqual(path.stat().st_mode & 0o777, mode)  # not mkstemp's 0600

    def test_without_hard_links_a_module_is_published_by_rename(self):
        import errno

        from openvlharness.toolsmith import pipeline as pipeline_module

        def unsupported(source, target):
            raise OSError(errno.EPERM, "Operation not permitted")
        pipeline = self.pipeline()
        metadata = {"tool_name": "river_length_lookup", "tool_description": "Report the length of a river."}
        with patch("os.link", unsupported), patch.object(pipeline_module, "_warned_no_hard_links", []), \
                self.assertLogs("openvlharness.toolsmith.pipeline", "WARNING") as logs:
            self.assertEqual(pipeline.admit(metadata, tool(), {})["decision"], "admitted")
            other = {"tool_name": "lake_depth_lookup", "tool_description": "Report a lake's depth."}
            self.assertEqual(pipeline.admit(other, tool(name="lake_depth_lookup"), {})["decision"], "admitted")
            # The name is still checked: an existing file is a collision, never replaced.
            existing = self.catalog / "bird_egg_lookup.py"
            existing.write_text("# someone else's\n")
            taken = {"tool_name": "bird_egg_lookup", "tool_description": "Report incubation days of a bird egg."}
            self.assertEqual(pipeline.admit(taken, tool(name="bird_egg_lookup"), {}),
                             {"decision": "duplicate", "tool": "bird_egg_lookup", "file": "bird_egg_lookup.py"})
        self.assertEqual(len(logs.output), 1)  # the race is logged once
        self.assertEqual((self.catalog / "river_length_lookup.py").read_text(), tool())
        self.assertEqual(existing.read_text(), "# someone else's\n")
        self.assertEqual([path.name for path in self.catalog.iterdir() if path.name.startswith(".")], [])

        def failing(source, target):
            raise OSError(errno.EIO, "Input/output error")
        with patch("os.link", failing), self.assertRaises(OSError):  # any other error still fails the gap
            pipeline.admit({"tool_name": "fjord_depth", "tool_description": "Report a fjord's depth."},
                           tool(name="fjord_depth"), {})
        self.assertFalse((self.catalog / "fjord_depth.py").exists())

    def test_mining_refuses_unclear_verdicts_and_reports_missing_trajectories(self):
        import pandas as pd

        from openvlharness.toolsmith.pipeline.mining import failures, judged_predictions

        rollout = Path(self.rollout)
        second = rollout / "Qwen3-VL-8B-Instruct_judge_judge_rs.xlsx"
        shutil.copy(rollout / "Qwen3-VL-8B-Instruct_judge_judge_vdr.xlsx", second)
        with self.assertRaisesRegex(ValueError, "several judged predictions .*judge_judge_rs.xlsx"):
            judged_predictions(rollout)
        second.unlink()
        summary = self.root / "MMSearch"  # writes a score summary under a judged-looking name
        summary.mkdir()
        pd.DataFrame([{"area": "news", "correct": 3, "total": 5}]).to_excel(
            summary / "Qwen3-VL-8B-Instruct_mmsearch_llm_judge_judge_vdr.xlsx", index=False)
        with self.assertRaisesRegex(ValueError, "has no hit column"):
            self.pipeline().run([str(summary)])
        self.judged_rollout("FVQA_train", "How long is the Danube?")  # its failed row c has no trajectory
        with self.assertLogs("openvlharness.toolsmith.pipeline.mining", "WARNING") as logs:
            gaps = list(failures(rollout))
        self.assertEqual([(gap.source, gap.tried_tools) for gap in gaps], [("FVQA_train:c", [])])
        self.assertIn("1 of 1 failed queries", logs.output[0])

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

    def test_a_run_directory_passed_for_its_dataset_directories_is_named(self):
        from openvlharness.toolsmith.pipeline.mining import judged_predictions

        with self.assertRaisesRegex(ValueError, "^" + re.escape(f"{self.root} is a run directory; pass its dataset "
                                                                 f"directories to --rollouts instead: {self.rollout}")):
            judged_predictions(self.root)
        run = self.root / "unjudged"
        (run / "FVQA_train").mkdir(parents=True)
        (run / "run_manifest.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "unjudged is a run directory; pass its dataset directories to "
                                                "--rollouts instead$"):
            judged_predictions(run)

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

                def generate(model, request, system_prompt=None, max_tokens=None, require_complete=False):
                    if why and system_prompt.startswith("You are an analyst"):
                        return why
                    return original(model, request, system_prompt, max_tokens, require_complete)
                if why:
                    (self.root / "work" / "critiques.jsonl").unlink()  # else the saved critique is reused
                with patch.object(StubModel, "generate", generate):
                    counts = self.pipeline().run([self.rollout], rebuild_gaps=True)
                self.assertEqual(sum(counts.values()), 1)  # the same source, decided again
                self.assertEqual(self.pipeline().run([self.rollout]), {})

    def test_an_empty_reply_leaves_its_step_undone_never_a_label_default_or_rejection(self):
        for role, expected in (("You classify", {"failed": 1}), ("You are a Python tool author", {"failed": 1}),
                               ("You are a senior Python code reviewer", {"failed": 1})):
            with self.subTest(role=role):
                shutil.rmtree(self.root / "work", ignore_errors=True)
                with patch.object(StubModel, "empty", role):
                    self.assertEqual(self.pipeline().run([self.rollout]), expected)  # undecided, not rejected
                self.assertEqual([entry["decision"] for entry in self.decisions()], ["failed"])
                self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})  # the next run retries it
                (self.catalog / "river_length_lookup.py").unlink()
        shutil.rmtree(self.root / "work", ignore_errors=True)
        with patch.object(StubModel, "empty", "You are an analyst"):
            self.assertEqual(self.pipeline().run([self.rollout]), {})
        self.assertEqual(json.loads((self.root / "work" / "mining.json").read_text())["failed_critiques"],
                         ["FVQA_train:a"])  # a failed critique, not an AMBIGUOUS label
        self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})

    def test_a_failed_self_critique_is_left_out_and_retried_alone(self):
        refused = self.judged_rollout("Infoseek_GDM_val", "refused question")
        asked, refusals = [], [RuntimeError("400 invalid_prompt")]
        original = StubModel.generate

        def generate(model, request, system_prompt=None, max_tokens=None, require_complete=False):
            if system_prompt.startswith("You are an analyst"):
                asked.append("refused" if "refused question" in request else "river")
                if "refused question" in request and refusals:
                    raise refusals.pop()
            return original(model, request, system_prompt, max_tokens, require_complete)
        with patch.object(StubModel, "generate", generate):
            with self.assertLogs("openvlharness.toolsmith.pipeline", "WARNING") as logs:
                self.assertEqual(self.pipeline().run([self.rollout, refused]), {"admitted": 1})
            self.assertTrue(any("1 self-critiques failed" in line for line in logs.output))
            record = json.loads((self.root / "work" / "mining.json").read_text())
            self.assertEqual(record["failed_critiques"], ["Infoseek_GDM_val:c"])
            self.assertEqual(sorted(asked), ["refused", "river"])
            asked.clear()
            self.assertEqual(sum(self.pipeline().run([self.rollout, refused]).values()), 1)  # the refused gap
            self.assertEqual(asked, ["refused"])  # the other critique came from critiques.jsonl
        sources = [json.loads(line)["source"] for line in (self.root / "work" / "gaps.jsonl").read_text().splitlines()]
        self.assertEqual(sorted(sources), ["FVQA_train:a", "Infoseek_GDM_val:c"])
        self.assertEqual(json.loads((self.root / "work" / "mining.json").read_text())["failed_critiques"], [])

    def test_each_critique_is_saved_as_it_completes_and_reused_by_any_later_mining(self):
        infoseek = self.judged_rollout("Infoseek_GDM_val", "Which country is this lake in?")
        asked, original = [], StubModel.generate

        def generate(model, request, system_prompt=None, max_tokens=None, require_complete=False):
            if system_prompt.startswith("You are an analyst"):
                asked.append(request)
            return original(model, request, system_prompt, max_tokens, require_complete)
        log = self.root / "work" / "critiques.jsonl"
        with patch.object(StubModel, "generate", generate):
            with patch("openvlharness.toolsmith.pipeline.mining.representatives", side_effect=RuntimeError("no GPU")):
                with self.assertRaisesRegex(RuntimeError, "no GPU"):  # stops after the critiques, while clustering
                    self.pipeline().run([self.rollout, infoseek])
            self.assertEqual(len(asked), 2)
            self.assertEqual(len(log.read_text().splitlines()), 2)
            self.assertFalse((self.root / "work" / "gaps.jsonl").exists())
            self.assertEqual(sum(self.pipeline().run([self.rollout, infoseek]).values()), 2)  # both gaps decided
            self.assertEqual(len(asked), 2)  # the rerun asked for nothing again
            with log.open("a") as stream:
                stream.write('{"analyst": {"alias": "stu')  # cut short by a stopped run
            third = self.judged_rollout("EncyclopedicVQA_val", "Which mountain range is this?")
            self.pipeline().run([self.rollout, infoseek, third], rebuild_gaps=True)
            self.assertEqual(len(asked), 3)  # a rebuild asks only about the new failure
            entries = [json.loads(line) for line in log.read_text().splitlines()[3:]]
            self.assertEqual([entry["label"] for entry in entries], ["NEW_TOOL_NEEDED"])
            self.config["models"] = {**self.config["models"], "analyst": {"alias": "another"}}
            self.pipeline().run([self.rollout, infoseek, third], rebuild_gaps=True)
            self.assertEqual(len(asked), 6)  # another analyst's critiques are its own

    def test_decisions_are_redone_after_their_settings_or_tool_change(self):
        self.config["critic_mean"] = 4.5  # the stub critic's mean is 4.0
        self.assertEqual(self.pipeline().run([self.rollout]), {"rejected": 1})
        self.assertEqual(self.pipeline().run([self.rollout]), {})  # same settings: reused
        self.config["critic_mean"] = 3.0
        self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})
        (self.catalog / "river_length_lookup.py").unlink()  # the admitted tool is gone: grown again
        self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})
        self.assertEqual(self.pipeline().run([self.rollout]), {})
        self.config["models"]["critic"] = {**self.config["models"]["critic"], "temperature": 0.7}
        self.assertEqual(sum(self.pipeline().run([self.rollout]).values()), 1)  # another critic: decided again

    def test_admission_takes_the_metadata_of_a_load_not_what_the_candidate_prints(self):
        (self.catalog / "phrase_translator.py").write_text(tool(name="phrase_translator",
                                                                description="Translate a phrase into French."))
        forged = tool(name="phrase_translator", body=(
            'print(json.dumps({"tool_name": "river_length_lookup", "tool_description": "Rivers.", '
            '"input_schema": {"type": "object"}}))\n        raise SystemExit(0)')).replace("import re", "import json\nimport re")
        computed = tool(name="river_length_lookup").replace('tool_name="river_length_lookup"',
                                                            'tool_name="river_" + "length_lookup"')
        reserved = tool().replace('"properties": {"river"', '"properties": {"num_results": {"type": "integer"}, "river"')
        for source, feedback in ((forged, "a tool named phrase_translator already exists"),
                                 (computed, "pass its tool_name as one string literal"),
                                 (reserved, "declares num_results, which the router reserves")):
            with self.subTest(feedback=feedback):
                shutil.rmtree(self.root / "work", ignore_errors=True)
                pipeline = self.pipeline()
                with patch.object(pipeline.generator, "_source", lambda request: source):
                    self.assertEqual(pipeline.run([self.rollout]), {"rejected": 1})
                decision = json.loads((self.root / "work" / "decisions.jsonl").read_text())
                self.assertIn(feedback, decision["attempts"][0])
                self.assertEqual(sorted(path.name for path in self.catalog.iterdir()), ["phrase_translator.py"])
        self.assertIn("phrase_translator", self.pipeline().descriptions)  # the catalog still loads

    def test_declared_name_is_the_single_literal_tool_name(self):
        from openvlharness.toolsmith.pipeline.verification import declared_name

        self.assertEqual(declared_name(tool()), "river_length_lookup")
        self.assertIsNone(declared_name(tool().replace('tool_name="river_length_lookup"', 'tool_name=NAME')))
        self.assertIsNone(declared_name(tool() + tool(name="other_lookup").split("import re", 1)[1]))
        self.assertIsNone(declared_name("def broken(:\n"))

    def test_the_command_line_rebuilds_gaps_only_when_asked(self):
        from openvlharness.toolsmith.pipeline import __main__ as cli

        calls = []

        class Recorder:
            critique_failures = 0

            def __init__(self, config, work_dir):
                self.work_dir, self.catalog, self.abandoned = Path(work_dir), "my_catalog", []

            def run(self, rollouts, *, rebuild_gaps=False, retry_failed=False):
                calls.append((rollouts, rebuild_gaps, retry_failed))
                if not rebuild_gaps:
                    raise ValueError("gaps.jsonl was mined from other inputs")
                self.work_dir.mkdir(exist_ok=True)
                (self.work_dir / "gaps.jsonl").write_text("{}\n" * 3)
                return {"admitted": 2}
        config = self.root / "pipeline.yaml"
        config.write_text("pipeline_config_version: paper_v1\n")
        argv = ["pipeline", "--config", str(config), "--rollouts", self.rollout, "--work-dir", str(self.root / "w")]
        with patch.object(cli, "Pipeline", Recorder), patch.object(sys, "argv", argv), \
                self.assertRaisesRegex(SystemExit, "error: gaps.jsonl was mined from other inputs"):
            cli.main()
        with patch.object(cli, "Pipeline", Recorder), patch.object(sys, "argv", argv + ["--rebuild-gaps"]), \
                patch("builtins.print") as printed:
            cli.main()  # nothing failed: exit status 0
        with patch.object(cli, "Pipeline", Recorder), patch.object(sys, "argv", argv + ["--rebuild-gaps",
                                                                                          "--retry-failed"]), \
                patch("builtins.print"):
            cli.main()
        self.assertEqual(calls, [([self.rollout], False, False), ([self.rollout], True, False),
                                 ([self.rollout], True, True)])
        gaps = self.root / "w" / "gaps.jsonl"
        printed.assert_called_once_with(f"Processed 2 of the 3 gaps in {gaps}; the other 1 had a decision already.\n"
                                        "  admitted: 2\nNew tools are in my_catalog; restart the router to serve them.")
        recorder = Recorder(None, self.root / "w")
        self.assertEqual(cli.summary({}, recorder), f"Nothing left to decide: all 3 gaps in {gaps} have a decision.")
        self.assertEqual(cli.summary({"failed": 3}, recorder),
                         f"Processed 3 of the 3 gaps in {gaps}.\n  failed: 3\n"
                         "The failed gaps have no decision yet; run the same command again to retry them (each is "
                         "retried until it has failed in 3 runs).")
        recorder.abandoned = ["FVQA_train:a"]
        self.assertEqual(cli.summary({}, recorder),
                         f"Nothing left to decide: all 3 gaps in {gaps} have a decision, or failed in 3 runs.\n"
                         "1 gaps failed in 3 runs with these settings and are set aside; pass --retry-failed to try them "
                         "again, also after a new API key (a change to the models' other settings, the gates' or "
                         "max_tokens retries them too).")
        recorder.abandoned = []
        gaps.write_text("")
        mining = self.root / "w" / "mining.json"
        for record, first in (({"labels": {}, "failed_critiques": ["FVQA_train:a", "FVQA_train:b"]},
                               "No gaps to decide yet: every self-critique failed, so no failed query has a label"),
                              ({"labels": {"SEARCH_NOISE": 4}, "failed_critiques": []},
                               "No gaps to decide: none of the 4 critiqued failed queries was labeled NEW_TOOL_NEEDED"),
                              ({"labels": {}, "failed_critiques": []}, "No gaps to decide: the rollouts hold no failed "
                                                                       "query")):
            mining.write_text(json.dumps(record))
            with self.subTest(record=record):
                self.assertTrue(cli.summary({}, recorder).startswith(f"{first} ({gaps} is empty)."))
        mining.write_text(json.dumps({"labels": {}, "failed_critiques": ["FVQA_train:a", "FVQA_train:b"],
                                      "abandoned_critiques": ["FVQA_train:c"]}))
        self.assertTrue(cli.summary({}, recorder).endswith(
            "\n2 self-critiques failed, so their queries are not mined yet; run the same command again to retry them."
            "\n1 self-critiques failed in 3 runs with these settings and are set aside; pass --retry-failed to try them "
            "again, also after a new API key (a change to the analyst's other settings or max_tokens retries them "
            "too)."))
        # A run in which a self-critique or a gap failed exits with status 3, after its summary: not 2, a usage error.
        for failure in ({"critique_failures": 2}, {"counts": {"failed": 1}}):
            class Failing(Recorder):
                critique_failures = failure.get("critique_failures", 0)

                def run(self, rollouts, *, rebuild_gaps=False, retry_failed=False):
                    self.work_dir.mkdir(exist_ok=True)
                    (self.work_dir / "gaps.jsonl").write_text("{}\n")
                    return failure.get("counts", {})
            with self.subTest(failure=failure), patch.object(cli, "Pipeline", Failing), \
                    patch.object(sys, "argv", argv), patch("builtins.print") as printed, \
                    self.assertRaises(SystemExit) as exited:
                cli.main()
            self.assertEqual(exited.exception.code, 3)
            printed.assert_called_once()
        with patch.object(sys, "argv", argv[:1]), patch("sys.stderr"), self.assertRaises(SystemExit) as usage:
            cli.main()
        self.assertEqual(usage.exception.code, 2)
        config.write_text("models:\n  generator:\n    base: ${TOOLSMITH_MODEL_BASE_URL}\n")
        with patch.dict(os.environ), patch.object(sys, "argv", argv), \
                self.assertRaisesRegex(SystemExit, r'TOOLSMITH_MODEL_BASE_URL \(models\.generator\.base\)\. '
                                                   r'Set it as in step 3 of "Grow your own catalog"'):
            os.environ.pop("TOOLSMITH_MODEL_BASE_URL", None)
            cli.main()

    def test_a_failing_gap_is_retried_until_it_failed_in_three_runs_unless_asked(self):
        from openvlharness.toolsmith.pipeline.generation import Generator

        def unavailable(generator, request):
            raise TimeoutError("model timed out")
        with patch.object(Generator, "_source", unavailable):
            for run in range(3):
                self.assertEqual(self.pipeline().run([self.rollout]), {"failed": 1})
            self.assertEqual([(entry["decision"], entry["error"]) for entry in self.decisions()],
                             [("failed", "TimeoutError: model timed out")] * 3)
            pipeline = self.pipeline()
            self.assertEqual(pipeline.run([self.rollout]), {})  # left alone after three failed runs
            self.assertEqual(pipeline.abandoned, ["FVQA_train:a"])
            self.config["critic_mean"] = 3.5  # other gate settings start afresh, failures included
            self.assertEqual(self.pipeline().run([self.rollout]), {"failed": 1})
            self.config["critic_mean"] = 3.0
        self.assertEqual(self.pipeline().run([self.rollout]), {})
        self.assertEqual(self.pipeline().run([self.rollout], retry_failed=True), {"admitted": 1})

    def test_a_new_max_tokens_or_endpoint_retries_what_failed_but_a_new_key_needs_retry_failed(self):
        from openvlharness.toolsmith.pipeline.generation import Generator

        self.config["models"] = {role: {"alias": "stub", "key": "sk-fixture-secret-1"}
                                 for role in ("generator", "critic", "analyst")}

        def incomplete(generator, request):
            raise IncompleteReply("the reply stopped at its token limit (length); raise its max_tokens")
        with patch.object(Generator, "_source", incomplete):
            for _ in range(3):
                self.assertEqual(self.pipeline().run([self.rollout]), {"failed": 1})
            self.assertEqual(self.pipeline().run([self.rollout]), {})  # set aside
            self.config["max_tokens"] = 32000  # the remedy the error names: retried at once
            self.assertEqual(self.pipeline().run([self.rollout]), {"failed": 1})
            self.pipeline().run([self.rollout])
            self.pipeline().run([self.rollout])
            self.assertEqual(self.pipeline().run([self.rollout]), {})  # three failures with this max_tokens too
            pipeline_before = self.pipeline()
            # The key is not in the digest, so a work directory cannot confirm a guessed one: a new key alone
            # leaves the gap set aside, and --retry-failed retries it.
            self.config["models"]["generator"]["key"] = "sk-fixture-secret-2"
            self.assertEqual(self.pipeline().failure_settings, pipeline_before.failure_settings)
            self.assertEqual(self.pipeline().critique_failure_settings, pipeline_before.critique_failure_settings)
            self.assertEqual(self.pipeline().run([self.rollout]), {})
            self.assertEqual(self.pipeline().run([self.rollout], retry_failed=True), {"failed": 1})
            self.config["models"]["generator"]["timeout"] = 900  # another endpoint or timeout retries it
            self.assertEqual(self.pipeline().run([self.rollout]), {"failed": 1})
        work = "".join(path.read_text() for path in (self.root / "work").iterdir())
        self.assertNotIn("sk-fixture-secret", work)
        self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})

    def test_credentials_in_an_endpoint_url_stay_out_of_the_failure_digest(self):
        def digests(base):
            self.config["models"] = {role: {"alias": "stub", "base": base, "key": "sk-fixture"}
                                     for role in ("generator", "critic", "analyst")}
            pipeline = self.pipeline()
            return pipeline.failure_settings, pipeline.critique_failure_settings
        first = digests("https://user:secret-1@models.example/v1?api_key=k1&token=t1&region=eu")
        self.assertEqual(digests("https://other:secret-2@models.example/v1?api_key=k2&token=t2&region=eu"), first)
        self.assertEqual(digests("https://models.example/v1?Key=k3&access_token=t3&region=eu"), first)
        for other in ("https://models.example/v1?region=us", "https://models.example/v2?region=eu",
                      "https://elsewhere.example/v1?region=eu"):  # another endpoint still retries
            with self.subTest(base=other):
                self.assertNotEqual(digests(other)[0], first[0])
                self.assertNotEqual(digests(other)[1], first[1])

    def test_a_new_max_tokens_asks_again_about_a_self_critique_that_failed_three_times(self):
        asked, original = [], StubModel.generate

        def generate(model, request, system_prompt=None, max_tokens=None, require_complete=False):
            if system_prompt.startswith("You are an analyst"):
                asked.append(max_tokens)
                if max_tokens < 32000:
                    raise IncompleteReply("the reply stopped at its token limit (length); raise its max_tokens")
            return original(model, request, system_prompt, max_tokens, require_complete)
        with patch.object(StubModel, "generate", generate):
            for _ in range(4):
                self.pipeline().run([self.rollout])
            self.assertEqual(asked, [12800] * 3)  # the fourth run set it aside
            self.config["max_tokens"] = 32000
            self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})
        self.assertEqual(asked, [12800] * 3 + [32000])

    def test_a_self_critique_without_a_label_is_retried_then_left_alone_unless_asked(self):
        asked, original = [], StubModel.generate

        def generate(model, request, system_prompt=None, max_tokens=None, require_complete=False):
            if system_prompt.startswith("You are an analyst"):
                asked.append(request)
                return "The agent should have tried harder."  # complete, but no label
            return original(model, request, system_prompt, max_tokens, require_complete)
        with patch.object(StubModel, "generate", generate):
            for run in range(3):
                pipeline = self.pipeline()
                self.assertEqual(pipeline.run([self.rollout]), {})
                self.assertEqual(pipeline.critique_failures, 1)
            record = json.loads((self.root / "work" / "mining.json").read_text())
            self.assertEqual((record["failed_critiques"], record["abandoned_critiques"], record["labels"]),
                             ([], ["FVQA_train:a"], {}))
            self.assertEqual(len(asked), 3)
            self.assertNotIn('"label"', (self.root / "work" / "critiques.jsonl").read_text())  # never a label
            self.assertEqual(self.pipeline().run([self.rollout]), {})
            self.assertEqual(len(asked), 3)  # not asked again, and not mined again
            self.pipeline().run([self.rollout], retry_failed=True)
            self.assertEqual(len(asked), 4)
        self.assertEqual(self.pipeline().run([self.rollout], retry_failed=True), {"admitted": 1})

    def test_a_decision_after_a_line_a_stopped_run_cut_short_is_kept(self):
        work = self.root / "work"
        work.mkdir()
        (work / "decisions.jsonl").write_text('{"source": "FVQA_train:a", "decision": "adm')  # cut short
        self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})
        lines = (work / "decisions.jsonl").read_text().split("\n")
        self.assertEqual((lines[0], json.loads(lines[1])["decision"], lines[2:]),
                         ('{"source": "FVQA_train:a", "decision": "adm', "admitted", [""]))
        self.assertEqual(self.pipeline().run([self.rollout]), {})  # read back: not decided again

    def test_line_separators_inside_the_work_files_do_not_split_their_lines(self):
        import pandas as pd

        question = "How long is the river\u2028in this map?\u0085(the blue one)\u2029"
        rollout = Path(self.rollout)
        pd.DataFrame([{"index": "a", "question": question, "answer": "2850 km", "prediction": "1000 km",
                       "image_path": "a.png", "hit": 0}]).to_excel(
            rollout / "Qwen3-VL-8B-Instruct_judge_judge_vdr.xlsx", index=False)
        asked, original = [], StubModel.generate

        def generate(model, request, system_prompt=None, max_tokens=None, require_complete=False):
            if system_prompt.startswith("You are an analyst"):
                asked.append(request)
                return "LABEL: NEW_TOOL_NEEDED  WHY: no tool reads\u2028river lengths"
            return original(model, request, system_prompt, max_tokens, require_complete)
        with patch.object(StubModel, "generate", generate):
            pipeline = self.pipeline()
            self.assertEqual(pipeline.run([self.rollout]), {"admitted": 1})
            self.assertEqual(self.pipeline().run([self.rollout]), {})  # gaps.jsonl and decisions.jsonl read back
            self.pipeline().run([self.rollout], rebuild_gaps=True)
            self.assertEqual(len(asked), 1)  # the saved critique was read back and reused
        from openvlharness.toolsmith.pipeline.__main__ import summary
        self.assertTrue(summary({}, pipeline).startswith("Nothing left to decide: all 1 gaps"))
        self.assertEqual(self.decisions()[0]["question"], question)

    def test_gate_one_sends_back_a_schema_pattern_as_feedback(self):
        original, replies = StubModel.generate, []

        def generate(model, request, system_prompt=None, max_tokens=None, require_complete=False):
            if system_prompt.startswith("You are a Python tool author") and not replies:
                replies.append(1)  # the first candidate validates its argument with a pattern
                return json.dumps({"name": "river_length_lookup", "code": tool().replace(
                    '{"type": "string"}', '{"type": "string", "pattern": "^([A-Za-z]+\\\\s?)+$"}', 1)})
            return original(model, request, system_prompt, max_tokens, require_complete)
        with patch.object(StubModel, "generate", generate):
            self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})  # revised without it
        (attempt,) = self.decisions()[0]["attempts"]
        self.assertTrue(attempt.startswith("Gate 1: execution failed: "), attempt)
        self.assertIn("uses pattern, which generated tools may not use", attempt)

    def test_a_critic_reply_cut_short_leaves_the_gap_undecided_even_when_it_parses(self):
        with patch.object(StubModel, "truncated", "You are a senior Python code reviewer"):
            self.assertEqual(self.pipeline().run([self.rollout]), {"failed": 1})  # never admitted on a cut reply
        self.assertFalse((self.catalog / "river_length_lookup.py").exists())
        self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})

    def test_max_tokens_and_complete_replies_reach_every_model_call(self):
        self.config["max_tokens"] = 777
        seen, original = {}, StubModel.generate

        roles = {"You are an analyst": "self-critique", "You classify": "answer shape",
                 "You are a Python tool author": "generator", "You are a senior Python code reviewer": "critic"}

        def generate(model, request, system_prompt=None, max_tokens=None, require_complete=False):
            role = next(role for prefix, role in roles.items() if system_prompt.startswith(prefix))
            seen.setdefault(role, set()).add((max_tokens, require_complete))
            return original(model, request, system_prompt, max_tokens, require_complete)
        with patch.object(StubModel, "generate", generate):
            self.assertEqual(self.pipeline().run([self.rollout]), {"admitted": 1})
        self.assertEqual(seen, dict.fromkeys(roles.values(), {(777, True)}))

    def test_each_critique_is_saved_before_the_others_finish(self):
        infoseek = self.judged_rollout("Infoseek_GDM_val", "Which country is this lake in?")
        log, original = self.root / "work" / "critiques.jsonl", StubModel.generate

        def generate(model, request, system_prompt=None, max_tokens=None, require_complete=False):
            if system_prompt.startswith("You are an analyst") and "lake" in request:
                # This critique ends only once the river's, which completes first, is on disk.
                for _ in range(100):
                    if log.exists() and "no tool reads river lengths" in log.read_text():
                        break
                    __import__("time").sleep(0.05)
                else:
                    raise TimeoutError("the first critique was not saved while the second ran")
            return original(model, request, system_prompt, max_tokens, require_complete)
        with patch.object(StubModel, "generate", generate):
            pipeline = self.pipeline()
            pipeline.run([self.rollout, infoseek])
        self.assertEqual(pipeline.critique_failures, 0)
        self.assertEqual(len([line for line in log.read_text().split("\n") if line]), 2)

    def test_the_pipeline_reads_catalog_metadata_without_importing_it(self):
        (self.catalog / "marker.py").write_text("import sys\n" + tool(
            name="atlas_marker", description="Translate a phrase into French.").replace(
            "class RiverLength", "sys.toolsmith_pipeline_marker = True\n\n\nclass RiverLength"))
        pipeline = self.pipeline()
        self.assertFalse(hasattr(sys, "toolsmith_pipeline_marker"))
        self.assertEqual(pipeline.descriptions["atlas_marker"], "Translate a phrase into French.")
        self.assertIn("- atlas_marker(river): Translate a phrase into French.", pipeline.generator.system)

    def test_the_preset_variables_the_pipeline_does_not_use_may_be_unset(self):
        agent = self.config["agent"]
        agent["services"]["client"]["base_url"] = "${TOOL_SERVER_URL}"
        agent["services"]["models"]["orchestrator"].update(base="${MODEL_BASE_URL}", key="${MODEL_API_KEY}")
        agent["toolsmith"]["catalog"] = "${CATALOG_DIR}"
        unset = {name: value for name, value in os.environ.items()
                 if name not in {"TOOL_SERVER_URL", "MODEL_BASE_URL", "MODEL_API_KEY", "CATALOG_DIR"}}
        with patch.dict(os.environ, unset, clear=True):
            with self.assertRaisesRegex(ValueError, r"variable required by agent configuration: "
                                                    r"CATALOG_DIR \(toolsmith\.catalog\)\. Set it"):
                self.pipeline()  # the toolsmith section's variables are required
            with patch.dict(os.environ, {"CATALOG_DIR": str(self.catalog)}):
                self.assertEqual(self.pipeline().catalog, self.catalog)
                agent["capabilities"]["tools"]["ocr"] = {"backend": "ocr"}
                agent["services"]["tools"]["ocr"] = {"server_url": "http://${OCR_HOST}:8000"}
                self.assertEqual(self.pipeline().catalog, self.catalog)  # also inside a longer value
                # Only service endpoints and keys get a stand-in: anything else unset stops the pipeline as before.
                for path, key in ((("models", "orchestrator"), "alias"), (("client",), "timeout_seconds")):
                    section = agent["services"]
                    for name in path:
                        section = section[name]
                    original, section[key] = section.get(key), "${UNSET_SETTING}"
                    with self.subTest(key=key), self.assertRaisesRegex(
                            ValueError, rf"UNSET_SETTING \(services\.{'.'.join(path)}\.{key}\)"):
                        self.pipeline()
                    if original is None:
                        del section[key]
                    else:
                        section[key] = original

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
