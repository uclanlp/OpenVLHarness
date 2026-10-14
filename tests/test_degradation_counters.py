"""Silent summarizer fallbacks are counted and exposed on the router status endpoint."""
import unittest

from openvlharness.services import degradation
from openvlharness.services.tools.serper_text_search import tool as text_tool
from openvlharness.services.tools.serper_image_search import tool as image_tool


class _FailingEngine:
    def __init__(self, exc=None, output=None):
        self.exc, self.output = exc, output

    def generate(self, *args, **kwargs):
        if self.exc is not None:
            raise self.exc
        return self.output


class DegradationCounterTests(unittest.TestCase):
    def setUp(self):
        degradation.reset()

    def test_record_and_snapshot(self):
        degradation.record("summarize", tool="Text_Search_Tool", reason="TimeoutError")
        degradation.record("summarize", tool="Text_Search_Tool", reason="empty_output")
        snap = degradation.snapshot()
        self.assertEqual(snap["counts"]["summarize"], 2)
        self.assertEqual(snap["counts"]["summarize:TimeoutError"], 1)
        self.assertEqual(snap["last"]["summarize"]["reason"], "empty_output")
        degradation.record("summarize", tool="Text_Search_Tool")
        self.assertEqual(degradation.snapshot()["counts"],
                         {"summarize": 3, "summarize:TimeoutError": 1, "summarize:empty_output": 1})

    def _summarize(self, module, engine):
        if module is text_tool:
            tool = module.Text_Search_Tool(llm_engine=engine)
        else:
            tool = module.Image_Search_Tool(llm_engine=engine)
        tool._llm_summarize = True
        tool._multimodal_summarize = False
        tool.llm_engine = engine
        raw = "1. Example\n   Source: https://example.com/a\n"
        if module is text_tool:
            return tool._summarize_with_llm("q", "search", raw, keep_k=3)
        return tool._summarize_with_llm(raw)

    def test_text_search_counts_each_fallback_reason(self):
        for engine, reason in ((_FailingEngine(exc=TimeoutError("slow")), "TimeoutError"),
                               (_FailingEngine(output=""), "empty_output"),
                               (_FailingEngine(output="not json"), "unparseable_json"),
                               (_FailingEngine(output='{"summary": "x"}'), "kept_indices_invalid")):
            with self.subTest(reason=reason):
                with self.assertLogs(text_tool.logger, level="WARNING") as logs:
                    rendered, _ = self._summarize(text_tool, engine)
                self.assertIn("Example", rendered)
                self.assertTrue(any("SUMMARIZE_DEGRADED" in line for line in logs.output))
                self.assertEqual(degradation.snapshot()["counts"][f"summarize:{reason}"], 1)

    def test_webpage_visit_counts_extraction_fallbacks(self):
        from openvlharness.services.tools.serper_webpage_visit import tool as web_tool
        tool = web_tool.Webpage_Visit_Tool(llm_engine=_FailingEngine(exc=TimeoutError("slow")))
        with self.assertLogs(web_tool.logger, level="WARNING"):
            out = tool._extract_content("page text", "goal", "")
        self.assertEqual(out["evidence"], "page text")
        with self.assertLogs(web_tool.logger, level="WARNING"):
            out = web_tool.Webpage_Visit_Tool._parse_extraction_response("not json", "page text", "")
        self.assertEqual(out["evidence"], "not json")
        parse = web_tool.Webpage_Visit_Tool._parse_extraction_response
        # Non-string fields and non-object replies are schema violations (logged and counted).
        for reply, expected in (('{"evidence": 5, "summary": "x"}', {"evidence": "page text", "summary": "x"}),
                                ('["a"]', {"evidence": "page text", "summary": "desc"}),
                                ('{"evidence": "found", "summary": 7}', {"evidence": "found", "summary": "desc"})):
            with self.subTest(reply=reply):
                with self.assertLogs(web_tool.logger, level="WARNING"):
                    out = parse(reply, "page text", "desc")
                self.assertEqual(out, expected)
        # Empty or missing string fields are valid per the prompt: fall back silently, keep a summary.
        self.assertEqual(parse('{"evidence": "", "summary": "x"}', "page text", "desc"),
                         {"evidence": "page text", "summary": "x"})
        self.assertEqual(parse('{"evidence": "  "}', "page text", "desc"),
                         {"evidence": "page text", "summary": "desc"})
        self.assertEqual(parse('{"evidence": "found", "summary": "ok"}', "page text", "desc"),
                         {"evidence": "found", "summary": "ok"})
        counts = degradation.snapshot()["counts"]
        self.assertEqual(counts["extract:TimeoutError"], 1)
        self.assertEqual(counts["extract:unparseable_json"], 1)
        self.assertEqual(counts["extract:invalid_schema"], 3)

    def test_image_search_counts_fallbacks(self):
        with self.assertLogs(image_tool.logger, level="WARNING"):
            rendered, _ = self._summarize(image_tool, _FailingEngine(exc=RuntimeError("down")))
        self.assertIn("Example", rendered)
        self.assertEqual(degradation.snapshot()["counts"]["summarize:RuntimeError"], 1)


if __name__ == "__main__":
    unittest.main()
