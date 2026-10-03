"""Memory boundary regressions against outputs captured from the old release.

Tests import only the installed new package. Golden generation is an external
experiment step; neither the old runtime nor network access is required here.
"""
import concurrent.futures
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openvlharness.memory import images, urls
from openvlharness.memory.store import MemoryStore


GOLDEN = json.loads((Path(__file__).parent / "fixtures/memory_old_release.json").read_text())
RED, BLUE, GREEN = (GOLDEN["data_urls"][color] for color in ("red", "blue", "green"))


def question(*values):
    return [{"type": "image", "value": value} for value in values]


def plain(value):
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, set):
        return sorted(value)
    return value


def as_messages(message):
    return [] if message is None else [message]


def depth_value():
    recorded = GOLDEN["bindings"]["value"]
    return {name: np.array(value["values"], dtype=value["dtype"])
            if isinstance(value, dict) and "dtype" in value else value.copy()
            if isinstance(value, list) else value
            for name, value in recorded.items()}


def add_depth(memory):
    return memory.store_binding(
        prefix="depth", query="Test Depth", value=depth_value(),
        source_tool="Depth_Estimation_Tool", source_args=GOLDEN["bindings"]["args"],
        descriptions=GOLDEN["bindings"]["descriptions"],
    )


class MemoryBoundaryTests(unittest.TestCase):
    def setUp(self):
        # An accidental external lookup must fail loudly, never contact a server.
        self.network_guard = patch.object(images, "_fetch_image_data_url",
                                          side_effect=AssertionError("unexpected network lookup"))
        self.network_guard.start()
        self.addCleanup(self.network_guard.stop)

    def assert_image_state(self, actual, expected):
        # Snapshot bookkeeping is checked separately through emission behavior.
        actual = plain(actual)
        actual.pop("last_snapshot", None)
        expected = {key: value for key, value in expected.items() if key != "last_snapshot"}
        self.assertEqual(actual, expected)

    def test_m01_input_order_and_initial_environment_match_old(self):
        memory = MemoryStore()
        message = memory.start(question(RED, BLUE))
        self.assertEqual(as_messages(message), GOLDEN["input_order"]["initial_messages"])
        self.assertEqual(plain(memory.state.images), GOLDEN["input_order"]["images"])
        self.assertEqual(memory.resolve_image("input_image_2"), (BLUE, "input_image_2", None))

    def test_m01_start_clears_images_bindings_urls_and_counters(self):
        memory = MemoryStore()
        memory.start(question(RED))
        memory.register_image(GREEN, provenance="tool_output:Zoom_In_Tool")
        memory.register_url("https://example.com/old", provenance="Text_Search_Tool")
        add_depth(memory)
        old_state = memory.state
        memory.start(question(BLUE))
        self.assertIsNot(memory.state, old_state)
        self.assertEqual(list(memory.state.images.registry), ["input_image_1"])
        self.assertEqual(memory.resolve_image("input_image_1")[0], BLUE)
        self.assertEqual(memory.state.images.tool_generated_counter, 0)
        self.assertEqual(memory.state.bindings.bindings, {})
        self.assertEqual(memory.state.bindings.binding_counters, {})
        self.assertEqual(memory.describe_bindings(), {})
        self.assertEqual(memory.state.urls.webpage_registry, {})
        self.assertEqual(memory.state.urls.webpage_counter, 0)
        self.assertEqual(memory.register_image(GREEN), "tool_generated_image_1")
        self.assertEqual(add_depth(memory), "depth_test_depth")

    def test_m01_concurrent_question_stores_are_isolated(self):
        def run(data):
            memory = MemoryStore()
            memory.start(question(data))
            image = memory.register_image(GREEN)
            binding = add_depth(memory)
            url = memory.register_url("https://example.com/one")
            return memory, image, binding, url
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            left, right = list(pool.map(run, [RED, BLUE]))
        self.assertEqual(left[1:], right[1:])
        self.assertIsNot(left[0].state, right[0].state)
        left[0].state.bindings.bindings["depth_test_depth"]["depth_meters"][0, 0] = 99
        self.assertEqual(right[0].state.bindings.bindings["depth_test_depth"]["depth_meters"][0, 0], 0)
        self.assertEqual(left[0].resolve_image("input_image_1")[0], RED)
        self.assertEqual(right[0].resolve_image("input_image_1")[0], BLUE)

    def test_m02_duplicate_inputs_keep_enumeration_aliases(self):
        memory = MemoryStore()
        memory.start(question(RED, RED, BLUE))
        self.assert_image_state(memory.state.images, GOLDEN["duplicate_inputs"])
        self.assertEqual(list(memory.state.images.registry), ["input_image_1", "input_image_3"])
        self.assertEqual(memory.resolve_image("image_2"), (RED, "input_image_1", None))

    def test_m02_exact_data_url_dedup_does_not_pixel_deduplicate(self):
        memory = MemoryStore()
        memory.start(question(RED))
        values = [RED, GREEN, GREEN, "  " + RED + "  ", RED.replace("image/png", "image/x-png")]
        refs = [memory.register_image(value, provenance="tool_output:Zoom_In_Tool") for value in values]
        self.assertEqual(refs, GOLDEN["duplicate_outputs"]["refs"])
        self.assert_image_state(memory.state.images, GOLDEN["duplicate_outputs"]["images"])

    def test_m02_downloaded_duplicate_preserves_counter_gap(self):
        memory = MemoryStore()
        memory.start(question(RED))
        with patch.object(images, "_fetch_image_data_url", return_value=RED) as fetch:
            result = memory.resolve_image("https://fixture.invalid/same.png")
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(plain(result), GOLDEN["download_dedup_counter_gap"]["resolution"])
        ref = memory.register_image(GREEN, provenance="tool_output:Zoom_In_Tool")
        self.assertEqual([ref], GOLDEN["download_dedup_counter_gap"]["next_refs"])
        self.assert_image_state(memory.state.images, GOLDEN["download_dedup_counter_gap"]["images"])

    def test_m03_alias_case_basename_single_fallback_and_missing_refs(self):
        for count in (1, 2):
            memory = MemoryStore()
            memory.start(question(*[RED, BLUE][:count]))
            for case in GOLDEN["lookups"][str(count)]:
                with self.subTest(image_count=count, query=case["query"]):
                    self.assertEqual(plain(memory.resolve_image(case["query"])), case["result"])

    def test_m03_local_path_aliases_and_unregistered_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            registered = Path(tmp) / "Original.PNG"
            unrelated = Path(tmp) / "unrelated.png"
            registered.write_bytes(b"fixture bytes")
            unrelated.write_bytes(b"unregistered file")
            memory = MemoryStore()
            with patch.object(images, "local_image_to_data_url", return_value=RED):
                memory.start(question(str(registered)))
            for ref in [str(registered), "original.png", "/invented/ORIGINAL.PNG"]:
                self.assertEqual(memory.resolve_image(ref), (RED, "input_image_1", None))
            # Existing files are deliberately not guessed via single-image fallback.
            self.assertEqual(memory.resolve_image(str(unrelated)), (None, None, None))

    def test_m04_lazy_registration_fetches_only_on_first_successful_reuse(self):
        memory = MemoryStore()
        memory.start(question(RED, BLUE))
        ref = memory.register_lazy_image("https://fixture.invalid/lazy.png", provenance="Text_Search_Tool")
        self.assertIsNone(memory.state.images.registry[ref]["data_url"])
        self.assertEqual(memory.register_lazy_image("https://fixture.invalid/lazy.png"), ref)
        with patch.object(images, "_fetch_image_data_url", return_value=GREEN) as fetch:
            first = memory.resolve_image(ref)
            second = memory.resolve_image(ref)
        self.assertEqual(ref, GOLDEN["lazy_success"]["ref"])
        self.assertEqual(plain(first), GOLDEN["lazy_success"]["first"])
        self.assertEqual(plain(second), GOLDEN["lazy_success"]["second"])
        self.assertEqual(fetch.call_count, GOLDEN["lazy_success"]["fetch_calls"])
        self.assert_image_state(memory.state.images, GOLDEN["lazy_success"]["images"])

    def test_m04_failed_lazy_fetch_keeps_ref_and_retries_on_later_reuse(self):
        memory = MemoryStore()
        memory.start(question(RED, BLUE))
        ref = memory.register_lazy_image("https://fixture.invalid/lazy.png", provenance="Text_Search_Tool")
        with patch.object(images, "_fetch_image_data_url", return_value=None) as fetch:
            result = memory.resolve_image(ref)
        self.assertEqual(plain(result), GOLDEN["lazy_failure"]["resolution"])
        self.assertEqual(fetch.call_count, GOLDEN["lazy_failure"]["fetch_calls"])
        self.assert_image_state(memory.state.images, GOLDEN["lazy_failure"]["images"])
        with patch.object(images, "_fetch_image_data_url", return_value=GREEN):
            self.assertEqual(memory.resolve_image(ref), (GREEN, ref, "lazy_fetch"))
        self.assertEqual(memory.state.images.tool_generated_counter, 1)

    def test_m05_search_url_classification_punctuation_and_nested_resolution(self):
        memory = MemoryStore()
        recorded = GOLDEN["url_rewrite"]
        output = urls.rewrite_search_result_text(memory.state, recorded["input"], source_tool="Text_Search_Tool", turn_idx=3)
        self.assertEqual(output, recorded["output"])
        self.assert_image_state(memory.state.images, recorded["images"])
        self.assertEqual(plain(memory.state.urls), recorded["urls"])
        rewrites = []
        resolved = urls._resolve_urls_in_value(memory.state, recorded["resolve_input"], rewrites=rewrites, path="urls")
        self.assertEqual(resolved, recorded["resolve_output"])
        self.assertEqual(rewrites, recorded["rewrites"])
        self.assertEqual(memory.resolve_url("webpage_url_999"), "webpage_url_999")

    def test_m05_latest_twenty_inventory_keeps_oldest_reference_resolvable(self):
        memory = MemoryStore()
        memory.start([])
        for number in range(1, 26):
            memory.register_url(f"https://example.com/page/{number}", provenance="Text_Search_Tool", turn_idx=number)
        self.assertEqual(as_messages(memory.build_environment_update()), GOLDEN["inventory_limit"]["messages"])
        self.assertEqual(memory.resolve_url("[webpage_url_1]"), GOLDEN["inventory_limit"]["oldest_resolution"])
        self.assertEqual(memory.state.urls.webpage_counter, 25)
        self.assertEqual(memory.register_url("https://example.com/page/1."), "webpage_url_1")
        self.assertEqual(memory.state.urls.webpage_counter, 25)

    def test_m06_binding_sanitizer_truncation_and_collisions_match_old(self):
        memory = MemoryStore()
        keys = [memory.store_binding(prefix="depth", query=query, value={}, source_tool="Depth_Estimation_Tool",
                                     source_args={"query": query}, descriptions={})
                for query in GOLDEN["bindings"]["queries"]]
        self.assertEqual(keys, GOLDEN["bindings"]["keys"])
        self.assertEqual(memory.state.bindings.binding_counters, GOLDEN["bindings"]["counters"])

    def test_m06_binding_descriptors_and_copied_typed_values_match_old(self):
        memory = MemoryStore()
        key = add_depth(memory)
        self.assertEqual(memory.describe_bindings(), {key: GOLDEN["bindings"]["descriptor"]})
        stored = memory.state.bindings.bindings[key]
        self.assertEqual(stored["depth_meters"].dtype, np.dtype("float32"))
        self.assertEqual(stored["depth_meters"].shape, (2, 2))
        copied = memory.copy_bindings_for_execution()
        np.testing.assert_array_equal(copied[key]["depth_meters"], stored["depth_meters"])
        copied[key]["depth_meters"][0, 0] = 99
        copied[key]["depth_shape"][0] = 99
        self.assertEqual(stored["depth_meters"][0, 0], 0)
        self.assertEqual(stored["depth_shape"], [2, 2])
        descriptor = memory.describe_bindings()[key]
        self.assertNotIn("image_base64", descriptor["source_args"])
        self.assertNotIn("binding_descriptors", descriptor["source_args"])
        self.assertNotIn("long_value", descriptor["source_args"])
        self.assertEqual(len(descriptor["source_args"]["short_value"]), 200)

    def test_m07_coalesced_order_and_emit_policies_match_old(self):
        for policy in ("always", "on_change"):
            with self.subTest(policy=policy):
                recorded = GOLDEN["environment_" + policy]
                memory = MemoryStore(emit_policy=policy)
                self.assertEqual(as_messages(memory.start(question(RED))), recorded["initial"])
                self.assertEqual(as_messages(memory.build_environment_update()), recorded["unchanged"])
                add_depth(memory)
                memory.register_url("https://www.example.com/article", provenance="Text_Search_Tool", turn_idx=1)
                self.assertEqual(as_messages(memory.build_environment_update()), recorded["combined"])
                self.assertEqual(as_messages(memory.build_environment_update()), recorded["unchanged_after"])
                memory.register_lazy_image("https://fixture.invalid/registered-before-failure.png", provenance="Text_Search_Tool")
                self.assertEqual(as_messages(memory.build_environment_update(successful=False)), recorded["failed"])
                self.assertEqual(as_messages(memory.build_environment_update()), recorded["after_failed"])

    def test_m07_lazy_byte_fetch_does_not_change_inventory_identity(self):
        memory = MemoryStore(emit_policy="on_change")
        memory.start(question(RED))
        ref = memory.register_lazy_image("https://fixture.invalid/lazy.png")
        self.assertIsNotNone(memory.build_environment_update())
        with patch.object(images, "_fetch_image_data_url", return_value=GREEN):
            memory.resolve_image(ref)
        self.assertIsNone(memory.build_environment_update())


if __name__ == "__main__":
    unittest.main()
