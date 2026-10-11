"""Every registered model profile and shipped preset resolves offline."""
from importlib.resources import files
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import yaml

from openvlharness.bootstrap import load_agent_config, resolved_model_id
from openvlharness.services.models import ModelAdapter, load_profile, require_raw_trace_profile


REGISTRY = yaml.safe_load(files("openvlharness.services").joinpath("model_registry.yaml").read_text())["profiles"]
PRESETS = sorted((Path(__file__).resolve().parents[1] / "configs" / "agent").glob("*.yaml"))
PRESET_ENV = {name: "http://fixture.invalid:8080" for name in (
    "TOOL_SERVER_URL", "SAM3_SERVER_URL", "OCR_SERVER_URL", "DEPTH_ESTIMATION_SERVER_URL",
    "MODEL_BASE_URL", "AUXILIARY_MODEL_BASE_URL")}
PRESET_ENV.update(MODEL_API_KEY="fixture-key", AUXILIARY_MODEL_API_KEY="fixture-key")
FIELDS = {"model_id", "prompt_bundle", "adapter", "final_answer", "context_policy", "extra_final_request",
          "continue_empty_answer", "append_final_message", "context_error_fallback", "allowed_options"}


class RegistryTests(unittest.TestCase):
    def test_every_profile_is_complete_and_builds_a_native_request(self):
        prompts = files("openvlharness.prompts")
        for alias, profile in REGISTRY.items():
            with self.subTest(alias=alias):
                self.assertLessEqual(FIELDS, set(profile))
                self.assertTrue(prompts.joinpath(profile["prompt_bundle"] + "_system.txt").is_file())
                adapter = ModelAdapter({"alias": alias, "base": "https://fixture.invalid/v1", "max_tokens": 64})
                payload = adapter.payload([adapter.encode_message({"role": "user", "content": "hi"})])
                self.assertEqual(payload["model"], profile["model_id"])
                if profile["adapter"] == "responses":
                    self.assertEqual(adapter.endpoint, "https://fixture.invalid/v1/responses")
                    self.assertEqual((payload["max_output_tokens"], payload["store"]), (64, False))
                else:
                    self.assertEqual(adapter.endpoint, "https://fixture.invalid/v1/chat/completions")
                    self.assertEqual(payload["max_tokens"], 64)

    def test_served_model_ids_are_unique(self):
        ids = [profile["model_id"] for profile in REGISTRY.values()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_qwen3_vl_sizes_share_the_8b_contract_without_raw_export(self):
        reference = {key: value for key, value in load_profile("qwen3-vl-8b-instruct").items()
                     if not key.startswith("raw_trace") and key != "model_id"}
        family = [alias for alias in REGISTRY if alias.startswith("qwen3-vl-")]
        self.assertGreaterEqual(len(family), 12)
        for alias in family:
            with self.subTest(alias=alias):
                profile = load_profile(alias)
                self.assertEqual({k: v for k, v in profile.items() if k != "model_id" and not k.startswith("raw_trace")},
                                 reference)
                self.assertTrue(profile["model_id"].startswith("Qwen/Qwen3-VL-"))
                if alias != "qwen3-vl-8b-instruct":
                    with self.assertRaisesRegex(ValueError, "Raw trace export"):
                        require_raw_trace_profile(alias)

    def test_openai_profiles_share_the_luna_contract(self):
        reference = {k: v for k, v in load_profile("gpt-6-luna").items() if k != "model_id"}
        for alias, profile in REGISTRY.items():
            if profile["adapter"] == "responses":
                with self.subTest(alias=alias):
                    self.assertEqual({k: v for k, v in load_profile(alias).items() if k != "model_id"}, reference)
                    adapter = ModelAdapter({"alias": alias, "base": "https://fixture.invalid", "reasoning_effort": "high"})
                    self.assertEqual(adapter.payload([])["reasoning"], {"effort": "high"})

    def test_openai_service_tier_is_sent_and_validated(self):
        adapter = ModelAdapter({"alias": "gpt-6-astra", "base": "https://fixture.invalid", "service_tier": "flex"})
        self.assertEqual(adapter.payload([])["service_tier"], "flex")
        self.assertNotIn("service_tier", ModelAdapter({"alias": "gpt-6-astra", "base": "https://fixture.invalid"}).payload([]))
        with self.assertRaisesRegex(ValueError, "service_tier"):
            ModelAdapter({"alias": "gpt-6-astra", "base": "https://fixture.invalid", "service_tier": "cheap"})
        with self.assertRaisesRegex(ValueError, "Unsupported options"):
            ModelAdapter({"alias": "kimi-k3", "base": "https://fixture.invalid", "service_tier": "flex"})

    def test_kimi_sends_reasoning_effort_top_level_and_rejects_vllm_options(self):
        adapter = ModelAdapter({"alias": "kimi-k3", "base": "https://fixture.invalid/v1",
                                "max_tokens": 65536, "temperature": 0.6, "top_p": 0.95, "reasoning_effort": "max"})
        payload = adapter.payload([{"role": "user", "content": "hi"}])
        self.assertEqual({k: payload[k] for k in ("model", "max_tokens", "temperature", "top_p", "reasoning_effort")},
                         {"model": "kimi-k3", "max_tokens": 65536, "temperature": 0.6, "top_p": 0.95,
                          "reasoning_effort": "max"})
        for option in ({"top_k": 20}, {"presence_penalty": 1.5}, {"repetition_penalty": 1.0}, {"return_token_ids": False}):
            with self.subTest(option=option), self.assertRaises(ValueError):
                ModelAdapter({"alias": "kimi-k3", "base": "https://fixture.invalid", **option})


class PresetTests(unittest.TestCase):
    def test_every_shipped_preset_loads_and_names_a_registered_orchestrator(self):
        self.assertGreaterEqual(len(PRESETS), 6)
        with patch.dict(os.environ, PRESET_ENV):
            for path in PRESETS:
                with self.subTest(preset=path.name):
                    config = load_agent_config(path)
                    alias = config["services"]["models"][config["agent"]["model_service"]]["alias"]
                    self.assertEqual(resolved_model_id(path), REGISTRY[alias]["model_id"])
                    self.assertEqual(config["services"]["models"]["auxiliary"]["alias"], "qwen3-vl-8b-instruct")
