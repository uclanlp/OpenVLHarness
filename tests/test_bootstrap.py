"""Configuration and import boundaries, independent of local deployments."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from openvlharness.bootstrap import (
    PAPER_BACKENDS, build_agent, build_capabilities, load_agent_config,
    memory_factory, resolved_model_id,
)


def config_for(*names):
    selected = {name: ({"generator": "python_coding_agent", "executor": "python_executor_local"}
                       if name == "coding" else {"backend": PAPER_BACKENDS[name]}) for name in names}
    tools = {}
    for name in names:
        backend = PAPER_BACKENDS[name]
        if name in {"grounding", "ocr", "depth", "camera_pose"}:
            tools[backend] = {"server_url": "http://raw.invalid:8080"}
        elif name == "coding":
            tools[backend] = {"model_service": "orchestrator", "context": "fresh"}
            tools["python_executor_local"] = {"transport": "local", "timeout_seconds": 30}
        elif name in {"text_search", "image_search", "web_visit"}:
            tools[backend] = {"model_service": "auxiliary"}
        else:
            tools[backend] = {}
    return {"agent_name": "MainAgent", "agent_config_version": "paper_v1",
            "agent": {"model_service": "orchestrator", "max_iterations": 3},
            "memory": {"modules": [{"name": "image"}, {"name": "tool_context"}, {"name": "url"}],
                       "emit_policy": "always"},
            "capabilities": {"registries": ["paper"], "tools": selected},
            "services": {"client": {"base_url": "http://router.invalid:8101", "timeout_seconds": 17, "retry": 0},
                         "server": {"host": "127.0.0.1", "port": 8101, "registration": "configured_only"},
                         "tools": tools,
                         "models": {"orchestrator": {"alias": "gpt-6-luna", "base": "https://model.invalid/v1",
                                                      "key": "fixture-key", "timeout": 19, "retry": 0},
                                    "auxiliary": {"alias": "qwen3-vl-8b-instruct", "base": "http://aux.invalid/v1",
                                                  "key": "fixture-aux-key", "timeout": 23, "retry": 0}}}}


class BootstrapBoundaryTests(unittest.TestCase):
    def setUp(self):
        guard = patch("requests.sessions.Session.request", side_effect=AssertionError("bootstrap must not request inference"))
        guard.start()
        self.addCleanup(guard.stop)

    def test_configuration_expands_env_without_mutating_input_and_reads_yaml(self):
        config = config_for("ocr")
        config["services"]["tools"]["ocr"]["server_url"] = "${OPENVL_TEST_RAW_URL}"
        before = copy.deepcopy(config)
        with patch.dict(os.environ, {"OPENVL_TEST_RAW_URL": "http://fixture.invalid:8085"}):
            loaded = load_agent_config(config)
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "agent.yaml"
                path.write_text(json.dumps(config))  # JSON is also a YAML mapping.
                self.assertEqual(load_agent_config(path), loaded)
        self.assertEqual(config, before)
        self.assertEqual(loaded["services"]["tools"]["ocr"]["server_url"], "http://fixture.invalid:8085")

    def test_missing_env_and_unknown_alias_fail_before_inference_or_backend_import(self):
        config = config_for("ocr")
        config["services"]["tools"]["ocr"]["server_url"] = "${OPENVL_TEST_MISSING_URL}"
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENVL_TEST_MISSING_URL"):
                load_agent_config(config)
        config = config_for("ocr")
        config["services"]["models"]["orchestrator"]["alias"] = "gpt-unregistered"
        with self.assertRaisesRegex(ValueError, "Register.*model_registry"):
            build_agent(config)

    def test_selected_capabilities_follow_old_router_order_and_ablation_is_explicit(self):
        config = load_agent_config(config_for("coding", "ocr", "grounding"))
        capabilities = build_capabilities(config)
        self.assertEqual(list(capabilities), ["OCR_Tool", "Python_Coding_Agent_Tool", "Visual_Grounding_Tool"])
        self.assertNotIn("Depth_Estimation_Tool", capabilities)
        self.assertEqual(capabilities["Python_Coding_Agent_Tool"].service_config["execution_timeout_seconds"], 30)
        self.assertEqual(build_capabilities(load_agent_config(config_for())), {})

    def test_context_budget_rejected_during_config_load_and_before_agent_dependencies(self):
        with patch("openvlharness.services.models.ModelAdapter.load_tokenizer") as tokenizer, \
             patch("openvlharness.bootstrap.build_capabilities") as capabilities:
            for alias in ("qwen3-vl-8b-instruct", "gpt-6-luna"):
                for budget in (None, 100):
                    config = config_for("ocr")
                    model = config["services"]["models"]["orchestrator"]
                    model["alias"] = alias
                    if budget is not None:
                        model["max_tokens"] = budget
                    config["agent"]["max_context_length"] = (budget or 4096) + 512
                    for factory in (load_agent_config, build_agent):
                        with self.subTest(alias=alias, budget=budget, factory=factory.__name__):
                            with self.assertRaisesRegex(ValueError, r"max_context_length.*max_tokens.*512"):
                                factory(config)
            tokenizer.assert_not_called()
            capabilities.assert_not_called()

    def test_context_budget_applies_to_selected_orchestrator_only(self):
        config = config_for("text_search")
        config["agent"].update(model_service="main", max_context_length=1000)
        models = config["services"]["models"]
        models["main"] = models.pop("orchestrator")
        models["main"]["max_tokens"] = 100
        models["auxiliary"]["max_tokens"] = 8192
        before = copy.deepcopy(config)
        self.assertEqual(load_agent_config(config), before)
        agent = build_agent(config)
        self.assertEqual(agent.adapter.max_context_length, 1000)
        self.assertEqual(agent.adapter.max_tokens, 100)
        self.assertEqual(config, before)

    def test_dino_requires_extension_and_never_adds_a_public_tool(self):
        config = config_for("grounding")
        config["capabilities"]["tools"]["grounding"]["backend"] = "grounding_dino"
        config["services"]["tools"]["grounding_dino"] = {"server_url": "http://dino.invalid:8088"}
        with self.assertRaisesRegex(ValueError, "Unsupported grounding backend"):
            load_agent_config(config)
        config["capabilities"]["registries"].append("extensions")
        capabilities = build_capabilities(load_agent_config(config))
        self.assertEqual(list(capabilities), ["Visual_Grounding_Tool"])
        self.assertEqual(capabilities["Visual_Grounding_Tool"].backend, "grounding_dino")

    def test_rejects_legacy_unknown_sections_task_knobs_and_unknown_backends(self):
        variants = []
        config = config_for("grounding"); config["agent_name"] = "BaseAgentWithTools"; variants.append(config)
        config = config_for("grounding"); config["eval"] = {}; variants.append(config)
        config = config_for("grounding"); config["capabilities"]["tools"]["grounding"]["query"] = "shoe"; variants.append(config)
        config = config_for("grounding"); config["capabilities"]["tools"]["grounding"]["backend"] = "unknown"; variants.append(config)
        config = config_for("grounding"); config["services"]["tools"]["sam3_segment"]["box_threshold"] = 0.2; variants.append(config)
        config = config_for("text_search"); config["services"]["tools"]["serper_text_search"]["num_results"] = 1; variants.append(config)
        for config in variants:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    load_agent_config(config)

    def test_missing_raw_endpoint_and_model_service_are_actionable(self):
        config = config_for("depth")
        config["services"]["tools"]["depth_estimation"] = {}
        with self.assertRaisesRegex(ValueError, "depth_estimation.server_url"):
            load_agent_config(config)
        config = config_for("text_search")
        config["services"]["tools"]["serper_text_search"]["model_service"] = "absent"
        with self.assertRaisesRegex(ValueError, "Missing model service 'absent'"):
            load_agent_config(config)

    def test_raw_endpoint_rejects_wrong_scheme_and_method_suffix_at_startup(self):
        for value in ("ftp://raw.invalid", "http://raw.invalid/v1/depth", "http://raw.invalid/v1/pose"):
            with self.subTest(value=value):
                config = config_for("depth")
                config["services"]["tools"]["depth_estimation"]["server_url"] = value
                with self.assertRaises(ValueError):
                    load_agent_config(config)

    def test_client_url_timeout_and_retry_are_validated_before_agent_use(self):
        for field, value in (("base_url", "http://router.invalid/execute"), ("base_url", "ftp://router.invalid"),
                             ("timeout_seconds", 0), ("retry", -1), ("retry", True)):
            with self.subTest(field=field, value=value):
                config = config_for("ocr")
                config["services"]["client"][field] = value
                with self.assertRaises(ValueError):
                    build_agent(config)

    def test_memory_factory_rejects_order_duplicates_and_bad_module_options(self):
        for memory in ({"modules": [{"name": "url"}, {"name": "image"}]},
                       {"modules": [{"name": "image"}, {"name": "image"}]},
                       {"modules": [{"name": "not_a_module"}]},
                       {"modules": [{"name": "image", "unknown": True}]},
                       {"modules": [{"name": "url", "webpage_env_update_limit": 0}]},
                       {"modules": [], "emit_policy": "sometimes"}):
            with self.subTest(memory=memory):
                with self.assertRaises(ValueError):
                    memory_factory(memory)()
        factory = memory_factory({"modules": [{"name": "url", "webpage_env_update_limit": None}]})
        self.assertIsNot(factory().state, factory().state)

    def test_build_agent_loads_packaged_prompts_and_same_backbone_coding_connection(self):
        config = config_for("coding", "ocr")
        agent = build_agent(config)
        self.assertEqual(agent.model, "gpt-6-luna")
        self.assertEqual(resolved_model_id(config), agent.model)
        self.assertTrue(agent.system_prompt)
        self.assertTrue(agent.user_prefix)
        self.assertEqual(agent.adapter.protocol, "responses")
        self.assertEqual(agent.capabilities["Python_Coding_Agent_Tool"].service_config["model_service"], "orchestrator")
        self.assertIsNot(agent.memory_factory().state, agent.memory_factory().state)

    def test_coding_rejects_backbone_mismatch_and_nonlocal_executor(self):
        config = config_for("coding")
        config["services"]["tools"]["python_coding_agent"]["model_service"] = "auxiliary"
        with self.assertRaises(ValueError):
            load_agent_config(config)
        config = config_for("coding")
        config["services"]["tools"]["python_executor_local"]["transport"] = "remote"
        with self.assertRaisesRegex(ValueError, "local transport"):
            build_capabilities(load_agent_config(config))

    def test_t02_installed_client_imports_outside_repo_without_optional_or_legacy_packages(self):
        # A fresh interpreter avoids hiding an eager import in pytest's module cache.
        script = textwrap.dedent('''
            import importlib.abc, json, sys
            blocked = {'torch', 'transformers', 'cv2', 'fastapi', 'uvicorn', 'openai',
                       'boto3', 'bs4', 'sam3', 'depth_anything_3', 'paddle', 'mini_vlm_toolkit'}
            class BlockOptional(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if fullname.split('.')[0] in blocked or 'subagent' in fullname:
                        raise ModuleNotFoundError('Optional package intentionally unavailable: '+fullname)
            sys.meta_path.insert(0, BlockOptional())
            import openvlharness
            from openvlharness.bootstrap import load_agent_config, build_agent
            from openvlharness.agent.agent import MainAgent
            from openvlharness.services.client import ToolClient
            assert not any(name.split('.')[0] in blocked for name in sys.modules)
            print(json.dumps({'imported': True, 'package': openvlharness.__file__}))
        ''')
        with tempfile.TemporaryDirectory() as cwd:
            proc = subprocess.run([sys.executable, "-I", "-c", script], cwd=cwd,
                                  capture_output=True, text=True, timeout=20)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertTrue(json.loads(proc.stdout)["imported"])


if __name__ == "__main__":
    unittest.main()
