import importlib
import os
import tempfile
import textwrap

MODULE_NAME = "tools.subagent.tool"
ENV_KEY = "SUBAGENT_CONFIG_DIR"


class _FakeBaseAgentWithTools:
    pass


class _FakeAgent(_FakeBaseAgentWithTools):
    def __init__(self):
        self.calls = 0
        self.last_message = None

    def generate(self, message, **kwargs):
        self.calls += 1
        self.last_message = message
        return {"prediction": "child answer"}


def _write_config(directory, file_name, body):
    path = os.path.join(directory, file_name)
    with open(path, "w", encoding="utf-8") as fout:
        fout.write(textwrap.dedent(body).strip() + "\n")
    return path


def _load_module_with_config_dir(config_dir):
    previous = os.environ.get(ENV_KEY)
    if config_dir is None:
        os.environ.pop(ENV_KEY, None)
    else:
        os.environ[ENV_KEY] = config_dir

    try:
        module = importlib.import_module(MODULE_NAME)
        module = importlib.reload(module)
        return module
    finally:
        if previous is None:
            os.environ.pop(ENV_KEY, None)
        else:
            os.environ[ENV_KEY] = previous


def test_subagent_discovery_builds_tool_class():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(
            tmpdir,
            "math.yaml",
            """
            subagent_name: MathAgent_Tool
            agent_name: BaseAgentWithTools
            agent_kwargs:
              base: http://child-model:8000
              key: EMPTY
              model: child-model
              timeout: 60
              tool_base: http://child-tools:8101
              tool_list:
                - Python_Executor_Tool
            """,
        )

        module = _load_module_with_config_dir(tmpdir)
        assert module.SUBAGENT_DISCOVERY_ERRORS == []
        assert len(module.SUBAGENT_TOOL_CLASSES) == 1

        tool = module.SUBAGENT_TOOL_CLASSES[0]()
        assert tool.tool_name == "MathAgent_Tool"
        assert tool.sanity_check() == []


def test_subagent_execute_uses_cached_agent_and_normalizes_base64_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = _write_config(
            tmpdir,
            "vision.yaml",
            """
            subagent_name: VisionAgent_Tool
            agent_name: BaseAgentWithTools
            agent_kwargs:
              base: http://child-model:8000
              key: EMPTY
              model: child-model
              timeout: 120
              tool_base: http://child-tools:8101
              tool_list:
                - Python_Executor_Tool
            """,
        )

        module = _load_module_with_config_dir(tmpdir)
        tool = module.SUBAGENT_TOOL_CLASSES[0]()

        fake_agent = _FakeAgent()
        build_calls = []

        def _fake_build_agent(config_path):
            build_calls.append(config_path)
            return fake_agent

        module.build_agent = _fake_build_agent
        module.BaseAgentWithTools = _FakeBaseAgentWithTools

        tiny_png_b64 = (
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO2fF6UAAAAASUVORK5CYII="
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "value": "inspect image"},
                    {"type": "image", "value": tiny_png_b64},
                ],
            }
        ]

        first = tool.execute(messages=messages)
        second = tool.execute(messages=messages)

        assert first["ok"] is True
        assert first["answer"] == "child answer"
        assert second["ok"] is True
        assert len(build_calls) == 1
        assert build_calls[0] == cfg_path
        assert fake_agent.calls == 2

        normalized_image_value = fake_agent.last_message[0]["content"][1]["value"]
        assert normalized_image_value.startswith("data:image/png;base64,")


def test_subagent_rejects_invalid_last_role():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(
            tmpdir,
            "dialog.yaml",
            """
            subagent_name: DialogAgent_Tool
            agent_name: BaseAgentWithTools
            agent_kwargs:
              base: http://child-model:8000
              key: EMPTY
              model: child-model
              timeout: 120
              tool_base: http://child-tools:8101
              tool_list:
                - Python_Executor_Tool
            """,
        )

        module = _load_module_with_config_dir(tmpdir)
        tool = module.SUBAGENT_TOOL_CLASSES[0]()

        payload = tool.execute(
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "text", "value": "I will continue."}],
                }
            ]
        )

        assert payload["ok"] is False
        assert payload["error"]["code"] == "invalid_messages_last_role"


def test_subagent_config_rejects_null_tool_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(
            tmpdir,
            "invalid.yaml",
            """
            subagent_name: InvalidAgent_Tool
            agent_name: BaseAgentWithTools
            agent_kwargs:
              base: http://child-model:8000
              key: EMPTY
              model: child-model
              timeout: 120
              tool_base: http://child-tools:8101
              tool_list: null
            """,
        )

        module = _load_module_with_config_dir(tmpdir)
        assert len(module.SUBAGENT_DISCOVERY_ERRORS) == 1
        assert "tool_list" in module.SUBAGENT_DISCOVERY_ERRORS[0]
        assert len(module.SUBAGENT_TOOL_CLASSES) == 1

        error_tool = module.SUBAGENT_TOOL_CLASSES[0]()
        issues = error_tool.sanity_check()
        assert len(issues) == 1
        assert "tool_list" in issues[0]


def test_subagent_config_rejects_subagent_tool_in_child_tool_list():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_config(
            tmpdir,
            "loop.yaml",
            """
            subagent_name: LoopAgent_Tool
            agent_name: BaseAgentWithTools
            agent_kwargs:
              base: http://child-model:8000
              key: EMPTY
              model: child-model
              timeout: 120
              tool_base: http://child-tools:8101
              tool_list:
                - loop_agent
            """,
        )

        module = _load_module_with_config_dir(tmpdir)
        assert len(module.SUBAGENT_DISCOVERY_ERRORS) == 1
        assert "forbidden subagent tool" in module.SUBAGENT_DISCOVERY_ERRORS[0]


def main():
    test_subagent_discovery_builds_tool_class()
    test_subagent_execute_uses_cached_agent_and_normalizes_base64_images()
    test_subagent_rejects_invalid_last_role()
    test_subagent_config_rejects_null_tool_list()
    test_subagent_config_rejects_subagent_tool_in_child_tool_list()
    print("Subagent tool tests completed.")


if __name__ == "__main__":
    main()
