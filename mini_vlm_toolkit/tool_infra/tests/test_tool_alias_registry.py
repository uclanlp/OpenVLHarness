import contextlib
import types

import tool_registry.loader as loader
from tools.base import BaseTool


class _ExampleFeatureTool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="ExampleFeature_Tool",
            tool_description="Example tool.",
            tool_version="0.1.0",
            input_types={"value": "str"},
            tool_aliases=["example_feature_alias"],
            output_type="dict",
        )

    def execute(self, value):
        return {"value": value}


class _CollisionToolA(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="CollisionA_Tool",
            tool_description="Collision A.",
            tool_version="0.1.0",
            input_types={},
            tool_aliases=["shared_alias"],
            output_type="dict",
        )

    def execute(self):
        return {}


class _CollisionToolB(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="CollisionB_Tool",
            tool_description="Collision B.",
            tool_version="0.1.0",
            input_types={},
            tool_aliases=["shared_alias"],
            output_type="dict",
        )

    def execute(self):
        return {}


class _InvalidAliasTool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="InvalidAlias_Tool",
            tool_description="Invalid alias tool.",
            tool_version="0.1.0",
            input_types={},
            tool_aliases=["bad-alias"],
            output_type="dict",
        )

    def execute(self):
        return {}


class _StrictFailingTool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name="StrictFailing_Tool",
            tool_description="Strict failing tool.",
            tool_version="0.1.0",
            input_types={},
            output_type="dict",
        )

    def sanity_check(self):
        return ["forced strict failure"]

    def execute(self):
        return {}


@contextlib.contextmanager
def _patch_loader_with_classes(class_map, *, auto_mode=False, module_attrs=None):
    module_name = "tests.fake_tool_module"
    module = types.ModuleType(module_name)
    for key, value in (module_attrs or {}).items():
        setattr(module, key, value)
    for class_name, class_obj in class_map.items():
        setattr(module, class_name, class_obj)

    original_get_tool_modules = loader._get_tool_modules
    original_import_module = loader.importlib.import_module

    def fake_get_tool_modules():
        return [module_name], auto_mode

    def fake_import_module(path):
        if path == module_name:
            return module
        return original_import_module(path)

    loader._get_tool_modules = fake_get_tool_modules
    loader.importlib.import_module = fake_import_module
    try:
        yield
    finally:
        loader._get_tool_modules = original_get_tool_modules
        loader.importlib.import_module = original_import_module


def test_load_tools_registers_explicit_and_snake_case_aliases():
    with _patch_loader_with_classes({"ExampleFeature_Tool": _ExampleFeatureTool}):
        tool_instances, tool_metadata = loader.load_tools()

    assert "ExampleFeature_Tool" in tool_instances
    assert "example_feature_alias" in tool_instances
    assert "example_feature" in tool_instances

    canonical_metadata = tool_metadata["ExampleFeature_Tool"]
    assert canonical_metadata["is_alias"] is False
    assert canonical_metadata["canonical_tool_name"] == "ExampleFeature_Tool"
    assert "example_feature_alias" in canonical_metadata["tool_aliases"]
    assert "example_feature" in canonical_metadata["tool_aliases"]

    alias_metadata = tool_metadata["example_feature_alias"]
    assert alias_metadata["is_alias"] is True
    assert alias_metadata["canonical_tool_name"] == "ExampleFeature_Tool"
    assert alias_metadata["tool_name"] == "example_feature_alias"


def test_alias_collision_raises():
    with _patch_loader_with_classes(
        {
            "CollisionA_Tool": _CollisionToolA,
            "CollisionB_Tool": _CollisionToolB,
        }
    ):
        try:
            loader.load_tools()
        except RuntimeError as exc:
            assert "Alias collision" in str(exc)
            return
    raise AssertionError("Expected alias collision to raise RuntimeError")


def test_invalid_alias_raises():
    with _patch_loader_with_classes({"InvalidAlias_Tool": _InvalidAliasTool}):
        try:
            loader.load_tools()
        except RuntimeError as exc:
            assert "valid Python identifier" in str(exc)
            return
    raise AssertionError("Expected invalid alias to raise RuntimeError")


def test_strict_module_sanity_failure_raises_in_auto_mode():
    with _patch_loader_with_classes(
        {"StrictFailing_Tool": _StrictFailingTool},
        auto_mode=True,
        module_attrs={"STRICT_TOOL_LOADING": True},
    ):
        try:
            loader.load_tools()
        except RuntimeError as exc:
            assert "Sanity check failed for StrictFailing_Tool" in str(exc)
            return
    raise AssertionError("Expected strict module sanity failure to raise RuntimeError")


def main():
    test_load_tools_registers_explicit_and_snake_case_aliases()
    test_alias_collision_raises()
    test_invalid_alias_raises()
    test_strict_module_sanity_failure_raises_in_auto_mode()
    print("Tool alias registry tests completed.")


if __name__ == "__main__":
    main()
