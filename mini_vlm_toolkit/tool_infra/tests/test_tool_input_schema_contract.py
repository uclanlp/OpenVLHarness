import ast
import os
from pathlib import Path

import server.app as server_app


TOOL_INFRA_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = TOOL_INFRA_ROOT / "src" / "tools"

# Keep dynamic schema values explicit and narrow: the static contract still
# rejects arbitrary module constants and code execution in INPUT_SCHEMA.
ALLOWED_ENV_INT_SCHEMA_CONSTANTS = {
    "serper_webpage_visit/tool.py": {
        "MAX_URLS": ("WEBPAGE_VISIT_MAX_URLS", "5"),
    },
}


def _iter_tool_files():
    return sorted(TOOLS_ROOT.rglob("tool.py"))


def _is_base_tool_subclass(class_def):
    for base in class_def.bases:
        if isinstance(base, ast.Name) and base.id == "BaseTool":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseTool":
            return True
    return False


def _single_assigned_name(node):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
    elif isinstance(node, ast.AnnAssign):
        target = node.target
    else:
        return None
    if isinstance(target, ast.Name):
        return target.id
    return None


def _eval_allowed_env_int(node, env_name, default):
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Name)
        or node.func.id != "int"
        or len(node.args) != 1
        or node.keywords
    ):
        raise ValueError("expected int(os.getenv(...))")

    getenv_call = node.args[0]
    if (
        not isinstance(getenv_call, ast.Call)
        or not isinstance(getenv_call.func, ast.Attribute)
        or not isinstance(getenv_call.func.value, ast.Name)
        or getenv_call.func.value.id != "os"
        or getenv_call.func.attr != "getenv"
        or len(getenv_call.args) != 2
        or getenv_call.keywords
    ):
        raise ValueError("expected os.getenv(env_name, default)")

    env_arg, default_arg = getenv_call.args
    if (
        not isinstance(env_arg, ast.Constant)
        or env_arg.value != env_name
        or not isinstance(default_arg, ast.Constant)
        or default_arg.value != default
    ):
        raise ValueError("unexpected env-backed schema constant")

    return int(os.getenv(env_name, default))


def _eval_schema_node(node, constants):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {
            _eval_schema_node(key, constants): _eval_schema_node(value, constants)
            for key, value in zip(node.keys, node.values)
        }
    if isinstance(node, ast.List):
        return [_eval_schema_node(item, constants) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval_schema_node(item, constants) for item in node.elts)
    if isinstance(node, ast.Name) and node.id in constants:
        value = constants[node.id]
        if value is None:
            raise ValueError(f"unresolved schema constant: {node.id}")
        return value
    return ast.literal_eval(node)


def _resolve_schema_expr(expr, constants):
    try:
        return ast.literal_eval(expr)
    except Exception:
        try:
            return _eval_schema_node(expr, constants)
        except Exception:
            return None


def _collect_schema_constants(module, tool_file):
    constants = {}
    rel_tool_file = tool_file.relative_to(TOOLS_ROOT).as_posix()
    allowed_env_ints = ALLOWED_ENV_INT_SCHEMA_CONSTANTS.get(rel_tool_file, {})
    for node in module.body:
        assigned_name = _single_assigned_name(node)
        if assigned_name in allowed_env_ints and node.value is not None:
            try:
                env_name, default = allowed_env_ints[assigned_name]
                constants[assigned_name] = _eval_allowed_env_int(
                    node.value, env_name, default
                )
            except Exception:
                constants[assigned_name] = None
            continue

        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("INPUT_SCHEMA"):
                    constants[target.id] = _resolve_schema_expr(node.value, constants)
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Name)
                and target.id.startswith("INPUT_SCHEMA")
                and node.value is not None
            ):
                constants[target.id] = _resolve_schema_expr(node.value, constants)
    return constants


def _extract_input_schema_expr(class_def):
    for node in class_def.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "__init__":
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            func = call.func
            if not isinstance(func, ast.Attribute) or func.attr != "__init__":
                continue
            owner = func.value
            if not isinstance(owner, ast.Call):
                continue
            if not isinstance(owner.func, ast.Name) or owner.func.id != "super":
                continue
            for kw in call.keywords:
                if kw.arg == "input_schema":
                    return kw.value
    return None


def _extract_execute_signature(class_def):
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and node.name == "execute":
            arg_names = set()
            for arg in node.args.posonlyargs:
                if arg.arg != "self":
                    arg_names.add(arg.arg)
            for arg in node.args.args:
                if arg.arg != "self":
                    arg_names.add(arg.arg)
            for arg in node.args.kwonlyargs:
                if arg.arg != "self":
                    arg_names.add(arg.arg)
            has_var_kwargs = node.args.kwarg is not None
            return arg_names, has_var_kwargs
    return set(), False


def _resolve_schema(expr, schema_constants):
    if expr is None:
        return None
    if isinstance(expr, ast.Name):
        return schema_constants.get(expr.id)
    return _resolve_schema_expr(expr, schema_constants)


def test_all_tool_classes_define_valid_input_schema():
    failures = []
    checked_classes = 0

    for tool_file in _iter_tool_files():
        source = tool_file.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(tool_file))
        schema_constants = _collect_schema_constants(module, tool_file)

        for class_def in module.body:
            if not isinstance(class_def, ast.ClassDef):
                continue
            if not _is_base_tool_subclass(class_def):
                continue

            checked_classes += 1
            class_label = f"{tool_file}:{class_def.name}"
            schema_expr = _extract_input_schema_expr(class_def)
            if schema_expr is None:
                failures.append(f"{class_label}: missing input_schema in super().__init__")
                continue

            schema = _resolve_schema(schema_expr, schema_constants)
            if not isinstance(schema, dict):
                failures.append(
                    f"{class_label}: input_schema must resolve to a dict literal, "
                    "INPUT_SCHEMA* constant, or whitelisted env-backed constant"
                )
                continue

            try:
                server_app._validate_input_schema(schema, class_label)
            except Exception as exc:
                failures.append(f"{class_label}: invalid schema: {exc}")
                continue

            execute_args, has_var_kwargs = _extract_execute_signature(class_def)
            if not execute_args and not has_var_kwargs:
                failures.append(f"{class_label}: missing execute method")
                continue

            properties = schema.get("properties", {}) or {}
            if not has_var_kwargs:
                unknown_properties = sorted(
                    prop for prop in properties.keys() if prop not in execute_args
                )
                if unknown_properties:
                    failures.append(
                        f"{class_label}: schema properties not accepted by execute: {unknown_properties}"
                    )

            required = schema.get("required", []) or []
            required_missing = sorted(
                key for key in required if key not in execute_args and not has_var_kwargs
            )
            if required_missing:
                failures.append(
                    f"{class_label}: required schema keys missing in execute signature: {required_missing}"
                )

    if checked_classes == 0:
        raise AssertionError("No BaseTool subclasses discovered in tool files.")
    if failures:
        formatted = "\n".join(f"- {item}" for item in failures)
        raise AssertionError(f"Input schema contract failures:\n{formatted}")


def test_auto_generated_skills_use_agent_facing_image_ids():
    metadata = {
        "tool_description": "Synthetic image tool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "Local image path"},
                "image_base64": {
                    "type": "string",
                    "description": "Base64 image data",
                },
                "query": {"type": "string", "description": "Object query"},
            },
            "required": ["query"],
            "oneOf": [
                {"required": ["image_path"]},
                {"required": ["image_base64"]},
            ],
            "additionalProperties": False,
        },
    }

    doc = server_app._auto_generate_skills("Synthetic_Image_Tool", metadata)

    assert "`image_id`" in doc
    assert "`image_path`" not in doc
    assert "`image_base64`" not in doc
    assert '"image_id": "<image_id>"' in doc
    assert '"image_path"' not in doc
    assert '"image_base64"' not in doc


def main():
    test_all_tool_classes_define_valid_input_schema()
    test_auto_generated_skills_use_agent_facing_image_ids()
    print("Tool input schema contract tests completed.")


if __name__ == "__main__":
    main()
