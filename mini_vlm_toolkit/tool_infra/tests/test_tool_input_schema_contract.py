import ast
from pathlib import Path

import server.app as server_app


TOOL_INFRA_ROOT = Path(__file__).resolve().parents[1]
TOOLS_ROOT = TOOL_INFRA_ROOT / "src" / "tools"


def _iter_tool_files():
    return sorted(TOOLS_ROOT.rglob("tool.py"))


def _is_base_tool_subclass(class_def):
    for base in class_def.bases:
        if isinstance(base, ast.Name) and base.id == "BaseTool":
            return True
        if isinstance(base, ast.Attribute) and base.attr == "BaseTool":
            return True
    return False


def _collect_schema_constants(module):
    constants = {}
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("INPUT_SCHEMA"):
                    try:
                        constants[target.id] = ast.literal_eval(node.value)
                    except Exception:
                        constants[target.id] = None
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Name)
                and target.id.startswith("INPUT_SCHEMA")
                and node.value is not None
            ):
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except Exception:
                    constants[target.id] = None
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
    try:
        return ast.literal_eval(expr)
    except Exception:
        return None


def test_all_tool_classes_define_valid_input_schema():
    failures = []
    checked_classes = 0

    for tool_file in _iter_tool_files():
        source = tool_file.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(tool_file))
        schema_constants = _collect_schema_constants(module)

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
                    f"{class_label}: input_schema must resolve to a dict literal or INPUT_SCHEMA* constant"
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


def main():
    test_all_tool_classes_define_valid_input_schema()
    print("Tool input schema contract tests completed.")


if __name__ == "__main__":
    main()
