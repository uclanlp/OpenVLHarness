import ast
import re


def _extract_code(text):
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_tool_execute_args(text):
    code = _extract_code(text)
    tree = ast.parse(code)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "execute":
            continue
        if not isinstance(node.func.value, ast.Name):
            continue
        if node.func.value.id != "tool":
            continue

        if node.args:
            raise ValueError("Only keyword arguments are supported")

        args = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError("Kwargs must be named")
            args[kw.arg] = ast.literal_eval(kw.value)

        return args

    raise ValueError("No tool.execute(...) call found")
