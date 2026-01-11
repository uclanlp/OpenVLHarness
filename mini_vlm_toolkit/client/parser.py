import ast
import re


def _strip_think_blocks(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def _extract_code_blocks(text):
    matches = re.findall(
        r"```\s*(?:[a-zA-Z0-9_+-]+)?\s*(.*?)\s*```",
        text,
        re.DOTALL,
    )
    return [match.strip() for match in matches if match.strip()]


def _extract_code(text):
    match = re.search(r"```\s*python\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    blocks = _extract_code_blocks(text)
    if blocks:
        for block in blocks:
            if "tool.execute" in block:
                return block
        return blocks[0]
    return text.strip()


def _extract_tool_execute_snippet(text):
    match = re.search(r"tool\.execute\s*\(", text)
    if not match:
        return None

    line_start = text.rfind("\n", 0, match.start()) + 1
    line_prefix = text[line_start:match.start()]
    exec_index = line_prefix.rfind("execution")
    if exec_index != -1:
        start = line_start + exec_index
    else:
        start = match.start()

    depth = 0
    in_string = None
    escape = False
    i = match.end() - 1
    end = None

    while i < len(text):
        if in_string:
            if in_string in ("'''", '"""'):
                if text[i:i + 3] == in_string:
                    in_string = None
                    i += 3
                    continue
                i += 1
                continue
            if escape:
                escape = False
            elif text[i] == "\\":
                escape = True
            elif text[i] == in_string:
                in_string = None
            i += 1
            continue

        if text[i:i + 3] in ("'''", '"""'):
            in_string = text[i:i + 3]
            i += 3
            continue

        if text[i] in ("'", '"'):
            in_string = text[i]
            i += 1
            continue

        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
        i += 1

    if end is None:
        snippet = text[start:].strip()
    else:
        snippet = text[start:end].strip()
    return snippet or None


def _parse_execute_args_from_code(code):
    tree = ast.parse(code)

    saw_execute = False
    saw_positional = False
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

        saw_execute = True
        if node.args:
            saw_positional = True
            continue

        args = {}
        for kw in node.keywords:
            if kw.arg is None:
                raise ValueError("Kwargs must be named")
            args[kw.arg] = ast.literal_eval(kw.value)

        return args

    if saw_execute and saw_positional:
        raise ValueError("Only keyword arguments are supported")
    raise ValueError("No tool.execute(...) call found")


def parse_tool_execute_args(text):
    cleaned = _strip_think_blocks(text)
    code = _extract_code(cleaned)
    last_error = None
    try:
        return _parse_execute_args_from_code(code)
    except (SyntaxError, ValueError) as exc:
        last_error = exc

    for block in _extract_code_blocks(cleaned):
        try:
            return _parse_execute_args_from_code(block)
        except (SyntaxError, ValueError) as exc:
            last_error = exc

    for source in (code, cleaned):
        snippet = _extract_tool_execute_snippet(source)
        if not snippet:
            continue
        try:
            return _parse_execute_args_from_code(snippet)
        except (SyntaxError, ValueError) as exc:
            last_error = exc

    if last_error:
        raise last_error
    raise ValueError("No tool.execute(...) call found")
