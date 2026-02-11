"""Parser for Python tool-call lines such as tool_name(...).

For JSON tool-call parsing, see json_tool_call_parser.py.
"""

import ast
import builtins
import keyword
import re


TOOL_CALL_PATTERN = re.compile(r"(?<![\w.])([A-Za-z_][A-Za-z0-9_]*)\s*\(")
BUILTIN_NAMES = set(dir(builtins))


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
            if TOOL_CALL_PATTERN.search(block):
                return block
        return blocks[0]
    return text.strip()


def _extract_tool_call_snippet(text):
    match = TOOL_CALL_PATTERN.search(text)
    if not match:
        return None

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


def _is_plausible_tool_name(name):
    if not isinstance(name, str):
        return False
    if not name.isidentifier():
        return False
    if keyword.iskeyword(name):
        return False
    if name in BUILTIN_NAMES:
        return False
    return True


def _iter_top_level_call_nodes(tree):
    for node in tree.body:
        call = None
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            call = node.value
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            call = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Call):
            call = node.value
        if call is not None:
            yield call


def _parse_python_tool_call_from_code(code):
    tree = ast.parse(code)
    tool_calls = []

    for node in _iter_top_level_call_nodes(tree):
        if not isinstance(node.func, ast.Name):
            continue
        if not _is_plausible_tool_name(node.func.id):
            continue
        tool_calls.append(node)

    if not tool_calls:
        raise ValueError("No tool_name(...) call found")
    if len(tool_calls) > 1:
        raise ValueError("Multiple tool_name(...) calls found")

    call = tool_calls[0]
    if call.args:
        raise ValueError("Only keyword arguments are supported")

    args = {}
    for kw in call.keywords:
        if kw.arg is None:
            raise ValueError("Kwargs must be named")
        args[kw.arg] = ast.literal_eval(kw.value)

    return call.func.id, args


def _should_abort_on_error(exc):
    return (
        isinstance(exc, ValueError)
        and str(exc) == "Multiple tool_name(...) calls found"
    )


def parse_python_tool_call(text):
    cleaned = _strip_think_blocks(text)
    code = _extract_code(cleaned)
    last_error = None
    try:
        tool_name, args = _parse_python_tool_call_from_code(code)
        return {"tool_name": tool_name, "args": args}
    except (SyntaxError, ValueError) as exc:
        if _should_abort_on_error(exc):
            raise
        last_error = exc

    for block in _extract_code_blocks(cleaned):
        try:
            tool_name, args = _parse_python_tool_call_from_code(block)
            return {"tool_name": tool_name, "args": args}
        except (SyntaxError, ValueError) as exc:
            if _should_abort_on_error(exc):
                raise
            last_error = exc

    for source in (code, cleaned):
        snippet = _extract_tool_call_snippet(source)
        if not snippet:
            continue
        try:
            tool_name, args = _parse_python_tool_call_from_code(snippet)
            return {"tool_name": tool_name, "args": args}
        except (SyntaxError, ValueError) as exc:
            if _should_abort_on_error(exc):
                raise
            last_error = exc

    if last_error:
        raise last_error
    raise ValueError("No tool_name(...) call found")


def parse_python_tool_call_args(text):
    return parse_python_tool_call(text)["args"]
