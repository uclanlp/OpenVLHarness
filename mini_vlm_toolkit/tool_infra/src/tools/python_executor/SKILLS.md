# Python_Executor_Tool Usage Guide

Execute a provided Python code snippet and return stdout plus local variables. Important: if you need the numeric/text result explicitly, use print(...) and/or assign the value to a variable. Bare expressions are executed but their value is not returned automatically. Do not use with untrusted code. Note: always include explicit import statements for any modules you use (e.g., `import math`, `import os`). No modules are pre-imported.

## Parameters
- `code` (string, required): Python code to execute. Can be raw code or a fenced ```python/```py code block. Use print(...) and/or variable assignment if you need explicit values in the result.

## Call Format
{"type":"tool_call","name":"Python_Executor_Tool","arguments":{"code":"import math\nresult = math.sqrt(144)\nprint(result)"}}

## Best Practices
1. ALWAYS USE print() — bare expressions do not appear in the output. If you need a value, print it.
2. ALWAYS IMPORT MODULES — no modules are pre-imported. Write `import math`, `import os`, etc. explicitly.
3. ASSIGN TO VARIABLES — local variables are captured and returned alongside stdout, so named variables help you inspect results.
4. USE FOR COMPUTATION — this tool is ideal for arithmetic, data processing, string manipulation, and any logic that benefits from exact computation rather than mental math.
5. KEEP CODE SELF-CONTAINED — each execution is independent. Do not rely on state from previous calls.

## Examples
### Example: Calculate an arithmetic expression
{"type":"tool_call","name":"Python_Executor_Tool","arguments":{"code":"result = 17 * 23 + 45\nprint(result)"}}
Expected result: `{"printed_output": "436\n", "variables": {"result": 436}, "execution_code": "..."}`

### Example: Count items in a list
{"type":"tool_call","name":"Python_Executor_Tool","arguments":{"code":"items = ['apple', 'banana', 'cherry', 'apple', 'banana']\nfrom collections import Counter\ncounts = Counter(items)\nprint(counts)"}}
Expected result: `{"printed_output": "Counter({'apple': 2, 'banana': 2, 'cherry': 1})\n", "variables": {"items": [...], "counts": ...}, "execution_code": "..."}`
