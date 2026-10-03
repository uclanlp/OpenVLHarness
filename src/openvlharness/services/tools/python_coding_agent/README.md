# Coding generation and execution

`Python_Coding_Agent_Tool` has one public input, `objective`. The capability
injects lightweight memory descriptors into the generation request. The router
uses `services.tools.python_coding_agent.model_service`, which must identify
the same model as the orchestrator, with `context: fresh`. It sends a separate
coding prompt and a 4096-token generation budget. GPT uses Responses with
medium reasoning; Qwen uses Chat Completions with temperature 0. The generator
does not inherit the full orchestrator conversation or sampling settings.

Actual NumPy arrays stay on the agent host. The local executor receives copied
bindings, captures stdout and returns repr strings for variables. Mutating the
execution copy does not mutate memory, and new variables are not automatically
persisted as typed bindings or converted to image artifacts.

Configure `services.tools.python_executor_local` with `transport: local` and
`timeout_seconds: 30` for the paper path. The main-thread alarm and worker-thread
trace limit stop Python loops. They do not guarantee interruption of arbitrary
blocking native code or network calls. Execution is not a security sandbox;
run the agent with permissions appropriate for generated Python.

A thrown generator exception is a failed tool envelope. A returned generator
error dictionary preserves the old success-envelope boundary and is retained
in the trace. A local execution exception is `execution_error` inside successful
generation output, with visible execution-error evidence. No generic subagent,
GPU package, ambient LLM connection or package-local API-key file is required.
