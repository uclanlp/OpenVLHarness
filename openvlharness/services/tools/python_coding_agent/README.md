# Coding generation and execution

`Python_Coding_Agent_Tool` has one public input, `objective`. The capability
injects lightweight memory descriptors into the generation request. The router
uses `services.tools.python_coding_agent.model_service`, which must identify
the same model as the orchestrator, with `context: fresh`. It sends a separate
coding prompt and a 4096-token generation budget. GPT uses Responses with
medium reasoning; Qwen uses Chat Completions with temperature 0. The generator
does not inherit the full orchestrator conversation or sampling settings.

Actual NumPy arrays stay on the agent host. The local executor serializes copied
bindings into a disposable subprocess, preserving array dtype and shape. It
captures child stdout separately from agent threads and returns bounded variable
previews. Stdout capture retains at most 64 KiB, with a truncation marker. At most
64 variables are shown, with 2,048 characters per preview and a 16,384-character
total preview budget. Mutating the
execution copy does not mutate memory, and new variables are not automatically
persisted as typed bindings or converted to image artifacts.

Configure `services.tools.python_executor_local` with `transport: local` and
`timeout_seconds: 30` for the paper path. A positive finite deadline covers child
startup, execution and preview rendering. On expiry the worker is forcibly killed;
on POSIX the entire process group is killed, including children that inherit it.
This stops blocking calls and loops that catch Python exceptions.

Execution is not a sandbox. Generated code runs in a subprocess as your user,
with network and filesystem access, in a temporary working directory. Its
environment omits credential-like variables: names containing KEY, TOKEN,
SECRET, PASSWORD or CREDENTIAL in any case, and names starting with `AWS_` or
`GOOGLE_APPLICATION`. Child processes that the code detaches into a new session
or process group can outlive the timeout. Run the agent with permissions
appropriate for generated Python.

A thrown generator exception is a failed tool envelope. A returned generator
error dictionary is retained as canonical `generation_error` and shown in
model-facing evidence; code is not executed after a generation failure.
A local execution exception is `execution_error` inside successful
generation output, with visible execution-error evidence. No generic subagent,
GPU package, ambient LLM connection or package-local API-key file is required.
