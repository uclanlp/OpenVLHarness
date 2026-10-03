# Local Python executor

This module runs on the **agent host**, not a separate raw HTTP server. It
operates on copied in-memory bindings so depth/camera arrays remain NumPy
values. See [coding semantics and timeout limits](../python_coding_agent/README.md).
