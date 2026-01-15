from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mini_vlm_toolkit.tool_registry.loader import load_tools

app = FastAPI()
tool_instances = {}
tool_metadata = {}


def _format_input_types(input_types):
    if isinstance(input_types, dict):
        return ", ".join(f"{key}: {value}" for key, value in input_types.items())
    if isinstance(input_types, (list, tuple)):
        return ", ".join(str(item) for item in input_types)
    if input_types is None:
        return ""
    return str(input_types)


def _format_routes():
    routes = []
    for route in app.routes:
        methods = getattr(route, "methods", None)
        path = getattr(route, "path", None)
        name = getattr(route, "name", None)
        if not path or not methods:
            continue
        method_list = ", ".join(sorted(m for m in methods if m))
        routes.append(f"{method_list} {path} ({name})")
    return sorted(routes)


@app.on_event("startup")
def load_tool_registry():
    global tool_instances, tool_metadata
    try:
        tool_instances, tool_metadata = load_tools()
        if tool_metadata:
            routes = _format_routes()
            if routes:
                print("Endpoints:")
                for route in routes:
                    print(f"  - {route}")
            print("Serving tools:")
            for tool_name in sorted(tool_metadata):
                metadata = tool_metadata[tool_name]
                inputs = _format_input_types(metadata.get("input_types")) or "none"
                output_type = metadata.get("output_type") or "none"
                print(f"  - {tool_name} (inputs: {inputs}) -> {output_type}")
    except Exception as exc:
        print(f"Tool startup failed: {exc}")
        raise SystemExit(1)


class ExecuteRequest(BaseModel):
    tool_name: str
    args: dict


@app.get("/tools")
def list_tools():
    return tool_metadata


@app.get("/health")
def health_check():
    return {"status": "ok", "tool_count": len(tool_metadata), "tools": sorted(tool_metadata)}


@app.post("/execute")
def execute_tool(request: ExecuteRequest):
    tool = tool_instances.get(request.tool_name)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"Tool not found: {request.tool_name}")

    try:
        result = tool.execute(**request.args)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return {"tool_name": request.tool_name, "result": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8101)
