from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from mini_vlm_tool_kit.tool_registry.loader import load_tools


tool_instances, tool_metadata = load_tools()

app = FastAPI()


class ExecuteRequest(BaseModel):
    tool_name: str
    args: dict


@app.get("/tools")
def list_tools():
    return tool_metadata


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

    uvicorn.run(app, host="127.0.0.1", port=8001)
