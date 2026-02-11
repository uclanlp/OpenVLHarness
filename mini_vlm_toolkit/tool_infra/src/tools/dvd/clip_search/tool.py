import os

from tools.base import BaseTool
from tools.api_utils import embeddings_request

TOOL_NAME = "ClipSearch_Tool"

# Env-driven configuration
ENV_EMBED_MODEL = "DVD_CS_EMBED_MODEL"
ENV_EMBED_BASE_URL = "DVD_CS_EMBED_BASE_URL"
ENV_EMBED_API_KEY = "DVD_CS_EMBED_API_KEY"
ENV_TOPK = "DVD_CS_TOPK"

DEFAULT_TOP_K = 16
INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "memory": {
            "type": "object",
            "description": "NanoVectorDB object (runtime-injected)",
        },
        "database": {
            "type": "object",
            "description": "Alias of memory for compatibility",
        },
        "event_description": {
            "type": "string",
            "description": "Event description to search for",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of top results to return",
            "minimum": 1,
        },
    },
    "required": ["event_description"],
    "anyOf": [
        {"required": ["memory"]},
        {"required": ["database"]},
    ],
    "additionalProperties": False,
}


class ClipSearch_Tool(BaseTool):
    def __init__(self):
        self.embed_model = os.getenv(ENV_EMBED_MODEL)
        self.embed_base_url = os.getenv(ENV_EMBED_BASE_URL)
        self.embed_api_key = os.getenv(ENV_EMBED_API_KEY, "EMPTY")
        self.default_topk = int(os.getenv(ENV_TOPK, str(DEFAULT_TOP_K)) or DEFAULT_TOP_K)

        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Search a video clip NanoVectorDB for events matching a description "
                "and return the top matching clip captions."
            ),
            tool_version="0.2.0",
            input_types={
                "database": "NanoVectorDB - preloaded vector database object",
                "event_description": "str - description of the event to search for",
                "top_k": "int - number of top results (default from env/DVD_CS_TOPK or 16)",
            },
            input_schema=INPUT_SCHEMA,
            output_type="str - concatenated captions for top results",
            demo_commands=[
                {
                    "command": (
                        "ClipSearch_Tool(database=db_obj, "
                        "event_description=\"A person enters the kitchen\")"
                    ),
                    "description": "Search for relevant clip captions.",
                }
            ],
            user_metadata={
                "env": {
                    ENV_EMBED_MODEL: "required",
                    ENV_EMBED_BASE_URL: "required",
                    ENV_EMBED_API_KEY: "optional",
                    ENV_TOPK: "optional",
                },
                "status": "implemented",
            },
        )

    def sanity_check(self):
        missing = []
        for name, value in [
            (ENV_EMBED_MODEL, self.embed_model),
            (ENV_EMBED_BASE_URL, self.embed_base_url),
        ]:
            if not value or not str(value).strip():
                missing.append(f"{name} is required")
        return missing

    def execute(self, memory=None, event_description=None, top_k=None, database=None):
        if memory is None:
            memory = database
        if memory is None:
            raise ValueError("memory is required")
        database = memory
        if not event_description:
            raise ValueError("event_description is required")

        if top_k is None:
            top_k = self.default_topk

        query_emb = embeddings_request(
            model=self.embed_model,
            input_text=event_description,
            base_url=self.embed_base_url,
            api_key=self.embed_api_key,
        )[0]

        results = database.query(query_emb, top_k=top_k)
        captions = [
            (data.get("time_start_secs", 0), data.get("caption", ""))
            for data in results
        ]
        captions = sorted(captions, key=lambda x: x[0])
        caption_text = "\n".join([cap[1] for cap in captions if cap[1]])
        return "Here is the searched video clip scripts:\n\n" + caption_text
