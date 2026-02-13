import os

from tools.base import BaseTool
from tools.api_utils import chat_completion_request, embeddings_request

TOOL_NAME = "GlobalBrowse_Tool"

# Required configuration (pulled at init)
ENV_VLM_MODEL = "DVD_GB_VLM_MODEL"
ENV_VLM_BASE_URL = "DVD_GB_VLM_BASE_URL"
ENV_VLM_API_KEY = "DVD_GB_VLM_API_KEY"

ENV_EMBED_MODEL = "DVD_GB_EMBED_MODEL"
ENV_EMBED_BASE_URL = "DVD_GB_EMBED_BASE_URL"
ENV_EMBED_API_KEY = "DVD_GB_EMBED_API_KEY"

ENV_TOPK = "DVD_GB_TOPK"

DEFAULT_TASK_DESCRIPTION = (
    "Given a detailed question, retrieve relevant video clip captions."
)
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
        "query": {"type": "string", "description": "Question or query to answer"},
    },
    "required": ["query"],
    "anyOf": [
        {"required": ["memory"]},
        {"required": ["database"]},
    ],
    "additionalProperties": False,
}


class GlobalBrowse_Tool(BaseTool):
    """
    Instantiate with VLM and embedding endpoints; runtime execute only needs
    a NanoVectorDB path and a query.
    """

    def __init__(self):
        # Load configuration at init (env-driven to stay loader-friendly)
        self.vlm_model = os.getenv(ENV_VLM_MODEL)
        self.vlm_base_url = os.getenv(ENV_VLM_BASE_URL)
        self.vlm_api_key = os.getenv(ENV_VLM_API_KEY, "EMPTY")

        self.embed_model = os.getenv(ENV_EMBED_MODEL)
        self.embed_base_url = os.getenv(ENV_EMBED_BASE_URL)
        self.embed_api_key = os.getenv(ENV_EMBED_API_KEY, "EMPTY")

        self.top_k = int(os.getenv(ENV_TOPK, "300") or 300)

        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Search a NanoVectorDB of video clip captions using embeddings, "
                "then answer the query with a VLM based on the retrieved captions."
            ),
            tool_version="0.2.0",
            input_types={
                "database": "NanoVectorDB - preloaded vector database object",
                "query": "str - question or description to answer",
            },
            input_schema=INPUT_SCHEMA,
            output_type="dict - subject_registry, query_related_event",
            demo_commands=[
                {
                    "command": (
                        "GlobalBrowse_Tool(database=db_obj, "
                        "query=\"What happens after the alarm rings?\")"
                    ),
                    "description": "Browse globally and answer a question.",
                }
            ],
            user_metadata={
                "env": {
                    ENV_VLM_MODEL: "required",
                    ENV_VLM_BASE_URL: "required",
                    ENV_VLM_API_KEY: "optional",
                    ENV_EMBED_MODEL: "required",
                    ENV_EMBED_BASE_URL: "required",
                    ENV_EMBED_API_KEY: "optional",
                    ENV_TOPK: "optional (default 300)",
                },
                "status": "implemented",
            },
        )

    def sanity_check(self):
        missing = []
        for name, value in [
            (ENV_VLM_MODEL, self.vlm_model),
            (ENV_VLM_BASE_URL, self.vlm_base_url),
            (ENV_EMBED_MODEL, self.embed_model),
            (ENV_EMBED_BASE_URL, self.embed_base_url),
        ]:
            if not value or not str(value).strip():
                missing.append(f"{name} is required")
        return missing

    def execute(self, memory=None, query=None, database=None):
        if memory is None:
            memory = database
        if memory is None:
            raise ValueError("memory is required")
        database = memory
        if not query:
            raise ValueError("query is required")

        # Embedding for query
        query_emb = embeddings_request(
            model=self.embed_model,
            input_text=query,
            base_url=self.embed_base_url,
            api_key=self.embed_api_key,
        )[0]

        results = database.query(query_emb, top_k=self.top_k)
        captions = [
            (data.get("time_start_secs", 0), data.get("caption", ""))
            for data in results
        ]
        captions = sorted(captions, key=lambda x: x[0])
        caption_text = "\n".join([cap[1] for cap in captions if cap[1]])
        clip_captions = "Here is the searched video clip scripts:\n\n" + caption_text

        input_msgs = [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable assistant specializing in analyzing "
                    "video content and providing detailed, insightful answers."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Below are descriptions of video clips, each with its "
                    "corresponding timestamp. Carefully review the sequence of "
                    "events, the details and movements of objects, and the actions "
                    "and poses of people. Based on these observations, provide a "
                    "thorough and specific answer to the following question, "
                    "referencing key events and timestamps as appropriate.\n"
                    "Question: {question}\n\n{clip_captions}"
                ).format(question=query, clip_captions=clip_captions),
            },
        ]

        response = chat_completion_request(
            model=self.vlm_model,
            base_url=self.vlm_base_url,
            api_key=self.vlm_api_key,
            messages=input_msgs,
            temperature=0.0,
            max_tokens=512,
        )
        message = response.get("choices", [{}])[0].get("message", {})
        content = message.get("content")
        if content is None:
            raise ValueError("No response from the model")

        subject_registry = None
        try:
            subject_registry = database.get_additional_data().get("subject_registry")
        except Exception:
            subject_registry = None
        return {
            "subject_registry": subject_registry,
            "query_related_event": content,
        }
