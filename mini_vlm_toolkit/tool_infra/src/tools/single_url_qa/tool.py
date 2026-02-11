import os
from typing import List

import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import OpenAI

from config import (
    DEFAULT_LLM_API_KEY,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)
from llm.engine import LLMEngine
from tools.base import BaseTool

TOOL_NAME = "Single_URL_QA_Tool"
KEY_FILENAME = "openai_api_key"
KEY_FILE_PATH = os.path.join(os.path.dirname(__file__), KEY_FILENAME)

SUMMARIZE_PROMPT_TEMPLATE = """
You are an expert AI assistant. Your task is to answer the user's query using
only the reference information below.

Instructions:
1) Read the query and the reference information.
2) Extract only the key facts needed to answer the query.
3) Provide a concise answer in your own words.

User Query:
{query}

Reference Information:
{reference_information}

Answer:
"""


class Single_URL_QA_Tool(BaseTool):
    require_llm_engine = True

    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description=(
                "Answer a query using content retrieved from a single URL."
            ),
            tool_version="1.0.0",
            input_types={"query": "str", "url": "str"},
            tool_aliases=["web_extractor"],
            output_type="str",
            demo_commands=[
                {
                    "command": (
                        "Single_URL_QA_Tool(query=\"What is the mass of the moon?\", "
                        "url=\"https://en.wikipedia.org/wiki/Moon\")"
                    ),
                    "description": "Answer a question using a Wikipedia page.",
                }
            ],
            user_metadata={
                "api_key_file": KEY_FILENAME,
            },
        )
        model = os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL)
        base_url = os.getenv("LLM_BASE_URL", DEFAULT_LLM_BASE_URL)
        api_key = os.getenv("LLM_API_KEY", DEFAULT_LLM_API_KEY)
        self.llm_engine = LLMEngine(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
            top_p=1.0,
        )
        self.chunk_size = 200
        self.chunk_overlap = 20
        self.top_k = 10
        self.embeddings_model = "text-embedding-3-small"
        self.max_window_size = 1000000
        self._openai_api_key = None
        self._openai_client = None

    def _load_openai_key(self):
        if self._openai_api_key:
            return self._openai_api_key
        if not os.path.exists(KEY_FILE_PATH):
            raise FileNotFoundError(
                f"Missing API key file at {KEY_FILE_PATH}. "
                "Create the file and place the key on a single line."
            )
        with open(KEY_FILE_PATH, "r") as handle:
            api_key = handle.read().strip()
        if not api_key:
            raise ValueError(
                f"Empty API key file at {KEY_FILE_PATH}. "
                "Place the key on a single line."
            )
        self._openai_api_key = api_key
        return api_key

    def _get_openai_client(self):
        if self._openai_client is None:
            api_key = self._load_openai_key()
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def sanity_check(self):
        try:
            self._load_openai_key()
        except Exception as exc:
            return [str(exc)]
        return []

    def _get_website_content(self, url):
        url = url.replace("arxiv.org/pdf", "arxiv.org/abs")
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
        return text[: self.max_window_size]

    def _chunk_website_content(self, content: str) -> List[str]:
        words = content.split()
        ptr = 0
        chunks = []
        while True:
            start, end = ptr, min(ptr + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end >= len(words):
                break
            ptr = end - self.chunk_overlap
        return chunks

    def _embed_strings(self, strings: List[str]) -> List[List[float]]:
        client = self._get_openai_client()
        embeddings = client.embeddings.create(
            input=strings,
            model=self.embeddings_model,
        )
        return [embedding.embedding for embedding in embeddings.data]

    @staticmethod
    def _cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def _rank_chunks(self, query_embedding, chunk_embeddings):
        similarities = [
            self._cosine_similarity(query_embedding, chunk_embedding)
            for chunk_embedding in chunk_embeddings
        ]
        return list(np.argsort(similarities)[::-1])

    @staticmethod
    def _concatenate_chunks(chunks):
        formatted = []
        for i, chunk in enumerate(chunks):
            formatted.append(f"Chunk [{i + 1}]\n{chunk}")
        return "\n".join(formatted)

    def _construct_final_output(self, query, reference_information):
        summary_prompt = SUMMARIZE_PROMPT_TEMPLATE.format(
            query=query,
            reference_information=reference_information,
        )
        return self.llm_engine.generate(summary_prompt)

    def execute(self, query, url):
        if not query:
            raise ValueError("query is required")
        if not url:
            raise ValueError("url is required")

        website_content = self._get_website_content(url)
        if not website_content:
            raise RuntimeError("No content extracted from the URL.")

        chunks = self._chunk_website_content(website_content)
        if not chunks:
            raise RuntimeError("No chunks created from the URL content.")

        embeddings = self._embed_strings([query] + chunks)
        query_embedding = embeddings[0]
        chunk_embeddings = embeddings[1:]

        ranked_chunks = self._rank_chunks(query_embedding, chunk_embeddings)
        top_chunks = [chunks[i] for i in ranked_chunks[: self.top_k]]

        reference_string = self._concatenate_chunks(top_chunks)
        return self._construct_final_output(query, reference_string)
