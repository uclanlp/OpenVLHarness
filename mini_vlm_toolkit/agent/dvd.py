"""
DVD Agent implementation extending BaseAgentWithTools for video question answering.
"""

import copy
import json
from typing import Annotated as A
from typing import Any

from .base import BaseAgentWithTools
from ..smp import get_logger
from ..utils import chat_completion_request

# Import DVD-specific utilities
from concurrent.futures import ThreadPoolExecutor, as_completed


class StopException(Exception):
    """
    Stop Execution by raising this exception (Signal that the task is Finished).
    """


class DVDAgent(BaseAgentWithTools):
    """
    DVD (Dynamic Video Database) Agent extending BaseAgent for video understanding.
    
    This agent uses a ReAct-style loop with function calling to answer video questions.
    It provides tools for frame inspection, clip searching, and global video browsing.
    """

    def __init__(self,
                 video_db_path: str,
                 video_caption_path: str,
                 max_iterations: int = 15,
                 base: str = "http://localhost:8000",
                 key: str = "empty",
                 model: str = None,
                 timeout: int = 600,
                 custom_prompt: bool = False,
                 max_tokens: int = 2048,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 repetition_penalty: float = 1.0,
                 retry: int = 3,
                 verbose: bool = False,
                 **kwargs):
        """
        Initialize the DVD Agent.
        
        Args:
            video_db_path: Path to the video database JSON file
            video_caption_path: Path to the video caption file
            max_iterations: Maximum number of iterations for the agent loop
            base: Base URL for the LLM API
            key: API key for the LLM
            model: Model name to use
            timeout: Request timeout in seconds
            custom_prompt: Whether to use custom prompts
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            retry: Number of retries for API calls
            verbose: Whether to print verbose logs
            **kwargs: Additional arguments
        """
        # Initialize parent BaseAgent
        super().__init__(
            base=base,
            key=key,
            model=model,
            timeout=timeout,
            custom_prompt=custom_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            retry=retry,
            verbose=verbose,
            **kwargs
        )

        self.logger = get_logger("DVDAgent")
        self.video_db_path = video_db_path
        self.video_caption_path = video_caption_path
        self.max_iterations = max_iterations

        # Initialize DVD-specific tools
        self.tools = [frame_inspect_tool, clip_search_tool, global_browse_tool, finish]
        if config.LITE_MODE:
            self.tools.remove(frame_inspect_tool)

        self.name_to_function_map = {tool.__name__: tool for tool in self.tools}
        self.function_schemas = [
            {"function": as_json_schema(func), "type": "function"}
            for func in self.name_to_function_map.values()
        ]

        # Initialize video database
        self.video_db = init_single_video_db(
            video_caption_path,
            video_db_path,
            config.AOAI_EMBEDDING_LARGE_DIM
        )
        self.logger.info(f"DVD Agent initialized with video database from {video_db_path}")

    def _construct_system_message(self) -> str:
        """Construct the system message for the DVD agent."""
        return """You are a helpful assistant who answers multi-step questions by sequentially invoking functions. Follow the THINK → ACT → OBSERVE loop:
  • THOUGHT Reason step-by-step about which function to call next.
  • ACTION   Call exactly one function that moves you closer to the final answer.
  • OBSERVATION Summarize the function's output.
You MUST plan extensively before each function call, and reflect extensively on the outcomes of the previous function calls.
Only pass arguments that come verbatim from the user or from earlier function outputs—never invent them. Continue the loop until the user's query is fully resolved, then end your turn with the final answer. If you are uncertain about code structure or video content, use the available tools to inspect rather than guessing. Plan carefully before each call and reflect on every result. Do not rely solely on blind function calls, as this degrades reasoning quality. Timestamps may be formatted as 'HH:MM:SS' or 'MM:SS'."""

    def _construct_user_prompt(self, question: str) -> str:
        """Construct the user prompt for a given question."""
        video_length = self.video_db.get_additional_data()['video_length']
        return f"""Carefully read the timestamps and narration in the following script, paying attention to the causal order of events, object details and movements, and people's actions and poses.

Here are tools you can use to reveal your reasoning process whenever the provided information is insufficient.

• To get a global information about events and main subjects in the video, use `global_browse_tool`.
• To search without a specific timestamp, use `clip_search_tool`.
• If the retrieved material lacks precise, question-relevant detail (e.g., an unknown name), call `frame_inspect_tool` with a list of time ranges (list[tuple[HH:MM:SS, HH:MM:SS]]).
• Whenever you are uncertain of an answer after searching, inspect frames in the relevant intervals with `frame_inspect_tool`.
• After locating an answer in the script, always make a **CONFIRM** with `frame_inspect_tool` query.


You can first use `global_browse_tool` to a global information about this video, then invoke multiple times of these tools to prgressively find the answer.

Based on your observations and tool outputs, provide a concise answer that directly addresses the question.

Total video length: {video_length} seconds.

Question: {question}"""

    def _append_tool_msg(self, tool_call_id: str, name: str, content: str, msgs: list) -> None:
        """Append a tool response message to the message list."""
        msgs.append({
            "tool_call_id": tool_call_id,
            "role": "tool",
            "name": name,
            "content": content,
        })

    def _exec_tool(self, tool_call: dict, msgs: list) -> None:
        """Execute a single tool call and append the result to messages."""
        name = tool_call["function"]["name"]
        if name not in self.name_to_function_map:
            self._append_tool_msg(tool_call["id"], name, f"Invalid function name: {name!r}", msgs)
            return

        # Parse arguments
        try:
            args = json.loads(tool_call["function"]["arguments"])
        except json.JSONDecodeError as exc:
            raise StopException(f"Error decoding arguments: {exc!s}")

        if "database" in args:
            args["database"] = self.video_db

        if "topk" in args:
            if config.OVERWRITE_CLIP_SEARCH_TOPK > 0:
                args["topk"] = config.OVERWRITE_CLIP_SEARCH_TOPK

        # Call the tool
        try:
            if self.verbose:
                self.logger.info(f"Calling function `{name}` with args: {args}")
            result = self.name_to_function_map[name](**args)
            self._append_tool_msg(tool_call["id"], name, result, msgs)
        except StopException as exc:  # graceful stop
            if self.verbose:
                self.logger.info(f"Finish task with message: '{exc!s}'")
            raise
        except Exception as e:
            self.logger.error(f"Error in tool call: {e}\nTool call: {tool_call}")
            raise

    def _call_llm_with_tools(self, msgs: list) -> dict:
        """Call the LLM with tool support."""
        if "qwen" in config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME.lower():
            response = call_qwen_sglang_with_tools(
                msgs,
                local_port=config.QWEN_MODEL_PORT,
                tools=self.function_schemas,
                temperature=self.temperature,
            )
        else:
            response = call_openai_model_with_tools(
                msgs,
                endpoints=config.AOAI_ORCHESTRATOR_LLM_ENDPOINT_LIST,
                model_name=config.AOAI_ORCHESTRATOR_LLM_MODEL_NAME,
                tools=self.function_schemas,
                temperature=self.temperature,
                api_key=config.OPENAI_API_KEY,
            )
        return response

    def run(self, message, **kwargs1) -> str:
        """
        Override the BaseAgent run method to implement DVD agent logic.
        
        This method processes a video question using the ReAct-style loop with function calling.
        
        Args:
            message: The user's question (string)
            **kwargs1: Additional arguments (not used in DVD agent)
        
        Returns:
            str: The agent's answer or error message
        """
        if not isinstance(message, str):
            self.logger.error(f"DVD Agent expects a string question, got {type(message)}")
            return "Error: Invalid input. Please provide a string question."

        # Construct initial messages
        msgs = [
            {
                "role": "system",
                "content": self._construct_system_message()
            },
            {
                "role": "user",
                "content": self._construct_user_prompt(message)
            }
        ]

        # Main ReAct loop
        for i in range(self.max_iterations):
            # Force a final `finish` on the last iteration to avoid hanging
            if i == self.max_iterations - 1:
                msgs.append({
                    "role": "user",
                    "content": "Please call the `finish` function to finish the task.",
                })

            # Call LLM with tools
            try:
                response = self._call_llm_with_tools(msgs)

                if response is None:
                    self.logger.error("LLM returned None response")
                    return self.fail_msg

                response.setdefault("role", "assistant")
                msgs.append(response)

                # Execute any requested tool calls
                try:
                    for tool_call in response.get("tool_calls", []):
                        self._exec_tool(tool_call, msgs)
                except StopException as exc:
                    # Task finished successfully
                    if self.verbose:
                        self.logger.info(f"DVD Agent finished with answer: {exc}")
                    return str(exc)

            except Exception as err:
                self.logger.error(f"Error during iteration {i}: {type(err).__name__}: {err}")
                if self.verbose:
                    self.logger.error(f"Full error: {err}", exc_info=True)

        # If we reach here, max iterations exceeded
        self.logger.warning(f"DVD Agent reached maximum iterations ({self.max_iterations})")
        return self.fail_msg + ": Maximum iterations reached without finishing."

    def chat(self, messages, **kwargs1) -> str:
        """
        Multi-turn chat is not directly supported by DVD Agent.
        Use run() method instead for single-turn video question answering.
        """
        self.logger.warning("DVD Agent does not support multi-turn chat. Use run() method instead.")
        if messages and isinstance(messages[-1], dict) and "content" in messages[-1]:
            # Extract the question from the last user message
            content = messages[-1]["content"]
            if isinstance(content, str):
                return self.run(content, **kwargs1)
        return self.fail_msg

    def stream_run(self, question: str):
        """
        Generator version of run() for streaming results.
        
        Yields:
            dict: Every assistant/tool message produced during reasoning.
        """
        if not isinstance(question, str):
            yield {"role": "error", "content": "Invalid input: expected string question"}
            return

        # Construct initial messages
        msgs = [
            {
                "role": "system",
                "content": self._construct_system_message()
            },
            {
                "role": "user",
                "content": self._construct_user_prompt(question)
            }
        ]

        # Main ReAct loop
        for i in range(self.max_iterations):
            # Force a final `finish` on the last iteration
            if i == self.max_iterations - 1:
                final_usr_msg = {
                    "role": "user",
                    "content": "Please call the `finish` function to finish the task.",
                }
                msgs.append(final_usr_msg)

            # Call LLM with tools
            try:
                response = self._call_llm_with_tools(msgs)

                if response is None:
                    yield {"role": "error", "content": "LLM returned None response"}
                    return

                response.setdefault("role", "assistant")
                msgs.append(response)
                yield response  # stream assistant reply

                # Execute any requested tool calls
                try:
                    for tool_call in response.get("tool_calls", []):
                        # Yield tool call information
                        tool_name = tool_call.get("function", {}).get("name", "unknown")
                        tool_args = tool_call.get("function", {}).get("arguments", "{}")
                        yield {
                            "role": "tool_call",
                            "name": tool_name,
                            "arguments": tool_args
                        }

                        self._exec_tool(tool_call, msgs)
                        # Yield the tool result message
                        if msgs[-1].get("role") == "tool":
                            yield msgs[-1]  # stream tool observation
                except StopException:
                    return

            except Exception as err:
                self.logger.error(f"Error during streaming iteration {i}: {err}")
                yield {"role": "error", "content": f"Error: {err}"}
                return

    def parallel_run(self, questions: list[str], max_workers: int = 4) -> list[str]:
        """
        Run multiple questions in parallel.
        
        Args:
            questions: List of questions to answer
            max_workers: Maximum number of parallel workers
        
        Returns:
            list[str]: List of answers corresponding to input questions
        """
        results = [None] * len(questions)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.run, q): idx
                for idx, q in enumerate(questions)
            }
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    self.logger.error(f"Error processing question {idx}: {e}")
                    results[idx] = self.fail_msg

        return results
