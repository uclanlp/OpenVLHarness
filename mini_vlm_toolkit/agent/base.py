import os
import requests
import numpy as np
import json
import time
import random as rd
import copy as cp
from .utils import _get_json, pretty_print_msgs
from .constant import PROMPT_REGISTRY
from ..memory import PersistentStateMemory, ImageStateMemory, ToolContextMemory
from ..utils.api_utils import (
    chat_completion_request,
    filter_kwargs_for_endpoint,
    tool_execute_request,
)
from typing import Any, Dict, List, Optional
import re

try:
    from ..smp import get_logger, encode_image_to_base64, parse_file
except Exception:  # pragma: no cover - optional heavy deps may be missing for lightweight usage/tests.
    import base64
    import hashlib
    import logging
    import mimetypes
    import tempfile
    from io import BytesIO
    from urllib.parse import urlparse

    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        return logger

    def encode_image_to_base64(img):
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _fallback_decode_data_url_to_file(data_url: str):
        if not isinstance(data_url, str) or not data_url.startswith("data:"):
            return None, data_url
        header, sep, payload = data_url.partition(",")
        if not sep:
            return None, data_url

        mime = header[5:].split(";", 1)[0].strip().lower()
        if "/" not in mime:
            return None, data_url

        try:
            if ";base64" in header.lower():
                binary = base64.b64decode(payload)
            else:
                binary = payload.encode("utf-8")
        except Exception:
            return None, data_url

        ext = mimetypes.guess_extension(mime) or ".bin"
        if ext == ".jpe":
            ext = ".jpg"
        digest = hashlib.md5(payload.encode("utf-8")).hexdigest()
        cache_dir = os.path.join(tempfile.gettempdir(), "mini_vlm_toolkit_fallback_files")
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, f"{digest}{ext}")
        if not os.path.isfile(local_path):
            with open(local_path, "wb") as fout:
                fout.write(binary)
        return mime, local_path

    def parse_file(value):
        if not isinstance(value, str):
            return None, value

        s = value.strip()
        if not s:
            return None, value

        if os.path.exists(s) and s != ".":
            if not os.path.isfile(s):
                return None, value
            suffix = os.path.splitext(s)[1].lower()
            if suffix == ".webp":
                return ("image/webp", s)
            mime = mimetypes.types_map.get(suffix, "unknown")
            return (mime, s)

        if s.startswith("data:image/"):
            mime, local_path = _fallback_decode_data_url_to_file(s)
            return mime, local_path

        parsed = urlparse(s)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            suffix = os.path.splitext(parsed.path)[1].lower()
            if suffix == ".webp":
                return ("image/webp", s)
            mime = mimetypes.types_map.get(suffix)
            if mime:
                return (mime, s)
            return ("url", s)

        return None, value

class BaseAgent:

    allowed_types = ['text', 'image', 'video']

    def __init__(self, 
                 base: str = "http://localhost:8000",
                 key: str = "empty",
                 model: str = None,
                 timeout: int = 600,
                 custom_prompt: bool = False,
                 max_tokens: int = 2048,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 repetition_penalty: float = 1.0,
                 reasoning_effort: Optional[str] = None,
                 retry: int = 3,
                 verbose: bool = False,
                 **kwargs):

        self.base = base
        self.key = key
        self.model = model
        self.timeout = timeout
        self.api_base = os.path.join(self.base, "v1/chat/completions")
        self.logger = get_logger("BaseAgent")
        self.custom_prompt = custom_prompt
        self.system_prompt = None
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.reasoning_effort = reasoning_effort
        self.retry = retry
        self.verbose = verbose
        self.wait = 1.0  # wait time between retries
        self.fail_msg = "Error: Unable to get response from the server."
        self.default_kwargs = kwargs
        self.default_kwargs.update({
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        })
        if self.reasoning_effort is not None:
            self.default_kwargs["reasoning_effort"] = self.reasoning_effort

        # check if server is reachable
        models_url = f"{self.base.rstrip('/')}/v1/models"
        health_headers = {"Content-Type": "application/json"}
        key_text = str(self.key or "").strip()
        if key_text and key_text.lower() not in {"empty", "none", "null"}:
            health_headers["Authorization"] = f"Bearer {key_text}"
        try:
            response = requests.get(models_url, headers=health_headers, timeout=20)
        except Exception as exc:
            raise ConnectionError(
                f"Unable to connect to the server at {self.base}. "
                f"Health check request failed: {exc}"
            ) from exc
        if response.status_code != 200:
            detail = (response.text or "").strip().replace("\n", " ")
            if len(detail) > 240:
                detail = f"{detail[:240]}..."
            raise ConnectionError(
                f"Unable to connect to the server at {self.base}. "
                f"GET /v1/models returned {response.status_code}. {detail}"
            )
        
        self.logger.info(f"Evaluating with model: {self.model}")
        self.logger.info(f"Sampling params: max_tokens={self.max_tokens}, temperature={self.temperature}, top_p={self.top_p}, repetition_penalty={self.repetition_penalty}")


    def build_prompt(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method.")

    def prepare_itlist(self, inputs):
        assert np.all([isinstance(x, dict) for x in inputs])
        has_images = np.sum([x['type'] == 'image' for x in inputs])
        if has_images:
            content_list = []
            for msg in inputs:
                if msg['type'] == 'text':
                    content_list.append(dict(type='text', text=msg['value']))
                elif msg['type'] == 'image':
                    from PIL import Image
                    img = Image.open(msg['value'])
                    b64 = encode_image_to_base64(img)
                    extra_args = msg.copy()
                    extra_args.pop('type')
                    extra_args.pop('value')
                    img_struct = dict(url=f'data:image/jpeg;base64,{b64}', **extra_args)
                    content_list.append(dict(type='image_url', image_url=img_struct))
        else:
            assert all([x['type'] == 'text' for x in inputs])
            text = '\n'.join([x['value'] for x in inputs])
            content_list = [dict(type='text', text=text)]
        return content_list

    def prepare_inputs(self, inputs, system_prompt=None):
        input_msgs = []
        active_system_prompt = self.system_prompt if system_prompt is None else system_prompt
        if active_system_prompt is not None:
            input_msgs.append(dict(role='system', content=active_system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, system_prompt=None, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs, system_prompt=system_prompt)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        top_p = kwargs.pop('top_p', self.top_p)
        repetition_penalty = kwargs.pop('repetition_penalty', self.repetition_penalty)
        endpoint_hint = f"{getattr(self, 'base', '')} {getattr(self, 'api_base', '')}".lower()
        is_openai_official = "api.openai.com" in endpoint_hint
        # For OpenAI official endpoint, strip internal orchestration-only field.
        kwargs.pop('dataset', None)                                                                                                                                                                         
        kwargs.pop('task_id', None)

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            n=1,
            **kwargs)
        # OpenAI official chat/completions endpoint rejects repetition_penalty.
        if not is_openai_official:
            payload["repetition_penalty"] = repetition_penalty
        response = requests.post(
            self.api_base,
            headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
        ret_code = response.status_code
        ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
        answer = self.fail_msg
        try:
            resp_struct = json.loads(response.text)
            answer = resp_struct['choices'][0]['message']['content'].strip()

        except:
            pass
        return ret_code, answer, response
    
    def check_content(self, msgs):
        """Check the content type of the input. Four types are allowed: str, dict, liststr, listdict.

        Args:
            msgs: Raw input messages.

        Returns:
            str: The message type.
        """
        if isinstance(msgs, str):
            return 'str'
        if isinstance(msgs, dict):
            return 'dict'
        if isinstance(msgs, list):
            types = [self.check_content(m) for m in msgs]
            if all(t == 'str' for t in types):
                return 'liststr'
            if all(t == 'dict' for t in types):
                return 'listdict'
        return 'unknown'

    def preproc_content(self, inputs):
        """Convert the raw input messages to a list of dicts.

        Args:
            inputs: raw input messages.

        Returns:
            list(dict): The preprocessed input messages. Will return None if failed to preprocess the input.
        """
        if self.check_content(inputs) == 'str':
            return [dict(type='text', value=inputs)]
        elif self.check_content(inputs) == 'dict':
            assert 'type' in inputs and 'value' in inputs
            return [inputs]
        elif self.check_content(inputs) == 'liststr':
            res = []
            for s in inputs:
                mime, pth = parse_file(s)
                if mime is None or mime == 'unknown':
                    res.append(dict(type='text', value=s))
                else:
                    res.append(dict(type=mime.split('/')[0], value=pth))
            return res
        elif self.check_content(inputs) == 'listdict':
            for item in inputs:
                assert 'type' in item and 'value' in item
                mime, s = parse_file(item['value'])
                if mime is None:
                    assert item['type'] == 'text', item['value']
                else:
                    assert mime.split('/')[0] == item['type']
                    item['value'] = s
            return inputs
        else:
            return None

    # May exceed the context windows size, so try with different turn numbers.
    def chat_inner(self, inputs, **kwargs):
        _ = kwargs.pop('dataset', None)
        while len(inputs):
            try:
                return self.generate_inner(inputs, **kwargs)
            except Exception as e:
                if self.verbose:
                    self.logger.info(f'{type(e)}: {e}')
                inputs = inputs[1:]
                while len(inputs) and inputs[0]['role'] != 'user':
                    inputs = inputs[1:]
                continue
        return -1, self.fail_msg + ': ' + 'Failed with all possible conversation turns.', None

    def chat(self, messages, **kwargs1):
        """The main function for multi-turn chatting. Will call `chat_inner` with the preprocessed input messages."""
        assert hasattr(self, 'chat_inner'), 'The API model should has the `chat_inner` method. '
        messages = cp.deepcopy(messages)
        for msg in messages:
            assert isinstance(msg, dict) and 'role' in msg and 'content' in msg, msg
            assert self.check_content(msg['content']) in ['str', 'dict', 'liststr', 'listdict'], msg
            msg['content'] = self.preproc_content(msg['content'])
        # merge kwargs
        kwargs = cp.deepcopy(self.default_kwargs)
        kwargs.update(kwargs1)

        answer = None
        # a very small random delay [0s - 0.5s]
        T = rd.random() * 0.5
        time.sleep(T)

        assert messages[-1]['role'] == 'user'

        for i in range(self.retry):
            try:
                ret_code, answer, log = self.chat_inner(cp.deepcopy(messages), **kwargs)
                if ret_code == 0 and self.fail_msg not in answer and answer != '':
                    if self.verbose:
                        print(answer)
                    return answer
                elif self.verbose:
                    if not isinstance(log, str):
                        try:
                            log = log.text
                        except Exception as e:
                            self.logger.warning(f'Failed to parse {log} as an http response: {str(e)}. ')
                    self.logger.info(f'RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}')
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'An error occured during try {i}: ')
                    self.logger.error(f'{type(err)}: {err}')
            # delay before each retry
            T = rd.random() * self.wait * 2
            time.sleep(T)

        return self.fail_msg if answer in ['', None] else answer

    def preprocess_message_with_role(self, message, *, base_system_prompt=None, return_system_prompt=False):
        system_prompt = ''
        new_message = []

        for data in message:
            assert isinstance(data, dict)
            role = data.get('role', 'user')
            if role == 'system':
                system_prompt += data.get('value', '') + '\n'
            else:
                new_item = cp.deepcopy(data)
                new_item.pop('role', None)
                new_message.append(new_item)

        resolved_system_prompt = self.system_prompt if base_system_prompt is None else base_system_prompt
        if system_prompt != '':
            if resolved_system_prompt is None:
                resolved_system_prompt = system_prompt
            else:
                if system_prompt not in resolved_system_prompt:
                    resolved_system_prompt += '\n' + system_prompt
        if return_system_prompt:
            return new_message, resolved_system_prompt
        return new_message

    def _build_trajectory_record(self, input_messages, answer, processed_input_msgs=None):
        record = {
            "input_messages": input_messages,
            "final_answer": answer,
        }
        if processed_input_msgs is not None:
            record["messages"] = list(processed_input_msgs) + [
                {"role": "assistant", "content": answer}
            ]
        return record

    def generate(self, message, **kwargs1):
        """The main function to generate the answer. Will call `generate_inner` with the preprocessed input messages.

        Args:
            message: raw input messages.

        Returns:
            str: The generated answer of the Failed Message if failed to obtain answer.
        """
        raw_message = cp.deepcopy(message)
        message = cp.deepcopy(message)
        active_system_prompt = self.system_prompt
        if self.check_content(message) == 'listdict':
            message, active_system_prompt = self.preprocess_message_with_role(
                message,
                base_system_prompt=self.system_prompt,
                return_system_prompt=True,
            )
        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        message = self.preproc_content(message)
        assert message is not None and self.check_content(message) == 'listdict'
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'
        # merge kwargs
        kwargs = cp.deepcopy(self.default_kwargs)
        kwargs.update(kwargs1)

        processed_input_msgs = self.prepare_inputs(
            cp.deepcopy(message),
            system_prompt=active_system_prompt,
        )

        answer = None
        last_log = None
        last_ret_code = None
        # a very small random delay [0s - 0.5s]
        T = rd.random() * 0.5
        time.sleep(T)

        for i in range(self.retry):
            try:
                ret_code, answer, log = self.generate_inner(
                    cp.deepcopy(message),
                    system_prompt=active_system_prompt,
                    **kwargs,
                )
                last_log = log
                last_ret_code = ret_code
                if ret_code == 0 and self.fail_msg not in answer and answer != '':
                    if self.verbose:
                        print(answer)
                    # Return both answer and extra_records
                    # Add for agent evaluation
                    trajectory = self._build_trajectory_record(
                        raw_message,
                        answer,
                        processed_input_msgs,
                    )
                    extra_records = {"trajectory": trajectory}
                    if isinstance(log, dict):
                        extra_records.update(log)
                    try:
                        reasoning = json.loads(log.text)["choices"][0]["message"].get("reasoning")
                        if reasoning:
                            extra_records["reasoning"] = reasoning
                    except Exception:
                        pass
                    return {"prediction": answer, "extra_records": extra_records}
                elif self.verbose:
                    if not isinstance(log, str):
                        try:
                            log = log.text
                        except Exception as e:
                            self.logger.warning(f'Failed to parse {log} as an http response: {str(e)}. ')
                    self.logger.info(f'RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}')
            except Exception as err:
                if self.verbose:
                    self.logger.error(f'An error occured during try {i}: ')
                    self.logger.error(f'{type(err)}: {err}')
            # delay before each retry
            T = rd.random() * self.wait * 2
            time.sleep(T)

        final_answer = self.fail_msg if answer in ['', None] else answer
        trajectory = self._build_trajectory_record(
            raw_message,
            final_answer,
            processed_input_msgs,
        )
        if last_ret_code is not None:
            trajectory["ret_code"] = last_ret_code
        return {"prediction": final_answer, "extra_records": {"trajectory": trajectory}}
    

class BaseAgentWithTools(BaseAgent):
    _ALLOWED_PROMPT_VERSIONS = ("v1", "v2", "v3", "v4", "v5", "v1_instruct", "v1_thinking", "v2_instruct", "v2_thinking", "v3_instruct")
    _LEGACY_RUNTIME_ENV_KEYS = ("runtime_environment_hooks", "runtime_environment_emit_policy")
    _SUPPORTED_MEMORY_NAMES = ("image", "tool_context")

    def __init__(self,
                tool_base,
                tool_list = None,
                tool_choice: Any = "auto",
                prompt_version: str = "v1",
                persistent_state_memory: Optional[Dict[str, Any]] = None,
                max_iterations: int = 15,
                experience_file: Optional[str] = None,
                instance_critique_dir: Optional[str] = None,
                max_context_length: Optional[int] = None,
                **kwargs):
        legacy_keys = [key for key in self._LEGACY_RUNTIME_ENV_KEYS if key in kwargs]
        if legacy_keys:
            raise ValueError(
                "Legacy runtime environment keys are no longer supported: "
                f"{legacy_keys}. Use 'persistent_state_memory' instead."
            )
        super().__init__(**kwargs)
        self.tool_base = tool_base
        self.prompt_version = str(prompt_version).strip().lower()
        if self.prompt_version not in self._ALLOWED_PROMPT_VERSIONS:
            raise ValueError(
                f"Invalid prompt_version '{prompt_version}'. "
                f"Expected one of: {self._ALLOWED_PROMPT_VERSIONS}."
            )
        self.tools = _get_json(self.tool_base, "/tools")
        if not isinstance(self.tools, dict):
            raise ConnectionError(f"Failed to get tools from {self.tool_base}/tools")
        self.json_schema = _get_json(self.tool_base, "/jsonschema")
        if not isinstance(self.json_schema, list):
            raise ConnectionError(f"Failed to get tool schemas from {self.tool_base}/jsonschema")
        self.tool_choice = tool_choice
        self.max_iterations = max_iterations
        self.max_context_length = max_context_length

        if tool_list is not None:
            self.tools = {name: tool for name, tool in self.tools.items() if name in tool_list}
            self.json_schema = [
                x for x in self.json_schema
                if isinstance(x, dict)
                and isinstance(x.get('function'), dict)
                and x['function'].get('name') in tool_list
            ]
            
        self.experience_file = experience_file
        self.instance_critique_dir = instance_critique_dir
        self.logger = get_logger("BaseAgentWithTools")
        self.logger.info(f"Available Tools: {list(self.tools.keys())}")
        self.system_prompt = self._construct_system_message()
        self._persistent_state_memory_spec = cp.deepcopy(persistent_state_memory)
        self.persistent_state_memory = self._build_persistent_state_memory(persistent_state_memory)
        self.persistent_state_memory.configure_tool_schema(self.json_schema)
        self.logger.info(
            "PersistentStateMemory: "
            f"{self.persistent_state_memory.__class__.__name__} "
            f"(name={self.persistent_state_memory.memory_name})"
        )

        # Eagerly load tokenizer once per agent instance so _count_tokens and
        # _build_raw_sft_from_tokens share it. Prefer the local cache to avoid
        # hammering the Hub API under concurrent workers (429s). Fall back to
        # a network load on the first run (or when the cache is missing).
        self._tokenizer = None
        if self.model:
            from transformers import AutoTokenizer
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model, trust_remote_code=True, local_files_only=True
                )
            except Exception:
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        self.model, trust_remote_code=True
                    )
                except Exception as e:
                    self.logger.warning(f"Could not load tokenizer for {self.model}: {e}")

    def _load_instance_critique(self, task_id: str) -> Optional[str]:
        """Load a per-instance critique file if it exists."""
        if not self.instance_critique_dir:
            return None
        fpath = os.path.join(self.instance_critique_dir, f"{task_id}.txt")
        if not os.path.isfile(fpath):
            return None
        with open(fpath, "r", encoding="utf-8") as f:
            return f.read().strip() or None

    def _get_prompt_templates(self) -> tuple[str, str]:
        version = str(getattr(self, "prompt_version", "v1")).strip().lower()
        return PROMPT_REGISTRY[version]

    def _construct_system_message(self) -> str:
        system_prompt, _ = self._get_prompt_templates()
        if getattr(self, "experience_file", None) and os.path.exists(self.experience_file):
            from ..experience.reflector import load_experience
            experience_text = load_experience(self.experience_file)
            if experience_text.strip():
                system_prompt += (
                    "\n\nAdditional learned tool-usage principles from prior runs:\n"
                    "- Treat these as reusable heuristics when choosing tools, arguments, and retries.\n"
                    "- Follow them when relevant, but do not violate tool protocols above.\n"
                    "<learned_principles>\n"
                    f"{experience_text}\n"
                    "</learned_principles>"
                )
                self.logger.info(f"Loaded experience from {self.experience_file}")
        return system_prompt

    def _build_persistent_state_memory(self, memory_spec: Optional[Dict[str, Any]]) -> PersistentStateMemory:
        spec = memory_spec
        if spec is None:
            spec = {"name": "image", "emit_policy": "always"}
        if not isinstance(spec, dict):
            raise ValueError(
                "persistent_state_memory must be null or a dict config, "
                "for example {'name': 'image', 'emit_policy': 'always'}."
            )
        if "name" not in spec:
            raise ValueError("persistent_state_memory config must include a string 'name' field.")

        name = spec.get("name")
        if not isinstance(name, str):
            raise ValueError("persistent_state_memory.name must be a string.")
        memory_name = name.strip().lower()

        if memory_name in {"", "none", "off", "disabled"}:
            return PersistentStateMemory()

        options = {key: value for key, value in spec.items() if key != "name"}
        if memory_name == "image":
            return ImageStateMemory(**options)
        if memory_name == "tool_context":
            return ToolContextMemory(**options)

        raise ValueError(
            f"Unknown persistent_state_memory '{name}'. "
            f"Supported memories: {list(self._SUPPORTED_MEMORY_NAMES)}."
        )

    def _new_persistent_state_memory(self) -> PersistentStateMemory:
        """Create a fresh per-run memory object.

        Memory instances are never shared across runs to guarantee stateless
        agent behavior under threaded benchmark execution.
        """
        # Unit tests that bypass __init__ may inject a custom factory.
        factory = getattr(self, "_persistent_state_memory_factory", None)
        if callable(factory):
            memory = factory()
            if not isinstance(memory, PersistentStateMemory):
                raise TypeError(
                    "persistent state memory factory must return PersistentStateMemory, "
                    f"got {type(memory).__name__}."
                )
            return memory

        sentinel = object()
        spec = getattr(self, "_persistent_state_memory_spec", sentinel)
        if spec is not sentinel:
            if isinstance(spec, dict) or spec is None:
                return self._build_persistent_state_memory(cp.deepcopy(spec))
            raise TypeError(
                "persistent state memory spec must be dict or null, "
                f"got {type(spec).__name__}."
            )

        # Backward-compatible fallback for tests/custom setups that inject a
        # prepared memory instance directly on the agent object.
        injected = getattr(self, "persistent_state_memory", None)
        if isinstance(injected, PersistentStateMemory):
            return injected
        raise RuntimeError("Unable to create PersistentStateMemory for this run.")

    def _append_messages_with_indices(
        self,
        msgs: List[Dict[str, Any]],
        produced: Any,
    ) -> List[int]:
        if isinstance(produced, dict):
            produced = [produced]
        if not isinstance(produced, list):
            return []

        indices = []
        for message in produced:
            if not isinstance(message, dict):
                continue
            if not isinstance(message.get("role"), str):
                continue
            msgs.append(message)
            indices.append(len(msgs) - 1)
        return indices

    def _summarize_message_for_trace(self, message: Dict[str, Any], max_chars: int = 180) -> Dict[str, Any]:
        role = message.get("role")
        content = message.get("content")
        summary: Dict[str, Any] = {"role": role}
        if isinstance(content, str):
            text = content.replace("\n", "\\n")
            if len(text) > max_chars:
                text = f"{text[:max_chars]}... (truncated)"
            summary.update({"content_type": "text", "content_preview": text})
            return summary
        if isinstance(content, list):
            part_types = [
                part.get("type")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("type"), str)
            ]
            text_part = next(
                (
                    part.get("text")
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") == "text"
                    and isinstance(part.get("text"), str)
                ),
                "",
            )
            text_preview = text_part.replace("\n", "\\n")
            if len(text_preview) > max_chars:
                text_preview = f"{text_preview[:max_chars]}... (truncated)"
            summary.update(
                {
                    "content_type": "multimodal",
                    "part_types": part_types,
                    "image_count": sum(1 for part_type in part_types if part_type == "image_url"),
                    "text_preview": text_preview,
                }
            )
            return summary
        summary["content_type"] = type(content).__name__
        return summary

    def _add_user_prompt_prefix(self, msgs) -> str:
        """Apply the default user prompt scaffold to the latest user text turn."""
        if not msgs or msgs[-1].get("role") != "user":
            return msgs
        _, user_prompt_prefix = self._get_prompt_templates()

        last_content = msgs[-1].get("content")
        if isinstance(last_content, list):
            for item in last_content:
                if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                    item["text"] = user_prompt_prefix.format(user_query=item["text"])
                    break
        elif isinstance(last_content, str):
            msgs[-1]["content"] = user_prompt_prefix.format(user_query=last_content)
        elif isinstance(last_content, dict) and "text" in last_content:
            last_content["text"] = user_prompt_prefix.format(user_query=last_content["text"])
        return msgs
    
    def _extract_tagged_content(self, text: str, tag: str) -> str:
        """Extract content between XML-style tags."""
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse assistant content sections used for debugging/reporting.

        Args:
            response: Model response text

        Returns:
            Dictionary with:
                - perception: str (what the agent observes)
                - thinking: str (agent's reasoning)
                - raw: str (original response)
        """
        return {
            "perception": self._extract_tagged_content(response, "perception"),
            "thinking": self._extract_tagged_content(response, "thinking"),
            "raw": response,
        }

    def _normalize_tool_calls(self, tool_calls, turn_idx: int) -> List[Dict[str, Any]]:
        """Normalize native OpenAI `tool_calls` into a stable internal format.

        Parsing failures are preserved as structured `parse_error` payloads so
        the model can receive them as tool errors rather than hard failures.
        """
        normalized = []
        if not isinstance(tool_calls, list):
            return normalized

        for idx, raw_call in enumerate(tool_calls):
            if not isinstance(raw_call, dict):
                continue
            fn = raw_call.get("function") or {}
            tool_name = fn.get("name")
            if not tool_name:
                continue

            call_id = raw_call.get("id") or f"call_{turn_idx + 1}_{idx + 1}"
            raw_args = fn.get("arguments", "{}")
            args_text = "{}"
            args_dict = {}
            parse_error = None

            if isinstance(raw_args, dict):
                args_dict = raw_args
                args_text = json.dumps(raw_args, ensure_ascii=False)
            elif isinstance(raw_args, str):
                args_text = raw_args
                if raw_args.strip():
                    try:
                        parsed_args = json.loads(raw_args)
                    except Exception as exc:
                        parsed_args = {}
                        parse_error = {
                            "code": "invalid_tool_arguments_json",
                            "message": "Tool call arguments are not valid JSON.",
                            "details": {
                                "raw_arguments": raw_args,
                                "exception": str(exc),
                            },
                        }
                    if isinstance(parsed_args, dict):
                        args_dict = parsed_args
                    elif parse_error is None:
                        parse_error = {
                            "code": "invalid_tool_arguments_type",
                            "message": "Tool call arguments JSON must decode to an object.",
                            "details": {"decoded_type": type(parsed_args).__name__},
                        }
            else:
                args_text = "{}"
                args_dict = {}
                parse_error = {
                    "code": "invalid_tool_arguments_type",
                    "message": "Tool call arguments must be a JSON object or a JSON-encoded string.",
                    "details": {"raw_type": type(raw_args).__name__},
                }

            normalized.append(
                {
                    "id": call_id,
                    "name": tool_name,
                    "raw_arguments": args_text,
                    "args": args_dict,
                    "assistant_tool_call": {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": tool_name, "arguments": args_text},
                    },
                    "parse_error": parse_error,
                }
            )
        return normalized

    def _enforce_single_tool_call_per_message(
        self,
        tool_calls: List[Dict[str, Any]],
        *,
        turn_idx: int,
        phase: str,
    ) -> List[Dict[str, Any]]:
        """Reject multi-call assistant messages by converting them into tool errors.

        We preserve all emitted `tool_call_id`s so the model receives protocol-
        consistent tool-result feedback messages, but no actual tool execution
        happens.
        """
        if not isinstance(tool_calls, list) or len(tool_calls) <= 1:
            return tool_calls

        tool_names = [str((call or {}).get("name") or "unknown") for call in tool_calls]
        self.logger.warning(
            "Policy violation: multiple tool calls in one assistant message are not allowed. "
            "turn=%s phase=%s tool_call_count=%s tool_names=%s",
            turn_idx + 1,
            phase,
            len(tool_calls),
            tool_names,
        )

        total_calls = len(tool_calls)
        for idx, call in enumerate(tool_calls, 1):
            if not isinstance(call, dict):
                continue
            original_error = call.get("parse_error")
            details = {
                "phase": phase,
                "turn": turn_idx + 1,
                "tool_call_count": total_calls,
                "tool_call_index": idx,
                "tool_names": tool_names,
            }
            if isinstance(original_error, dict):
                details["suppressed_error"] = cp.deepcopy(original_error)
            call["parse_error"] = {
                "code": "multiple_tool_calls_not_allowed",
                "message": (
                    "Multiple tool calls in one assistant message are not allowed. "
                    "Call at most one tool per turn."
                ),
                "details": details,
            }
        return tool_calls

    def _coerce_ok_flag(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in {"true", "1", "yes"}:
                return True
            if lower in {"false", "0", "no", ""}:
                return False
        return bool(value)

    def _normalize_execute_payload(self, payload, tool_name: str, tool_call_id: str) -> Dict[str, Any]:
        """Normalize `/execute` responses into the agent envelope contract.

        Handles new transport envelopes, legacy payloads, and raw error-shaped
        responses while always returning
        `ok/tool_name/result/error/tool_call_id/meta`.
        """
        if isinstance(payload, dict):
            if "ok" in payload and ("result" in payload or "error" in payload):
                error = payload.get("error")
                result = payload.get("result")
                ok = self._coerce_ok_flag(payload.get("ok"))
                if error not in (None, "", {}) and result in (None, "", {}):
                    ok = False
                return {
                    "ok": ok,
                    "tool_name": payload.get("tool_name", tool_name),
                    "result": result,
                    "error": error,
                    "tool_call_id": payload.get("tool_call_id", tool_call_id),
                    "meta": payload.get("meta", {}),
                }
            if "result" in payload and ("tool_name" in payload or "error" in payload):
                error = payload.get("error")
                return {
                    "ok": error in (None, "", {}),
                    "tool_name": payload.get("tool_name", tool_name),
                    "result": payload.get("result"),
                    "error": error,
                    "tool_call_id": payload.get("tool_call_id", tool_call_id),
                    "meta": payload.get("meta", {}),
                }
            if any(key in payload for key in ("error", "detail", "message")):
                error = payload.get("error")
                if not isinstance(error, dict):
                    message = payload.get("detail") or payload.get("message") or str(error)
                    error = {
                        "code": "tool_execution_error_payload",
                        "message": message or "Tool execution returned an error payload.",
                    }
                return {
                    "ok": False,
                    "tool_name": payload.get("tool_name", tool_name),
                    "result": payload.get("result"),
                    "error": error,
                    "tool_call_id": payload.get("tool_call_id", tool_call_id),
                    "meta": {"warning": "non_envelope_error_payload"},
                }
            return {
                "ok": True,
                "tool_name": tool_name,
                "result": payload,
                "error": None,
                "tool_call_id": tool_call_id,
                "meta": {"warning": "non_envelope_payload"},
            }

        return {
            "ok": True,
            "tool_name": tool_name,
            "result": payload,
            "error": None,
            "tool_call_id": tool_call_id,
            "meta": {"warning": "unstructured_tool_payload"},
        }

    def _post_process_tool_payload(
        self,
        normalized_payload: Dict[str, Any],
        tool_call: Dict[str, Any],
        run_memory,
    ) -> Dict[str, Any]:
        """Hook for subclasses to post-process a tool execution payload.

        Called after ``_normalize_execute_payload`` and image-resolution
        metadata have been applied but before the payload is consumed by
        memory.  The default implementation is a no-op passthrough.
        """
        return normalized_payload

    def _extract_final_answer_from_content(self, content: str) -> str:
        if content is None:
            return ""
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except Exception:
                content = str(content)

        prompt_version = str(getattr(self, "prompt_version", "v1")).strip().lower()

        # Support v2 "Action: Final Answer" + "Action Input:" response style.
        if prompt_version in ("v2", "v3"):
            action_name_match = re.search(
                r"^\s*Action\s*:\s*(.+?)\s*$",
                content,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if action_name_match:
                action_name = action_name_match.group(1).strip()
                if action_name and action_name.lower() != "final answer":
                    return ""

            action_match = re.search(
                r"^\s*Action\s*:\s*Final\s*Answer\s*$",
                content,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if action_match:
                tail = content[action_match.end():]
                action_input_match = re.search(
                    r"^\s*Action\s*Input\s*:\s*",
                    tail,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
                if action_input_match:
                    answer = tail[action_input_match.end():].strip()
                    if answer:
                        return answer
                    return ""

            v2_scaffold_line = r"^\s*(Question|Observation|Thought|Action|Action\s+Input)\s*:\s*.*$"
            if re.search(v2_scaffold_line, content, flags=re.IGNORECASE | re.MULTILINE):
                non_scaffold = re.sub(v2_scaffold_line, "", content, flags=re.IGNORECASE | re.MULTILINE).strip()
                if not non_scaffold:
                    return ""
                fa_match = re.search(r"Final\s+Answer\s*:\s*", non_scaffold, re.IGNORECASE)
                if fa_match:
                    return non_scaffold[fa_match.end():].strip()

        # Strict <answer> extraction for prompts that use that tag.
        if prompt_version in ("v1_instruct", "v1_thinking", "v2_instruct", "v2_thinking", "v3_instruct"):
            answer_match = re.search(r"<answer>(.*?)</answer>", content, flags=re.DOTALL | re.IGNORECASE)
            if answer_match:
                return answer_match.group(1).strip()
            return ""

        cleaned = re.sub(r"<perception>.*?</perception>", "", content, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = cleaned.strip()
        if cleaned.lower().startswith("final answer:"):
            cleaned = cleaned.split(":", 1)[1].strip()
        if cleaned:
            return cleaned
        if re.search(r"</?\s*(perception|thinking)\s*>", content, flags=re.IGNORECASE):
            return ""
        return content.strip()

    # ------------------------------------------------------------------ #
    # Context-length protection helpers
    # ------------------------------------------------------------------ #

    def _count_tokens(self, msgs, tools=None):
        """Return token count for *msgs* using the cached tokenizer, or None."""
        try:
            tokenizer = getattr(self, "_tokenizer", None)
            if tokenizer is None:
                return None
            sanitized = self._sanitize_msgs_for_template(msgs)
            tmpl_kwargs = dict(tokenize=True, add_generation_prompt=True)
            if tools:
                tmpl_kwargs["tools"] = tools
            tokens = tokenizer.apply_chat_template(sanitized, **tmpl_kwargs)
            return len(tokens)
        except Exception:
            return None

    @staticmethod
    def _is_environment_update(msg):
        """Check if a message is an Environment Update from persistent state memory."""
        content = msg.get("content", "")
        if isinstance(content, str):
            return content.startswith("Environment Update")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    if part.get("text", "").startswith("Environment Update"):
                        return True
        return False

    def _compact_history(self, msgs, token_budget, tools=None):
        """Truncate older conversation messages in-place to fit *token_budget*.

        Walks from oldest to newest and replaces tool/assistant/user content
        with a short placeholder.  Preserves:
        - The system message and initial user message block
        - The most recent assistant+tool exchange
        - The latest Environment Update message (needed for image ID tracking)

        Returns True if the messages now fit within *token_budget*.
        """
        # Find the index of the first assistant message (start of tool turns)
        first_assistant = None
        for idx, m in enumerate(msgs):
            if m.get("role") == "assistant":
                first_assistant = idx
                break
        if first_assistant is None:
            return False  # nothing to compact

        # Find the last assistant message index — preserve the most recent turn
        last_assistant = None
        for idx in range(len(msgs) - 1, -1, -1):
            if msgs[idx].get("role") == "assistant":
                last_assistant = idx
                break

        # Find the latest Environment Update — always preserve it
        last_env_update = None
        for idx in range(len(msgs) - 1, -1, -1):
            if msgs[idx].get("role") == "user" and self._is_environment_update(msgs[idx]):
                last_env_update = idx
                break

        # Compact from oldest turns forward, skipping the latest turn and env update
        for idx in range(first_assistant, last_assistant or len(msgs)):
            if idx == last_env_update:
                continue  # preserve latest environment update
            role = msgs[idx].get("role")
            if role in ("tool", "assistant", "user"):
                content = msgs[idx].get("content", "")
                if isinstance(content, str) and len(content) > 100:
                    msgs[idx]["content"] = "[Truncated to save context]"
                elif isinstance(content, list):
                    msgs[idx]["content"] = "[Truncated to save context]"
                # Also strip tool_calls from compacted assistant messages
                if role == "assistant":
                    msgs[idx].pop("tool_calls", None)
                count = self._count_tokens(msgs, tools)
                if count is not None and count <= token_budget:
                    return True

        count = self._count_tokens(msgs, tools)
        return count is not None and count <= token_budget

    def _build_vanilla_fallback_msgs(self, msgs):
        """Build a minimal user-only message list for base-agent-style fallback.

        Used when context length is exceeded — drops all tool conversation
        and retries with just the original input and a direct-answer prompt.
        """
        fallback = []
        # Do not keep the tool-agent system prompt; fallback should behave like BaseAgent, we only keep the original user question.
        for m in msgs:
            if m.get("role") == "user":
                fallback.append(m)
                break
        return fallback

    def _ensure_context_fits(self, msgs, model_tools, tool_choice):
        """Check context length and compact/force final answer if needed.

        Returns (model_tools, tool_choice) — may be modified if the model is
        forced to answer directly.
        """
        max_ctx = getattr(self, "max_context_length", None)
        if not max_ctx:
            return model_tools, tool_choice
        tokenizer = getattr(self, "_tokenizer", None)
        if tokenizer is None:
            return model_tools, tool_choice

        token_budget = max_ctx - self.max_tokens - 512
        current_tokens = self._count_tokens(msgs, model_tools)
        if current_tokens is None or current_tokens <= token_budget:
            return model_tools, tool_choice

        self.logger.warning(
            f"[ToolLoop] Token count ({current_tokens}) exceeds budget "
            f"({token_budget}), compacting history..."
        )
        if self._compact_history(msgs, token_budget, model_tools):
            return model_tools, tool_choice

        # Still over limit — fall back to vanilla QA
        self.logger.warning(
            "[ToolLoop] Still over limit after compaction, falling back to vanilla QA"
        )
        fallback_msgs = self._build_vanilla_fallback_msgs(msgs)
        msgs.clear()
        msgs.extend(fallback_msgs)
        return None, "none"

    @staticmethod
    def _sanitize_msgs_for_template(msgs):
        """Replace image_url content items with [image] placeholder for chat template."""
        sanitized = cp.deepcopy(msgs)
        for msg in sanitized:
            content = msg.get("content")
            if isinstance(content, list):
                msg["content"] = [
                    {"type": "text", "text": "[image]"}
                    if (isinstance(item, dict) and item.get("type") == "image_url")
                    else item
                    for item in content
                ]
        return sanitized

    def _truncate_for_debug(self, value: Any, max_chars: int = 280) -> str:
        try:
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
        text = text.replace("\n", "\\n")
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}... (truncated)"

    def _build_raw_sft_from_tokens(self, terminating_response, msgs, task_id):
        """Build an SFT-ready trajectory from the terminating response's token ids.

        Returns ``{"messages": [...], "images": [{"filename", "bytes_b64"}, ...]}``
        or ``None`` when the response does not carry ``prompt_token_ids`` /
        ``choices[0].token_ids`` (e.g. OpenAI endpoint, or user did not enable
        ``extra_body.return_token_ids``).

        Token ids cannot recover image bytes (vLLM replaces images with
        ``<|image_pad|>``), so we walk ``msgs`` to collect the original data
        URIs in order and map them to ``<|vision_start|>...<|vision_end|>``
        matches by position. Mismatch raises ``ValueError`` — the run has a
        token/message drift and is not safe for SFT.
        """
        import base64
        from ..utils.reporting_utils import iter_data_uris

        if not isinstance(terminating_response, dict):
            return None
        prompt_ids = terminating_response.get("prompt_token_ids")
        choices = terminating_response.get("choices") or []
        completion_ids = (choices[0] or {}).get("token_ids") if choices else None
        if not isinstance(prompt_ids, list) or not isinstance(completion_ids, list):
            return None

        tokenizer = self._tokenizer
        if tokenizer is None:
            self.logger.warning(f"[raw_sft] no tokenizer available for {self.model}")
            return None

        images = list(iter_data_uris(msgs))

        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        full_ids = list(prompt_ids) + list(completion_ids) + [im_end_id]
        decoded = tokenizer.decode(full_ids, skip_special_tokens=False)

        vision_re = re.compile(r"<\|vision_start\|>.*?<\|vision_end\|>", re.DOTALL)
        vision_matches = vision_re.findall(decoded)
        if len(vision_matches) != len(images):
            raise ValueError(
                f"[raw_sft] vision block/image count mismatch for task {task_id}: "
                f"{len(vision_matches)} vision blocks vs {len(images)} images in msgs."
            )

        sentinel_text = vision_re.sub(
            lambda m, _c=[0]: (_c.__setitem__(0, _c[0] + 1) or f"__IMAGE_SLOT_{_c[0] - 1}__"),
            decoded,
        )

        image_entries = []
        for idx, (data, ext) in enumerate(images):
            image_entries.append({
                "filename": f"{task_id}_{idx}{ext}",
                "bytes_b64": base64.b64encode(data).decode("ascii"),
            })

        messages_out = []
        slot_re = re.compile(r"__IMAGE_SLOT_(\d+)__")
        for part in sentinel_text.split("<|im_start|>")[1:]:
            head, _, body = part.partition("\n")
            role = head.strip()
            if role not in {"system", "user", "assistant"}:
                raise ValueError(f"[raw_sft] unexpected role in raw trajectory: {role!r}")
            body = body.split("<|im_end|>", 1)[0]

            if not slot_re.search(body):
                messages_out.append({
                    "role": role,
                    "content": [{"type": "text", "text": body.strip()}],
                })
                continue

            content = []
            cursor = 0
            for m in slot_re.finditer(body):
                pre = body[cursor:m.start()].strip()
                if pre:
                    content.append({"type": "text", "text": pre})
                slot_idx = int(m.group(1))
                content.append({
                    "type": "image",
                    "image": f"images/{image_entries[slot_idx]['filename']}",
                })
                cursor = m.end()
            trailing = body[cursor:].strip()
            if trailing:
                content.append({"type": "text", "text": trailing})
            messages_out.append({"role": role, "content": content})

        return {"messages": messages_out, "images": image_entries}

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def _run(
        self,
        question,
        return_messages: bool = False,
        **kwargs,
    ) -> tuple:
        """
        Run the native tool-calling loop.

        Flow: `chat/completions` -> `message.tool_calls` -> `/execute` ->
        append user-role tool-result messages -> continue until a final answer
        is produced or `max_iterations` is reached.
        """
        debug_tool_loop = bool(kwargs.pop("debug_tool_loop", False))
        debug_char_limit = int(kwargs.pop("debug_char_limit", 280))
        chat_timeout_seconds = kwargs.pop("timeout_seconds", self.timeout)
        tool_choice = kwargs.pop("tool_choice", self.tool_choice)
        active_system_prompt = kwargs.pop("system_prompt", self.system_prompt)
        task_id = kwargs.pop("task_id", None)
        endpoint_hint = f"{getattr(self, 'base', '')} {getattr(self, 'api_base', '')}".lower()
        is_openai_official = "api.openai.com" in endpoint_hint
        model_name = getattr(self, "model", "") or ""
        is_openai_native = is_openai_official or model_name.strip().lower().startswith("gpt-5")
        # For OpenAI / GPT-5 endpoint, strip internal orchestration-only field.
        if is_openai_native:
            _ = kwargs.pop("dataset", None)
        repetition_penalty = kwargs.pop("repetition_penalty", getattr(self, "repetition_penalty", 1.0))
        if not is_openai_native:
            kwargs["repetition_penalty"] = repetition_penalty
        runtime_tools = kwargs.pop("tools", None)
        model_tools = self.json_schema
        if isinstance(runtime_tools, list):
            model_tools = runtime_tools
        elif runtime_tools is not None and debug_tool_loop:
            self.logger.info("[ToolLoop] Ignoring non-list runtime tools override; using agent tool schema.")

        run_memory = self._new_persistent_state_memory()
        run_memory.reset()
        # Phase 2: stamp the row identity so detection-tool post-processors
        # (e.g. SAM3) can attach (boxes, scores) sidecar entries that the
        # inference loop will persist as tool_detections/<task_id>.json.
        if hasattr(run_memory, "reset_for_row"):
            run_memory.reset_for_row(str(task_id) if task_id is not None else None)
        msgs = self.prepare_inputs(question, system_prompt=active_system_prompt)

        msgs = self._add_user_prompt_prefix(msgs)
        self._append_messages_with_indices(
            msgs,
            run_memory.on_run_start(
                agent=self,
                question=question,
                messages=msgs,
            ),
        )

        # Use this for debugging the input messages
        # pretty_print_msgs(msgs)
        chat_history = []
        final_answer = None
        terminating_response = None
        for i in range(self.max_iterations):
            if debug_tool_loop:
                self.logger.info(f"[ToolLoop] turn={i + 1}/{self.max_iterations}, messages={len(msgs)}")
            if i == self.max_iterations - 1:
                msgs.append(
                    {
                        "role": "user",
                        "content": "Please provide your best final answer now if enough information is available.",
                    }
                )

            try:
                response = chat_completion_request(
                    model=self.model,
                    messages=msgs,
                    base_url=self.api_base,
                    api_key=self.key,
                    tools=model_tools,
                    tool_choice=tool_choice,
                    timeout_seconds=chat_timeout_seconds,
                    **kwargs,
                )
            except RuntimeError as e:
                err_msg = str(e).lower()
                if "context length" in err_msg or "maximum input length" in err_msg:
                    self.logger.warning(
                        f"[ToolLoop] Context length error, falling back to vanilla QA: {e}"
                    )
                    fallback_msgs = self._build_vanilla_fallback_msgs(msgs)
                    response = chat_completion_request(
                        model=self.model,
                        messages=fallback_msgs,
                        base_url=self.api_base,
                        api_key=self.key,
                        tools=None,
                        tool_choice="none",
                        timeout_seconds=chat_timeout_seconds,
                        **kwargs,
                    )
                else:
                    raise
            if not isinstance(response, dict):
                raise RuntimeError(
                    f"Invalid chat response type: expected dict, got {type(response).__name__}"
                )
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise RuntimeError(f"Invalid chat response payload: missing choices. Response={self._truncate_for_debug(response)}")
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise RuntimeError(
                    f"Invalid chat response payload: choices[0].message must be an object. Response={self._truncate_for_debug(response)}"
                )
            raw_response = message.get("content")
            raw_response = raw_response if isinstance(raw_response, str) else ""
            if debug_tool_loop and raw_response:
                self.logger.info(
                    f"[ToolLoop] assistant_content={self._truncate_for_debug(raw_response, max_chars=debug_char_limit)}"
                )

            native_tool_calls = self._normalize_tool_calls(message.get("tool_calls"), turn_idx=i)
            native_tool_calls = self._enforce_single_tool_call_per_message(
                native_tool_calls,
                turn_idx=i,
                phase="main",
            )
            parsed_response = self.parse_response(raw_response)
            if debug_tool_loop and native_tool_calls:
                self.logger.info(f"[ToolLoop] tool_calls={len(native_tool_calls)}")

            turn_info = {
                "turn": i + 1,
                "assistant": {
                    "content": raw_response,
                    "perception": parsed_response["perception"],
                    "thinking": parsed_response["thinking"],
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "name": call["name"],
                            "raw_arguments": call["raw_arguments"],
                            "args": call["args"],
                            "parse_error": call.get("parse_error"),
                        }
                        for call in native_tool_calls
                    ],
                },
                "tool_responses": [],
                "memory_outputs": [],
                "memory_turn_message_indices": [],
            }

            if native_tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": [call["assistant_tool_call"] for call in native_tool_calls],
                }
                msgs.append(assistant_message)

                deferred_memory_outputs: List[Dict[str, Any]] = []
                for call in native_tool_calls:
                    image_resolution = {"rewrites": [], "errors": []}
                    parse_error = call.get("parse_error")
                    if parse_error:
                        normalized_payload = {
                            "ok": False,
                            "tool_name": call["name"],
                            "result": None,
                            "error": parse_error,
                            "tool_call_id": call["id"],
                            "meta": {"source": "agent.tool_call_parser"},
                        }
                    else:
                        resolved_args, image_resolution = run_memory.resolve_tool_call_args(
                            agent=self,
                            args=call["args"],
                            tool_name=call.get("name", ""),
                        )
                        if debug_tool_loop:
                            self.logger.info(
                                f"[ToolLoop] execute {call['name']} id={call['id']} args={self._truncate_for_debug(resolved_args, max_chars=debug_char_limit)}"
                            )
                            if image_resolution.get("rewrites"):
                                self.logger.info(
                                    f"[ToolLoop] image_arg_rewrites={self._truncate_for_debug(image_resolution['rewrites'], max_chars=debug_char_limit)}"
                                )
                        execute_request = {
                            "tool_name": call["name"],
                            "args": resolved_args,
                            "tool_call_id": call["id"],
                        }
                        if image_resolution.get("errors"):
                            normalized_payload = {
                                "ok": False,
                                "tool_name": call["name"],
                                "result": None,
                                "error": {
                                    "code": "unresolved_image_reference",
                                    "message": "Failed to map one or more image_path arguments to runtime images.",
                                    "details": image_resolution.get("errors", []),
                                },
                                "tool_call_id": call["id"],
                                "meta": {"image_resolution": image_resolution},
                            }
                        else:
                            try:
                                payload = tool_execute_request(
                                    base_url=self.tool_base,
                                    tool_call=execute_request,
                                    timeout=self.timeout,
                                )
                                normalized_payload = self._normalize_execute_payload(
                                    payload, tool_name=call["name"], tool_call_id=call["id"]
                                )
                            except Exception as e:
                                normalized_payload = {
                                    "ok": False,
                                    "tool_name": call["name"],
                                    "result": None,
                                    "error": {"code": "tool_execution_error", "message": str(e)},
                                    "tool_call_id": call["id"],
                                    "meta": {},
                                }
                            if image_resolution.get("rewrites"):
                                meta = normalized_payload.get("meta")
                                if not isinstance(meta, dict):
                                    meta = {}
                                meta["image_resolution"] = image_resolution
                                normalized_payload["meta"] = meta
                            normalized_payload = self._post_process_tool_payload(
                                normalized_payload, call, run_memory,
                            )
                    if debug_tool_loop:
                        self.logger.info(
                            f"[ToolLoop] result ok={normalized_payload.get('ok')} id={call['id']} error={self._truncate_for_debug(normalized_payload.get('error'), max_chars=debug_char_limit)}"
                        )

                    turn_info["tool_responses"].append(normalized_payload)
                    memory_result = run_memory.consume_tool_payload(
                        agent=self,
                        turn_idx=i,
                        tool_call=call,
                        payload=normalized_payload,
                    )
                    immediate_messages = memory_result.get("immediate_messages")
                    deferred_messages = memory_result.get("deferred_messages")
                    prepared_tool_payload = memory_result.get("prepared_payload")
                    processor_info = memory_result.get("processor_info")

                    if debug_tool_loop:
                        emitted_preview = ""
                        if isinstance(immediate_messages, list) and immediate_messages:
                            first_msg = next(
                                (m for m in immediate_messages if isinstance(m, dict)),
                                None,
                            )
                            if isinstance(first_msg, dict):
                                summary = self._summarize_message_for_trace(first_msg)
                                emitted_preview = self._truncate_for_debug(
                                    summary, max_chars=debug_char_limit
                                )
                        self.logger.info(
                            f"[ToolLoop] emitted_tool_result_message={emitted_preview}"
                        )
                        if isinstance(processor_info, dict):
                            self.logger.info(
                                "[ToolLoop] tool_output_processor=%s",
                                self._truncate_for_debug(processor_info, max_chars=debug_char_limit),
                            )

                    immediate_indices = self._append_messages_with_indices(msgs, immediate_messages)
                    memory_output = {
                        "tool_call_id": call["id"],
                        "tool_name": call["name"],
                        "memory_name": run_memory.memory_name,
                        "immediate_message_indices": immediate_indices,
                        "deferred_message_indices": [],
                        "message_summaries": [
                            self._summarize_message_for_trace(msgs[idx]) for idx in immediate_indices
                        ],
                    }
                    if isinstance(prepared_tool_payload, dict):
                        image_refs = prepared_tool_payload.get("image_refs")
                        if isinstance(image_refs, list) and image_refs:
                            memory_output["image_refs"] = image_refs
                    if isinstance(processor_info, dict) and processor_info:
                        memory_output["tool_output_processor"] = cp.deepcopy(processor_info)
                    turn_info["memory_outputs"].append(memory_output)

                    if isinstance(deferred_messages, list) and deferred_messages:
                        deferred_memory_outputs.append(
                            {
                                "memory_output": memory_output,
                                "messages": deferred_messages,
                            }
                        )

                if deferred_memory_outputs:
                    for deferred in deferred_memory_outputs:
                        deferred_indices = self._append_messages_with_indices(msgs, deferred["messages"])
                        deferred["memory_output"]["deferred_message_indices"] = deferred_indices
                        deferred["memory_output"]["message_summaries"].extend(
                            [self._summarize_message_for_trace(msgs[idx]) for idx in deferred_indices]
                        )
                    if debug_tool_loop:
                        image_count = sum(
                            1
                            for deferred in deferred_memory_outputs
                            for idx in deferred["memory_output"]["deferred_message_indices"]
                            for part in (msgs[idx].get("content") if isinstance(msgs[idx].get("content"), list) else [])
                            if isinstance(part, dict) and part.get("type") == "image_url"
                        )
                        self.logger.info(
                            f"[ToolLoop] appended user tool-image updates={len(deferred_memory_outputs)} image_parts={image_count}"
                        )

                turn_info["memory_turn_message_indices"] = self._append_messages_with_indices(
                    msgs,
                    run_memory.emit_after_tool_turn(
                        agent=self,
                        turn_idx=i,
                        tool_calls=native_tool_calls,
                        tool_payloads=turn_info["tool_responses"],
                        messages=msgs,
                    ),
                )
                chat_history.append(turn_info)
                if i == self.max_iterations - 1:
                    if debug_tool_loop:
                        self.logger.info(
                            "[ToolLoop] last iteration ended with tool calls; requesting one final text-only answer."
                        )
                    final_response = chat_completion_request(
                        model=self.model,
                        messages=msgs,
                        base_url=self.api_base,
                        api_key=self.key,
                        tools=model_tools,
                        tool_choice="none",
                        timeout_seconds=chat_timeout_seconds,
                        **kwargs,
                    )
                    if not isinstance(final_response, dict):
                        raise RuntimeError(
                            f"Invalid chat response type: expected dict, got {type(final_response).__name__}"
                        )
                    final_choices = final_response.get("choices")
                    if not isinstance(final_choices, list) or not final_choices or not isinstance(final_choices[0], dict):
                        raise RuntimeError(
                            f"Invalid chat response payload: missing choices. Response={self._truncate_for_debug(final_response)}"
                        )
                    final_message = final_choices[0].get("message")
                    if not isinstance(final_message, dict):
                        raise RuntimeError(
                            "Invalid chat response payload: choices[0].message must be an object. "
                            f"Response={self._truncate_for_debug(final_response)}"
                        )

                    final_raw_response = final_message.get("content")
                    final_raw_response = final_raw_response if isinstance(final_raw_response, str) else ""
                    final_tool_calls = self._normalize_tool_calls(final_message.get("tool_calls"), turn_idx=i + 1)
                    final_tool_calls = self._enforce_single_tool_call_per_message(
                        final_tool_calls,
                        turn_idx=i + 1,
                        phase="final_text_only",
                    )
                    final_parsed_response = self.parse_response(final_raw_response)
                    final_answer = self._extract_final_answer_from_content(final_raw_response)
                    terminating_response = final_response

                    if final_tool_calls:
                        msgs.append(
                            {
                                "role": "assistant",
                                "content": final_message.get("content") or "",
                                "tool_calls": [call["assistant_tool_call"] for call in final_tool_calls],
                            }
                        )
                    else:
                        msgs.append({"role": "assistant", "content": final_raw_response})

                    final_turn_info = {
                        "turn": i + 2,
                        "assistant": {
                            "content": final_raw_response,
                            "perception": final_parsed_response["perception"],
                            "thinking": final_parsed_response["thinking"],
                            "tool_calls": [
                                {
                                    "id": call["id"],
                                    "name": call["name"],
                                    "raw_arguments": call["raw_arguments"],
                                    "args": call["args"],
                                    "parse_error": call.get("parse_error"),
                                }
                                for call in final_tool_calls
                            ],
                        },
                        "tool_responses": [],
                        "final_answer": final_answer,
                    }
                    chat_history.append(final_turn_info)

                    raw_sft = self._build_raw_sft_from_tokens(terminating_response, msgs, task_id)
                    status = final_answer if final_answer else self.fail_msg
                    if return_messages:
                        return status, chat_history, msgs, raw_sft
                    return status, chat_history
                continue

            final_answer = self._extract_final_answer_from_content(raw_response)
            turn_info["final_answer"] = final_answer
            chat_history.append(turn_info)
            if debug_tool_loop:
                self.logger.info(
                    f"[ToolLoop] final_answer={self._truncate_for_debug(final_answer, max_chars=debug_char_limit)}"
                )
            if final_answer:
                terminating_response = response
                raw_sft = self._build_raw_sft_from_tokens(terminating_response, msgs, task_id)
                self._last_tool_detections = list(getattr(run_memory, "tool_detections", []) or [])
                if return_messages:
                    return final_answer, chat_history, msgs, raw_sft
                return final_answer, chat_history

            msgs.append({"role": "assistant", "content": raw_response})

        self._last_tool_detections = list(getattr(run_memory, "tool_detections", []) or [])
        if return_messages:
            return self.fail_msg, chat_history, msgs, None
        return self.fail_msg, chat_history

    def generate(self, message, **kwargs1):
        """Generate one final answer using the native tool-calling loop."""

        task_id = kwargs1.pop("task_id", None)

        raw_message = cp.deepcopy(message)
        message = cp.deepcopy(message)
        active_system_prompt = self.system_prompt
        if self.check_content(message) == 'listdict':
            message, active_system_prompt = self.preprocess_message_with_role(
                message,
                base_system_prompt=self.system_prompt,
                return_system_prompt=True,
            )
        # Prepend instance critique if available (before content normalization)
        if task_id is not None and self.instance_critique_dir:
            critique = self._load_instance_critique(str(task_id))
            if critique:
                critique_text = f"Here are the feedback for your previous failed attempt:\n{critique}\n\n"
                content_type = self.check_content(message)
                if content_type == 'str':
                    message = message + "\n\n" + critique_text
                elif content_type == 'listdict':
                    message.append({"type": "text", "value": critique_text})
                elif content_type == 'dict':
                    message = [message, {"type": "text", "value": critique_text}]
                elif content_type == 'liststr':
                    message.append(critique_text)

        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        message = self.preproc_content(message)
        assert message is not None and self.check_content(message) == 'listdict'
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'
        # merge kwargs
        kwargs = cp.deepcopy(self.default_kwargs)
        kwargs.update(kwargs1)
        kwargs = filter_kwargs_for_endpoint(kwargs, "chat")

        # Run the ReAct-style loop. Catch RuntimeError per-sample so a single
        # failed request does not crash the entire benchmark run.
        try:
            answer, chat_history, messages, raw_sft = self._run(
                message,
                return_messages=True,
                system_prompt=active_system_prompt,
                task_id=task_id,
                **kwargs,
            )
        except RuntimeError as exc:
            self.logger.warning(
                f"[generate] Sample failed with RuntimeError, marking as failed: {exc}"
            )
            trajectory = {
                "input_messages": raw_message,
                "chat_history": [],
                "messages": [],
                "final_answer": self.fail_msg,
                "error": str(exc),
            }
            return {
                "prediction": self.fail_msg,
                "extra_records": {
                    "trajectory": trajectory,
                    "tool_detections": list(getattr(self, "_last_tool_detections", []) or []),
                },
            }

        final_answer = self.fail_msg if answer in ['', None] else answer
        trajectory = {
            "input_messages": raw_message,
            "chat_history": chat_history,
            "messages": messages,
            "final_answer": final_answer,
        }
        if raw_sft is not None:
            trajectory["raw_sft"] = raw_sft
        return {
            "prediction": final_answer,
            "extra_records": {
                "trajectory": trajectory,
                # Phase 2: SAM3 (and other detection-tool) per-call (boxes, scores)
                # candidate sets, captured during the agent's tool loop. Empty for
                # BaseAgent or for tool-use rows that didn't trigger a scoring tool.
                "tool_detections": list(getattr(self, "_last_tool_detections", []) or []),
            },
        }


class BaseAgentWithToolsV2(BaseAgentWithTools):
    """Text-based tool calling agent.

    Embeds full tool JSON schemas in the system prompt and parses
    tool calls from <tool_call> XML blocks in the model's text output,
    instead of relying on native OpenAI tool-calling protocol.
    """

    _TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def __init__(self, stop_tokens=None, save_raw_response: bool = False, **kwargs):
        kwargs.setdefault("prompt_version", "v3")
        super().__init__(**kwargs)
        self.save_raw_response = save_raw_response
        self.stop_tokens = stop_tokens if stop_tokens is not None else ["</tool_call>"]
        # Rebuild system prompt with tool schemas embedded.
        self.system_prompt = self._construct_system_message()
        self.logger = get_logger("BaseAgentWithToolsV2")

    # -------------------------------------------------------------- #
    # Prompt construction
    # -------------------------------------------------------------- #

    def _construct_system_message(self) -> str:
        sys_template, _ = self._get_prompt_templates()
        directory = self._build_tools_directory()
        prompt = sys_template.replace("{tools_directory}", directory)
        # Append experience (same as parent pattern).
        if getattr(self, "experience_file", None) and os.path.exists(self.experience_file):
            from ..experience.reflector import load_experience
            experience_text = load_experience(self.experience_file)
            if experience_text.strip():
                prompt += (
                    "\n\nAdditional learned tool-usage principles from prior runs:\n"
                    "- Treat these as reusable heuristics when choosing tools, arguments, and retries.\n"
                    "- Follow them when relevant, but do not violate tool protocols above.\n"
                    "<learned_principles>\n"
                    f"{experience_text}\n"
                    "</learned_principles>"
                )
                self.logger.info(f"Loaded experience from {self.experience_file}")
        return prompt

    def _build_tools_directory(self) -> str:
        entries = []
        for entry in self.json_schema:
            if isinstance(entry, dict) and isinstance(entry.get("function"), dict):
                entries.append(json.dumps(entry, ensure_ascii=False, indent=2))
        return "\n".join(entries)

    # -------------------------------------------------------------- #
    # Text-based tool-call parsing
    # -------------------------------------------------------------- #

    def _parse_tool_calls_from_text(self, content: str, turn_idx: int) -> List[Dict[str, Any]]:
        if not content:
            return []
        matches = self._TOOL_CALL_RE.findall(content)
        if not matches:
            return []

        results = []
        for idx, block in enumerate(matches):
            block = block.strip()
            call_id = f"call_{turn_idx + 1}_{idx + 1}"

            try:
                parsed = json.loads(block)
            except Exception as exc:
                results.append({
                    "id": call_id,
                    "name": "unknown",
                    "raw_arguments": block,
                    "args": {},
                    "assistant_tool_call": {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "unknown", "arguments": block},
                    },
                    "parse_error": {
                        "code": "invalid_tool_call_json",
                        "message": "Could not parse <tool_call> block as JSON.",
                        "details": {"raw": block, "exception": str(exc)},
                    },
                })
                continue

            if not isinstance(parsed, dict) or "name" not in parsed:
                results.append({
                    "id": call_id,
                    "name": "unknown",
                    "raw_arguments": block,
                    "args": {},
                    "assistant_tool_call": {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "unknown", "arguments": block},
                    },
                    "parse_error": {
                        "code": "invalid_tool_call_format",
                        "message": "Tool call JSON must be an object with a 'name' field.",
                        "details": {"raw": block},
                    },
                })
                continue

            tool_name = parsed["name"]
            args = parsed.get("arguments", {})
            if not isinstance(args, dict):
                args = {}
            raw_arguments = json.dumps(args, ensure_ascii=False)
            results.append({
                "id": call_id,
                "name": tool_name,
                "raw_arguments": raw_arguments,
                "args": args,
                "assistant_tool_call": {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool_name, "arguments": raw_arguments},
                },
                "parse_error": None,
            })
        return results

    # -------------------------------------------------------------- #
    # Per-call execution (overridable by subclasses)
    # -------------------------------------------------------------- #

    def _execute_tool_call(self, call, run_memory, debug_tool_loop, debug_char_limit):
        """Execute a single parsed tool call. Returns (normalized_payload, image_resolution)."""
        image_resolution = {"rewrites": [], "errors": []}
        resolved_args, image_resolution = run_memory.resolve_tool_call_args(
            agent=self, args=call["args"], tool_name=call.get("name", ""),
        )
        if debug_tool_loop:
            self.logger.info(
                f"[ToolLoop] execute {call['name']} id={call['id']} "
                f"args={self._truncate_for_debug(resolved_args, max_chars=debug_char_limit)}"
            )
        execute_request = {
            "tool_name": call["name"],
            "args": resolved_args,
            "tool_call_id": call["id"],
        }
        if image_resolution.get("errors"):
            normalized_payload = {
                "ok": False,
                "tool_name": call["name"],
                "result": None,
                "error": {
                    "code": "unresolved_image_reference",
                    "message": "Failed to map one or more image_path arguments to runtime images.",
                    "details": image_resolution.get("errors", []),
                },
                "tool_call_id": call["id"],
                "meta": {"image_resolution": image_resolution},
            }
        else:
            try:
                payload = tool_execute_request(
                    base_url=self.tool_base,
                    tool_call=execute_request,
                    timeout=self.timeout,
                )
                normalized_payload = self._normalize_execute_payload(
                    payload, tool_name=call["name"], tool_call_id=call["id"],
                )
            except Exception as e:
                normalized_payload = {
                    "ok": False,
                    "tool_name": call["name"],
                    "result": None,
                    "error": {"code": "tool_execution_error", "message": str(e)},
                    "tool_call_id": call["id"],
                    "meta": {},
                }
            if image_resolution.get("rewrites"):
                meta = normalized_payload.get("meta")
                if not isinstance(meta, dict):
                    meta = {}
                meta["image_resolution"] = image_resolution
                normalized_payload["meta"] = meta
        return normalized_payload, image_resolution

    # -------------------------------------------------------------- #
    # Main loop
    # -------------------------------------------------------------- #

    def _run(
        self,
        question,
        return_messages: bool = False,
        **kwargs,
    ) -> tuple:
        debug_tool_loop = bool(kwargs.pop("debug_tool_loop", False))
        debug_char_limit = int(kwargs.pop("debug_char_limit", 280))
        chat_timeout_seconds = kwargs.pop("timeout_seconds", self.timeout)
        _ = kwargs.pop("tool_choice", None)  # not used in V2
        active_system_prompt = kwargs.pop("system_prompt", self.system_prompt)
        task_id = kwargs.pop("task_id", None)  # Phase 2: stamp on memory
        endpoint_hint = f"{getattr(self, 'base', '')} {getattr(self, 'api_base', '')}".lower()
        is_openai_official = "api.openai.com" in endpoint_hint
        model_name = getattr(self, "model", "") or ""
        is_openai_native = is_openai_official or model_name.strip().lower().startswith("gpt-5")
        _ = kwargs.pop("dataset", None)  # always remove — not a vLLM/API param
        repetition_penalty = kwargs.pop("repetition_penalty", getattr(self, "repetition_penalty", 1.0))
        if not is_openai_native:
            kwargs["repetition_penalty"] = repetition_penalty
        _ = kwargs.pop("tools", None)  # not used in V2

        run_memory = self._new_persistent_state_memory()
        run_memory.reset()
        # Phase 2: per-row sidecar bookkeeping (mirrors BaseAgentWithTools._run).
        if hasattr(run_memory, "reset_for_row"):
            run_memory.reset_for_row(str(task_id) if task_id is not None else None)

        msgs = self.prepare_inputs(question, system_prompt=active_system_prompt)
        msgs = self._add_user_prompt_prefix(msgs)
        self._append_messages_with_indices(
            msgs,
            run_memory.on_run_start(agent=self, question=question, messages=msgs),
        )

        chat_history = []
        final_answer = None

        for i in range(self.max_iterations):
            if debug_tool_loop:
                self.logger.info(f"[ToolLoop] turn={i + 1}/{self.max_iterations}, messages={len(msgs)}")

            if i == self.max_iterations - 1:
                msgs.append({
                    "role": "user",
                    "content": "Please provide your best final answer now if enough information is available.",
                })

            # Context-length protection: compact or force final answer
            self._ensure_context_fits(msgs, None, None)

            # No native tools; use stop tokens.
            try:
                response = chat_completion_request(
                    model=self.model,
                    messages=msgs,
                    base_url=self.api_base,
                    api_key=self.key,
                    tools=None,
                    tool_choice=None,
                    stop=self.stop_tokens,
                    timeout_seconds=chat_timeout_seconds,
                    **kwargs,
                )
            except RuntimeError as e:
                err_msg = str(e).lower()
                if "context length" in err_msg or "input tokens" in err_msg:
                    self.logger.warning(
                        f"[ToolLoop] Context length error, falling back to vanilla QA: {e}"
                    )
                    fallback_msgs = self._build_vanilla_fallback_msgs(msgs)
                    response = chat_completion_request(
                        model=self.model,
                        messages=fallback_msgs,
                        base_url=self.api_base,
                        api_key=self.key,
                        tools=None,
                        tool_choice=None,
                        stop=None,
                        timeout_seconds=chat_timeout_seconds,
                        **kwargs,
                    )
                else:
                    raise
            if not isinstance(response, dict):
                raise RuntimeError(f"Invalid chat response type: expected dict, got {type(response).__name__}")
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise RuntimeError(
                    f"Invalid chat response payload: missing choices. Response={self._truncate_for_debug(response)}"
                )
            message = choices[0].get("message")
            if not isinstance(message, dict):
                raise RuntimeError(
                    f"Invalid chat response payload: choices[0].message must be an object. "
                    f"Response={self._truncate_for_debug(response)}"
                )
            raw_response = message.get("content")
            raw_response = raw_response if isinstance(raw_response, str) else ""

            # Repair truncated closing tags if generation was stopped by stop token.
            if raw_response and "<tool_call>" in raw_response and "</tool_call>" not in raw_response:
                raw_response += "</tool_call>"
            if debug_tool_loop and raw_response:
                self.logger.info(
                    f"[ToolLoop] assistant_content={self._truncate_for_debug(raw_response, max_chars=debug_char_limit)}"
                )

            text_tool_calls = self._parse_tool_calls_from_text(raw_response, turn_idx=i)

            # Fallback: if text parsing found nothing but the server returned
            # native tool_calls (e.g. vLLM's Qwen tool parser strips
            # <tool_call> from content and puts it in message.tool_calls),
            # convert them so they go through _execute_tool_call as normal.
            if not text_tool_calls:
                native_tc = self._normalize_tool_calls(message.get("tool_calls"), turn_idx=i)
                if native_tc:
                    text_tool_calls = native_tc

            text_tool_calls = self._enforce_single_tool_call_per_message(
                text_tool_calls, turn_idx=i, phase="main",
            )

            parsed_response = self.parse_response(raw_response)

            turn_info = {
                "turn": i + 1,
                "assistant": {
                    "content": raw_response,
                    "perception": parsed_response["perception"],
                    "thinking": parsed_response["thinking"],
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "name": call["name"],
                            "raw_arguments": call["raw_arguments"],
                            "args": call["args"],
                            "parse_error": call.get("parse_error"),
                        }
                        for call in text_tool_calls
                    ],
                },
                "tool_responses": [],
                "memory_outputs": [],
                "memory_turn_message_indices": [],
            }

            if text_tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "content": raw_response,
                    "tool_calls": [call["assistant_tool_call"] for call in text_tool_calls],
                }
                msgs.append(assistant_message)

                deferred_memory_outputs: List[Dict[str, Any]] = []
                for call in text_tool_calls:
                    parse_error = call.get("parse_error")
                    if parse_error:
                        normalized_payload = {
                            "ok": False,
                            "tool_name": call["name"],
                            "result": None,
                            "error": parse_error,
                            "tool_call_id": call["id"],
                            "meta": {"source": "agent.tool_call_parser"},
                        }
                    else:
                        normalized_payload, image_resolution = self._execute_tool_call(
                            call, run_memory, debug_tool_loop, debug_char_limit
                        )
                    turn_info["tool_responses"].append(normalized_payload)
                    memory_result = run_memory.consume_tool_payload(
                        agent=self, turn_idx=i, tool_call=call, payload=normalized_payload,
                    )
                    immediate_messages = memory_result.get("immediate_messages")
                    deferred_messages = memory_result.get("deferred_messages")
                    prepared_tool_payload = memory_result.get("prepared_payload")
                    processor_info = memory_result.get("processor_info")

                    immediate_indices = self._append_messages_with_indices(msgs, immediate_messages)
                    memory_output = {
                        "tool_call_id": call["id"],
                        "tool_name": call["name"],
                        "memory_name": run_memory.memory_name,
                        "immediate_message_indices": immediate_indices,
                        "deferred_message_indices": [],
                        "message_summaries": [
                            self._summarize_message_for_trace(msgs[idx]) for idx in immediate_indices
                        ],
                    }
                    if isinstance(prepared_tool_payload, dict):
                        image_refs = prepared_tool_payload.get("image_refs")
                        if isinstance(image_refs, list) and image_refs:
                            memory_output["image_refs"] = image_refs
                    if isinstance(processor_info, dict) and processor_info:
                        memory_output["tool_output_processor"] = cp.deepcopy(processor_info)
                    turn_info["memory_outputs"].append(memory_output)

                    if isinstance(deferred_messages, list) and deferred_messages:
                        deferred_memory_outputs.append({
                            "memory_output": memory_output,
                            "messages": deferred_messages,
                        })

                # Append deferred messages (e.g. images) from memory.
                if deferred_memory_outputs:
                    for deferred in deferred_memory_outputs:
                        deferred_indices = self._append_messages_with_indices(msgs, deferred["messages"])
                        deferred["memory_output"]["deferred_message_indices"] = deferred_indices
                        deferred["memory_output"]["message_summaries"].extend(
                            [self._summarize_message_for_trace(msgs[idx]) for idx in deferred_indices]
                        )

                turn_info["memory_turn_message_indices"] = self._append_messages_with_indices(
                    msgs,
                    run_memory.emit_after_tool_turn(
                        agent=self, turn_idx=i,
                        tool_calls=text_tool_calls,
                        tool_payloads=turn_info["tool_responses"],
                        messages=msgs,
                    ),
                )
                chat_history.append(turn_info)

                if i == self.max_iterations - 1:
                    if debug_tool_loop:
                        self.logger.info(
                            "[ToolLoop] last iteration ended with tool calls; requesting one final text-only answer."
                        )
                    final_response = chat_completion_request(
                        model=self.model,
                        messages=msgs,
                        base_url=self.api_base,
                        api_key=self.key,
                        tools=None,
                        tool_choice=None,
                        stop=self.stop_tokens,
                        timeout_seconds=chat_timeout_seconds,
                        **kwargs,
                    )
                    if not isinstance(final_response, dict):
                        raise RuntimeError(
                            f"Invalid chat response type: expected dict, got {type(final_response).__name__}"
                        )
                    final_choices = final_response.get("choices")
                    if not isinstance(final_choices, list) or not final_choices or not isinstance(final_choices[0], dict):
                        raise RuntimeError(
                            f"Invalid chat response payload: missing choices. Response={self._truncate_for_debug(final_response)}"
                        )
                    final_message = final_choices[0].get("message")
                    if not isinstance(final_message, dict):
                        raise RuntimeError(
                            "Invalid chat response payload: choices[0].message must be an object. "
                            f"Response={self._truncate_for_debug(final_response)}"
                        )

                    final_raw_response = final_message.get("content")
                    final_raw_response = final_raw_response if isinstance(final_raw_response, str) else ""

                    if final_raw_response and "<tool_call>" in final_raw_response and "</tool_call>" not in final_raw_response:
                        final_raw_response += "</tool_call>"
                    final_text_tool_calls = self._parse_tool_calls_from_text(final_raw_response, turn_idx=i + 1)
                    final_text_tool_calls = self._enforce_single_tool_call_per_message(
                        final_text_tool_calls,
                        turn_idx=i + 1,
                        phase="final_text_only",
                    )
                    final_parsed_response = self.parse_response(final_raw_response)
                    final_answer = self._extract_final_answer_from_content(final_raw_response)

                    if final_text_tool_calls:
                        msgs.append({
                            "role": "assistant",
                            "content": final_raw_response,
                            "tool_calls": [call["assistant_tool_call"] for call in final_text_tool_calls],
                        })
                    else:
                        msgs.append({"role": "assistant", "content": final_raw_response})

                    final_turn_info = {
                        "turn": i + 2,
                        "assistant": {
                            "content": final_raw_response,
                            "perception": final_parsed_response["perception"],
                            "thinking": final_parsed_response["thinking"],
                            "tool_calls": [
                                {
                                    "id": call["id"],
                                    "name": call["name"],
                                    "raw_arguments": call["raw_arguments"],
                                    "args": call["args"],
                                    "parse_error": call.get("parse_error"),
                                }
                                for call in final_text_tool_calls
                            ],
                        },
                        "tool_responses": [],
                        "final_answer": final_answer,
                    }
                    chat_history.append(final_turn_info)

                    status = final_answer if final_answer else self.fail_msg
                    self._last_tool_detections = list(getattr(run_memory, "tool_detections", []) or [])
                    if return_messages:
                        return status, chat_history, msgs, None
                    return status, chat_history
                continue

            final_answer = self._extract_final_answer_from_content(raw_response)
            turn_info["final_answer"] = final_answer
            chat_history.append(turn_info)
            if final_answer:
                self._last_tool_detections = list(getattr(run_memory, "tool_detections", []) or [])
                if return_messages:
                    return final_answer, chat_history, msgs, None
                return final_answer, chat_history

            msgs.append({"role": "assistant", "content": raw_response})

        self._last_tool_detections = list(getattr(run_memory, "tool_detections", []) or [])
        if return_messages:
            return self.fail_msg, chat_history, msgs, None
        return self.fail_msg, chat_history

class BaseAgentWithToolsV3(BaseAgentWithToolsV2):
    """Deferred tool-schema loading agent.

    Only shows tool name + description in the system prompt.  When the model
    wants to use a tool it hasn't seen before it calls it with
    ``{"get_tool_schema": true}`` and the server returns the SKILLS.md (or
    auto-generated schema).  The model then re-calls with real arguments.

    Execution logic is identical to V2 — the server handles the
    ``get_tool_schema`` intercept transparently.
    """

    _ALLOWED_PROMPT_VERSIONS = ("v5",)

    def __init__(self, **kwargs):
        kwargs.setdefault("prompt_version", "v5")
        super().__init__(**kwargs)
        self.stop_tokens = ["</tool_call>"]  # matches post-training
        self.logger = get_logger("BaseAgentWithToolsV3")

    # -------------------------------------------------------------- #
    # Per-run state + schema intercept
    # -------------------------------------------------------------- #

    def _new_persistent_state_memory(self):
        memory = super()._new_persistent_state_memory()
        memory.schema_seen_tools = set()
        return memory

    def _execute_tool_call(self, call, run_memory, debug_tool_loop, debug_char_limit):
        schema_seen = run_memory.schema_seen_tools
        intercept = self._maybe_intercept_for_schema(call, schema_seen)
        if intercept is not None:
            if debug_tool_loop:
                self.logger.info(f"[ToolLoop] auto-schema intercept for '{call['name']}' id={call['id']}")
            return intercept, {"rewrites": [], "errors": []}
        return super()._execute_tool_call(call, run_memory, debug_tool_loop, debug_char_limit)

    # -------------------------------------------------------------- #
    # Prompt construction overrides
    # -------------------------------------------------------------- #

    def _build_tools_directory(self) -> str:
        """Compact directory: one line per canonical tool (name + description)."""
        seen = set()
        lines = []
        for tool_name, metadata in self.tools.items():
            if metadata.get("is_alias"):
                continue
            canonical = metadata.get("canonical_tool_name", tool_name)
            if canonical in seen:
                continue
            seen.add(canonical)
            description = metadata.get("tool_description", "No description.")
            lines.append(f"- {tool_name}: {description}")
        return "\n".join(lines)

    # -------------------------------------------------------------- #
    # Two-tier tool name resolution + auto-schema intercept
    # -------------------------------------------------------------- #

    def _resolve_tool_name(self, raw_name: str) -> str | None:
        """Tier 1 structured match: exact -> case-insensitive -> alias lookup."""
        tools = self.tools
        # Exact
        if raw_name in tools:
            return tools[raw_name].get("canonical_tool_name", raw_name)
        # Case-insensitive
        raw_cf = raw_name.casefold()
        for name, meta in tools.items():
            if name.casefold() == raw_cf:
                return meta.get("canonical_tool_name", name)
        # Alias
        for name, meta in tools.items():
            for alias in (meta.get("tool_aliases") or []):
                if alias.casefold() == raw_cf:
                    return meta.get("canonical_tool_name", name)
        return None

    @staticmethod
    def _normalize_for_scan(s: str) -> str:
        return s.casefold().replace("_", "").replace(" ", "").replace("-", "")

    def _scan_text_for_tool_name(self, text: str) -> str | None:
        """Tier 2: scan raw text for a known tool name. Returns canonical name or None."""
        text_norm = self._normalize_for_scan(text)
        candidates = []
        for name, meta in self.tools.items():
            if meta.get("is_alias"):
                continue
            name_norm = self._normalize_for_scan(name)
            if name_norm in text_norm:
                candidates.append(name)
        if len(candidates) == 1:
            return candidates[0]
        # If multiple, use longest match (most specific)
        if candidates:
            candidates.sort(key=lambda n: len(n), reverse=True)
            return candidates[0]
        return None

    def _fetch_tool_schema(self, tool_name: str, tool_call_id: str) -> dict:
        """Fetch the schema for *tool_name* from the tool server and wrap it."""
        try:
            payload = tool_execute_request(
                base_url=self.tool_base,
                tool_call={
                    "tool_name": "Get_Tool_Schema",
                    "args": {"tool_name": tool_name},
                    "tool_call_id": tool_call_id,
                },
                timeout=self.timeout,
            )
            schema_text = payload.get("result", "") if isinstance(payload, dict) else str(payload)
            result_text = (
                f"[Tool Schema for '{tool_name}']\n{schema_text}\n\n"
                f"Now call {tool_name} again with the correct arguments shown above."
            )
        except Exception as e:
            result_text = f"Schema fetch failed for {tool_name}: {e}. Try calling with your best guess."
        
        return {
            "ok": True,
            "tool_name": tool_name,
            "result": result_text,
            "error": None,
            "tool_call_id": tool_call_id,
            "meta": {"source": "auto_schema_on_first_use"},
        }

    def _maybe_intercept_for_schema(self, call, schema_seen_tools):
        """Auto-return tool schema on first invocation with two-tier name resolution."""
        tool_name = call["name"]
        
        # Tier 1: structured name resolution (if we got a real name from parsing)
        if tool_name != "unknown":
            resolved = self._resolve_tool_name(tool_name)
            if resolved is not None and resolved != tool_name:
                self.logger.info(f"[ToolLoop] Resolved tool name '{tool_name}' -> '{resolved}'")
                call["name"] = resolved
                tool_name = resolved
            elif resolved is None:
                # Name not recognized — fall through to Tier 2
                raw_text = call.get("raw_arguments", "")
                scanned = self._scan_text_for_tool_name(tool_name + " " + raw_text)
                if scanned:
                    self.logger.info(f"[ToolLoop] Scanned tool name '{tool_name}' -> '{scanned}'")
                    call["name"] = scanned
                    tool_name = scanned

        # Tier 2: name is "unknown" (malformed JSON) — scan raw block
        if tool_name == "unknown":
            raw_text = call.get("raw_arguments", "")
            scanned = self._scan_text_for_tool_name(raw_text)
            if scanned:
                self.logger.info(f"[ToolLoop] Scanned unknown call -> '{scanned}'")
                call["name"] = scanned
                tool_name = scanned
            else:
                return None  # can't resolve — let existing error handling deal with it

        # Skip Get_Tool_Schema itself and already-seen tools
        if tool_name == "Get_Tool_Schema" or tool_name in schema_seen_tools:
            return None

        schema_seen_tools.add(tool_name)
        return self._fetch_tool_schema(tool_name, call["id"])

    # V8 uses the same <tool_call> format and natural-language "Final Answer"
    # as V2, so we inherit _parse_tool_calls_from_text and
    # _extract_final_answer_from_content from the parent class.
