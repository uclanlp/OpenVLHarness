import os
import requests
import numpy as np
import json
import time
import random as rd
import copy as cp
from .utils import _get_json, pretty_print_msgs
from .constant import *
from .runtime_environment import RuntimeEnvironmentHook, ImageEnvironment
from ..utils.api_utils import (
    chat_completion_request,
    filter_kwargs_for_endpoint,
    local_image_to_data_url,
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

    def prepare_inputs(self, inputs):
        input_msgs = []
        if self.system_prompt is not None:
            input_msgs.append(dict(role='system', content=self.system_prompt))
        assert isinstance(inputs, list) and isinstance(inputs[0], dict)
        assert np.all(['type' in x for x in inputs]) or np.all(['role' in x for x in inputs]), inputs
        if 'role' in inputs[0]:
            assert inputs[-1]['role'] == 'user', inputs[-1]
            for item in inputs:
                input_msgs.append(dict(role=item['role'], content=self.prepare_itlist(item['content'])))
        else:
            input_msgs.append(dict(role='user', content=self.prepare_itlist(inputs)))
        return input_msgs

    def generate_inner(self, inputs, **kwargs) -> str:
        input_msgs = self.prepare_inputs(inputs)
        temperature = kwargs.pop('temperature', self.temperature)
        max_tokens = kwargs.pop('max_tokens', self.max_tokens)
        top_p = kwargs.pop('top_p', self.top_p)
        repetition_penalty = kwargs.pop('repetition_penalty', self.repetition_penalty)

        headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
        payload = dict(
            model=self.model,
            messages=input_msgs,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            n=1,
            **kwargs)
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

    def preprocess_message_with_role(self, message):
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

        if system_prompt != '':
            if self.system_prompt is None:
                self.system_prompt = system_prompt
            else:
                if system_prompt not in self.system_prompt:
                    self.system_prompt += '\n' + system_prompt
        return new_message

    def _build_trajectory_record(self, input_messages, answer):
        record = {
            "input_messages": input_messages,
            "final_answer": answer,
        }
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
        if self.check_content(message) == 'listdict':
            message = self.preprocess_message_with_role(message)
        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        message = self.preproc_content(message)
        assert message is not None and self.check_content(message) == 'listdict'
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'
        # merge kwargs
        kwargs = cp.deepcopy(self.default_kwargs)
        kwargs.update(kwargs1)

        answer = None
        last_log = None
        last_ret_code = None
        # a very small random delay [0s - 0.5s]
        T = rd.random() * 0.5
        time.sleep(T)

        for i in range(self.retry):
            try:
                ret_code, answer, log = self.generate_inner(cp.deepcopy(message), **kwargs)
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
                    )
                    extra_records = {"trajectory": trajectory}
                    if isinstance(log, dict):
                        extra_records.update(log)
                    return {"prediction": answer, "extra_records": extra_records}
                    if isinstance(log, dict):
                        return {"prediction": answer, "extra_records": log}
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

        final_answer = self.fail_msg if answer in ['', None] else answer
        trajectory = self._build_trajectory_record(
            raw_message,
            final_answer,
        )
        if last_ret_code is not None:
            trajectory["ret_code"] = last_ret_code
        return {"prediction": final_answer, "extra_records": {"trajectory": trajectory}}
    

class BaseAgentWithTools(BaseAgent):
    _DROP_VALUE = object()
    _ALLOWED_PROMPT_VERSIONS = ("v1", "v2")
    _ALLOWED_RUNTIME_ENV_EMIT_POLICIES = ("always", "on_change")
    _IMAGE_REGISTRY_DEFAULT_ALIASES = ("image", "image.jpg", "image.jpeg", "image.png")
    _IMAGE_RESULT_KEY_HINTS = (
        "all_masks_viz_base64",
        "image_base64",
        "overlay_image_base64",
        "zoomed_image_base64",
        "blended_image_base64",
        "cropped_image_base64",
        "edited_image_base64",
        "annotated_image_base64",
        "depth_map_base64",
        "masks_png_b64",
        "subplot_images_base64",
    )

    def __init__(self, 
                tool_base, 
                tool_list = None,
                tool_choice: Any = "auto",
                prompt_version: str = "v1",
                runtime_environment_hooks: Optional[List[Any]] = None,
                runtime_environment_emit_policy: str = "always",
                max_iterations: int = 15, 
                **kwargs):
        super().__init__(**kwargs)
        self.tool_base = tool_base
        self.prompt_version = str(prompt_version).strip().lower()
        if self.prompt_version not in self._ALLOWED_PROMPT_VERSIONS:
            raise ValueError(
                f"Invalid prompt_version '{prompt_version}'. "
                f"Expected one of: {self._ALLOWED_PROMPT_VERSIONS}."
            )
        self.runtime_environment_emit_policy = str(runtime_environment_emit_policy).strip().lower()
        if self.runtime_environment_emit_policy not in self._ALLOWED_RUNTIME_ENV_EMIT_POLICIES:
            raise ValueError(
                f"Invalid runtime_environment_emit_policy '{runtime_environment_emit_policy}'. "
                f"Expected one of: {self._ALLOWED_RUNTIME_ENV_EMIT_POLICIES}."
            )
        self.tools = _get_json(self.tool_base, "/tools")
        if not isinstance(self.tools, dict):
            raise ConnectionError(f"Failed to get tools from {self.tool_base}/tools")
        self.json_schema = _get_json(self.tool_base, "/jsonschema")
        if not isinstance(self.json_schema, list):
            raise ConnectionError(f"Failed to get tool schemas from {self.tool_base}/jsonschema")
        self.tool_choice = tool_choice
        self.max_iterations = max_iterations
        
        if tool_list is not None:
            self.tools = {name: tool for name, tool in self.tools.items() if name in tool_list}
            self.json_schema = [
                x for x in self.json_schema
                if isinstance(x, dict)
                and isinstance(x.get('function'), dict)
                and x['function'].get('name') in tool_list
            ]
            
        self.logger = get_logger("BaseAgentWithTools")
        self.logger.info(f"Available Tools: {list(self.tools.keys())}")
        self.system_prompt = self._construct_system_message()
        self.last_run_messages = []
        self.last_chat_history = []
        self._image_registry = {}
        self._image_alias_to_ref = {}
        self._image_url_to_ref = {}
        self.runtime_environment_hooks = self._build_runtime_environment_hooks(runtime_environment_hooks)
        self._augment_image_path_schema_descriptions()

    def _get_prompt_templates(self) -> tuple[str, str]:
        version = str(getattr(self, "prompt_version", "v1")).strip().lower()
        if version == "v2":
            return DEFAULT_SYSTEM_PROMPT_V2, DEFAULT_USER_PROMPT_PREFIX_V2
        return DEFAULT_SYSTEM_PROMPT_V1, DEFAULT_USER_PROMPT_PREFIX_V1

    def _construct_system_message(self) -> str:
        system_prompt, _ = self._get_prompt_templates()
        return system_prompt

    def _augment_image_path_schema_descriptions(self) -> None:
        if not isinstance(self.json_schema, list):
            return
        suffix = " You may pass a runtime image ID from Environment Update messages instead of a filesystem path."
        for entry in self.json_schema:
            if not isinstance(entry, dict):
                continue
            fn = entry.get("function") or {}
            params = fn.get("parameters") or {}
            properties = params.get("properties") or {}
            if not isinstance(properties, dict):
                continue
            for key, spec in properties.items():
                if not isinstance(spec, dict):
                    continue
                key_lower = str(key).lower()
                if key_lower.endswith("_path") and "image" in key_lower:
                    desc = spec.get("description", "")
                    if suffix.strip() not in str(desc):
                        spec["description"] = f"{desc}{suffix}".strip()

    def _build_runtime_environment_hooks(self, hook_specs: Optional[List[Any]]) -> List[RuntimeEnvironmentHook]:
        specs = hook_specs
        if specs is None:
            specs = ["image"]
        elif isinstance(specs, (str, RuntimeEnvironmentHook, dict)):
            specs = [specs]
        elif not isinstance(specs, list):
            raise ValueError(
                "runtime_environment_hooks must be null, a hook name, a hook object, "
                "a hook config dict, or a list of these values."
            )

        hooks: List[RuntimeEnvironmentHook] = []
        for spec in specs:
            hook = self._build_single_runtime_environment_hook(spec)
            if hook is not None:
                hooks.append(hook)
        return hooks

    def _build_single_runtime_environment_hook(self, spec: Any) -> Optional[RuntimeEnvironmentHook]:
        if isinstance(spec, RuntimeEnvironmentHook):
            return spec

        options: Dict[str, Any] = {}
        if isinstance(spec, dict):
            name = spec.get("name")
            if "kwargs" in spec and isinstance(spec.get("kwargs"), dict):
                options.update(spec["kwargs"])
            for key, value in spec.items():
                if key not in {"name", "kwargs"}:
                    options[key] = value
        else:
            name = spec

        if not isinstance(name, str):
            raise ValueError(
                f"Invalid runtime environment hook spec '{spec}'. "
                "Expected string name, hook object, or dict with a 'name' field."
            )

        hook_name = name.strip().lower()
        if hook_name in {"", "none", "off", "disabled"}:
            return None

        if hook_name == "image":
            emit_policy = options.get("emit_policy", self.runtime_environment_emit_policy)
            return ImageEnvironment(emit_policy=emit_policy)

        raise ValueError(
            f"Unknown runtime environment hook '{name}'. Supported hooks: ['image']."
        )

    def _reset_runtime_environment_hooks(self) -> None:
        hooks = getattr(self, "runtime_environment_hooks", [])
        if not isinstance(hooks, list):
            return
        for hook in hooks:
            reset = getattr(hook, "reset", None)
            if callable(reset):
                reset()

    def _run_runtime_environment_hook_stage(
        self,
        stage: str,
        *,
        msgs: List[Dict[str, Any]],
        question: Any = None,
        turn_idx: Optional[int] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        hooks = getattr(self, "runtime_environment_hooks", [])
        if not isinstance(hooks, list):
            return out

        for hook in hooks:
            produced = []
            if stage == "after_initial_user_message":
                produced = hook.after_initial_user_message(
                    agent=self,
                    question=question,
                    messages=msgs,
                )
            elif stage == "after_tool_turn":
                produced = hook.after_tool_turn(
                    agent=self,
                    turn_idx=turn_idx if turn_idx is not None else -1,
                    tool_calls=tool_calls or [],
                    tool_payloads=tool_payloads or [],
                    messages=msgs,
                )
            if isinstance(produced, dict):
                produced = [produced]
            if not isinstance(produced, list):
                continue
            for message in produced:
                if (
                    isinstance(message, dict)
                    and message.get("role") == "user"
                    and isinstance(message.get("content"), str)
                ):
                    out.append(message)
        return out

    def _append_runtime_environment_messages(
        self,
        stage: str,
        *,
        msgs: List[Dict[str, Any]],
        question: Any = None,
        turn_idx: Optional[int] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        tool_payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        env_messages = self._run_runtime_environment_hook_stage(
            stage,
            msgs=msgs,
            question=question,
            turn_idx=turn_idx,
            tool_calls=tool_calls,
            tool_payloads=tool_payloads,
        )
        if env_messages:
            msgs.extend(env_messages)

    def _normalize_image_alias(self, alias: str) -> str:
        if not isinstance(alias, str):
            return ""
        return alias.strip().lower()

    def _register_image_data_url(
        self,
        ref_id: str,
        data_url: str,
        *,
        source: str,
        aliases: Optional[List[str]] = None,
    ) -> Optional[str]:
        if not isinstance(data_url, str):
            return None
        data_url = data_url.strip()
        if not data_url.startswith("data:image/"):
            return None

        existing_ref = self._image_url_to_ref.get(data_url)
        if existing_ref and existing_ref in self._image_registry:
            entry = self._image_registry[existing_ref]
            for alias in aliases or []:
                norm_alias = self._normalize_image_alias(alias)
                if not norm_alias:
                    continue
                self._image_alias_to_ref[norm_alias] = existing_ref
                entry["aliases"].add(alias)
            return existing_ref

        entry = {
            "id": ref_id,
            "data_url": data_url,
            "source": source,
            "aliases": set(),
        }
        self._image_registry[ref_id] = entry
        self._image_url_to_ref[data_url] = ref_id

        for alias in [ref_id] + list(aliases or []):
            norm_alias = self._normalize_image_alias(alias)
            if not norm_alias:
                continue
            self._image_alias_to_ref[norm_alias] = ref_id
            entry["aliases"].add(alias)

        return ref_id

    def _build_image_registry_from_inputs(self, question) -> Dict[str, Dict[str, Any]]:
        """Build runtime image IDs from the current user input payload.

        Registers each incoming image as `input_image_<n>` (plus aliases) so
        model-generated `*_image_path` arguments can be rewritten to
        `*_image_base64` before `/execute`.
        """
        self._image_registry = {}
        self._image_alias_to_ref = {}
        self._image_url_to_ref = {}
        if not isinstance(question, list):
            return self._image_registry

        image_items = []
        for item in question:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image" and isinstance(item.get("value"), str):
                image_items.append(item.get("value").strip())

        for idx, image_value in enumerate(image_items, 1):
            ref_id = f"input_image_{idx}"
            aliases = [ref_id, f"image_{idx}"]
            data_url = None
            if image_value.startswith("data:image/"):
                data_url = image_value
            elif self._is_likely_base64(image_value):
                raw = image_value
                if raw.startswith("base64:"):
                    raw = raw.split(":", 1)[1]
                data_url = f"data:image/png;base64,{raw}"
            else:
                aliases.append(image_value)
                aliases.append(os.path.basename(image_value))
                try:
                    data_url = local_image_to_data_url(image_value)
                except Exception:
                    data_url = None

            registered = self._register_image_data_url(
                ref_id,
                data_url,
                source="user_input",
                aliases=aliases,
            )
            if registered and len(image_items) == 1:
                for alias in self._IMAGE_REGISTRY_DEFAULT_ALIASES:
                    self._image_alias_to_ref[self._normalize_image_alias(alias)] = registered
                    self._image_registry[registered]["aliases"].add(alias)

        return self._image_registry

    def _resolve_registry_image_data_url(self, image_ref: str):
        """Resolve an image reference into data URL + optional matched metadata."""
        if not isinstance(image_ref, str):
            return None, None, None

        raw = image_ref.strip()
        if not raw:
            return None, None, None

        if raw.startswith("data:image/"):
            return raw, None, None

        if self._is_likely_base64(raw):
            if raw.startswith("base64:"):
                raw = raw.split(":", 1)[1]
            return f"data:image/png;base64,{raw}", None, None

        alias_candidates = [self._normalize_image_alias(raw)]
        alias_candidates.append(self._normalize_image_alias(os.path.basename(raw)))
        for alias in alias_candidates:
            ref_id = self._image_alias_to_ref.get(alias)
            if ref_id and ref_id in self._image_registry:
                return self._image_registry[ref_id]["data_url"], ref_id, None

        if os.path.isfile(raw):
            return None, None, None

        if len(self._image_registry) == 1:
            only_ref = next(iter(self._image_registry.keys()))
            return (
                self._image_registry[only_ref]["data_url"],
                only_ref,
                "single_image_registry_default",
            )

        return None, None, None

    def _is_image_path_key(self, key: str) -> bool:
        key_lower = str(key).lower()
        return key_lower.endswith("_path") and "image" in key_lower

    def _resolve_image_paths_in_args(
        self,
        value: Any,
        *,
        path: str,
        rewrites: List[Dict[str, Any]],
        errors: List[Dict[str, Any]],
    ) -> None:
        """Recursively rewrite `*_image_path` fields to `*_image_base64`.

        All successful rewrites and unresolved references are collected in
        `rewrites/errors` for transparent downstream debugging.
        """
        if isinstance(value, dict):
            for key, child in list(value.items()):
                child_path = f"{path}.{key}"
                if isinstance(child, (dict, list)):
                    self._resolve_image_paths_in_args(
                        child,
                        path=child_path,
                        rewrites=rewrites,
                        errors=errors,
                    )

            for path_key in list(value.keys()):
                if not self._is_image_path_key(path_key):
                    continue
                base64_key = f"{path_key[:-5]}_base64"
                if value.get(base64_key):
                    continue

                raw_path = value.get(path_key)
                if not isinstance(raw_path, str) or not raw_path.strip():
                    continue

                data_url, matched_ref, fallback_mode = self._resolve_registry_image_data_url(raw_path)
                if data_url:
                    value[base64_key] = data_url
                    value.pop(path_key, None)
                    rewrite_item = {
                        "field": f"{path}.{path_key}",
                        "target_field": f"{path}.{base64_key}",
                        "original": raw_path,
                    }
                    if matched_ref:
                        rewrite_item["image_ref"] = matched_ref
                    if fallback_mode:
                        rewrite_item["fallback_mode"] = fallback_mode
                    rewrites.append(rewrite_item)
                elif not os.path.isfile(raw_path):
                    errors.append(
                        {
                            "field": f"{path}.{path_key}",
                            "value": raw_path,
                            "reason": "unresolved_image_reference",
                            "available_image_ids": list(self._image_registry.keys()),
                        }
                    )
            return

        if isinstance(value, list):
            for idx, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    self._resolve_image_paths_in_args(
                        child,
                        path=f"{path}[{idx}]",
                        rewrites=rewrites,
                        errors=errors,
                    )

    def _resolve_tool_call_args(self, args: Dict[str, Any]):
        """Return rewritten tool args plus image-resolution metadata."""
        if not isinstance(args, dict):
            return {}, {"rewrites": [], "errors": []}

        resolved = cp.deepcopy(args)
        rewrites = []
        errors = []
        self._resolve_image_paths_in_args(
            resolved,
            path="args",
            rewrites=rewrites,
            errors=errors,
        )
        return resolved, {"rewrites": rewrites, "errors": errors}

    def _sanitize_ref_token(self, token: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_]+", "_", str(token or "tool"))
        return safe.strip("_") or "tool"

    def _register_tool_output_images(self, payload: Dict[str, Any], image_parts: List[Dict[str, Any]]) -> List[str]:
        """Register tool-output images as reusable runtime image references."""
        refs = []
        call_id = self._sanitize_ref_token(payload.get("tool_call_id") or "tool")
        tool_name = self._sanitize_ref_token(payload.get("tool_name") or "tool")
        for idx, part in enumerate(image_parts, 1):
            if not isinstance(part, dict):
                continue
            image_url = (part.get("image_url") or {}).get("url")
            ref_id = f"tool_{call_id}_image_{idx}"
            assigned_ref = self._register_image_data_url(
                ref_id,
                image_url,
                source=f"tool_output:{tool_name}",
                aliases=[ref_id, f"{tool_name}_image_{idx}"],
            )
            if assigned_ref:
                refs.append(assigned_ref)
        return refs
    
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

    def _is_likely_base64(self, value: str) -> bool:
        if not isinstance(value, str):
            return False
        value = value.strip()
        if len(value) < 64:
            return False
        if value.startswith("data:image/"):
            return True
        if value.startswith("base64:"):
            value = value.split(":", 1)[1]
        return bool(re.fullmatch(r"[A-Za-z0-9+/=\s]+", value))

    def _is_image_result_key(self, key: str) -> bool:
        k = str(key).lower()
        if k in self._IMAGE_RESULT_KEY_HINTS:
            return True
        if k.endswith("_base64") or k.endswith("_b64"):
            return any(
                hint in k
                for hint in (
                    "image",
                    "viz",
                    "mask",
                    "depth",
                    "overlay",
                    "edited",
                    "cropped",
                    "subplot",
                )
            )
        return False

    def _guess_image_mime(self, key: str) -> str:
        k = str(key).lower()
        if "jpg" in k or "jpeg" in k:
            return "image/jpeg"
        return "image/png"

    def _to_image_data_url(self, key: str, value: str) -> Optional[str]:
        if not isinstance(value, str):
            return None
        raw = value.strip()
        if raw.startswith("data:image/"):
            return raw
        if raw.startswith("base64:"):
            raw = raw.split(":", 1)[1]
        if not self._is_likely_base64(raw):
            return None
        mime = self._guess_image_mime(key)
        return f"data:{mime};base64,{raw}"

    def _extract_images_and_scrub_result(
        self,
        value: Any,
        image_parts: List[Dict[str, Any]],
        *,
        key_hint: str = "",
        image_context: bool = False,
    ):
        if isinstance(value, dict):
            compact = {}
            for key, item in value.items():
                child_context = image_context or self._is_image_result_key(key)
                compact_item = self._extract_images_and_scrub_result(
                    item,
                    image_parts,
                    key_hint=str(key),
                    image_context=child_context,
                )
                if compact_item is not self._DROP_VALUE:
                    compact[key] = compact_item
            if not compact and image_context:
                return self._DROP_VALUE
            return compact

        if isinstance(value, list):
            compact_list = []
            for item in value:
                compact_item = self._extract_images_and_scrub_result(
                    item,
                    image_parts,
                    key_hint=key_hint,
                    image_context=image_context,
                )
                if compact_item is not self._DROP_VALUE:
                    compact_list.append(compact_item)
            if not compact_list and image_context:
                return self._DROP_VALUE
            return compact_list

        if isinstance(value, str):
            raw = value.strip()
            is_explicit_image = raw.startswith("data:image/")
            should_try_image = is_explicit_image or image_context or self._is_image_result_key(key_hint)
            if should_try_image:
                url = self._to_image_data_url(key_hint or "image", raw)
                if url:
                    image_parts.append({"type": "image_url", "image_url": {"url": url}})
                    return self._DROP_VALUE

        return value

    def _extract_image_parts(self, result):
        """Split tool result into image parts and a scrubbed non-image payload."""
        image_parts = []
        compact_result = self._extract_images_and_scrub_result(
            result,
            image_parts,
            key_hint="result",
            image_context=False,
        )
        if compact_result is self._DROP_VALUE:
            compact_result = None
        return image_parts, compact_result

    def _prepare_tool_message_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare compact text payload plus extracted image parts from tool output."""
        result = payload.get("result")
        image_parts, compact_result = self._extract_image_parts(result)
        image_refs = self._register_tool_output_images(payload, image_parts) if image_parts else []
        if image_parts:
            result_for_text = compact_result
        else:
            result_for_text = compact_result if isinstance(result, (dict, list)) else result
        compact_payload = {
            "ok": payload.get("ok"),
            "result": result_for_text,
            "error": payload.get("error"),
        }
        if image_refs:
            compact_payload["image_refs"] = image_refs

        try:
            text_payload = json.dumps(compact_payload, ensure_ascii=False)
        except Exception:
            text_payload = str(compact_payload)

        return {
            "text_payload": text_payload,
            "image_parts": image_parts,
            "image_refs": image_refs,
        }

    def _build_tool_message_content(
        self,
        payload: Dict[str, Any],
        *,
        prepared_payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build strict text-only `role=tool` message content."""
        prepared = prepared_payload or self._prepare_tool_message_payload(payload)
        text_payload = prepared.get("text_payload")
        if isinstance(text_payload, str):
            return text_payload
        try:
            return json.dumps(text_payload, ensure_ascii=False)
        except Exception:
            return str(text_payload)

    def _build_tool_image_user_message(
        self,
        payload: Dict[str, Any],
        *,
        prepared_payload: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Build optional `role=user` multimodal update carrying tool-output images."""
        prepared = prepared_payload or self._prepare_tool_message_payload(payload)
        image_parts = prepared.get("image_parts")
        if not isinstance(image_parts, list) or not image_parts:
            return None

        image_refs = prepared.get("image_refs")
        refs = image_refs if isinstance(image_refs, list) else []
        tool_name = str(payload.get("tool_name") or "tool").strip()
        tool_call_id = str(payload.get("tool_call_id") or "unknown").strip()
        lines = [
            f"Tool `{tool_name}` (tool_call_id={tool_call_id}) also returned the following image(s). Use these images for reasoning if helpful.",
            "See the preceding role=tool message for structured output.",
        ]
        if refs:
            lines.append(f"Image refs: {', '.join(refs)}")

        return {
            "role": "user",
            "content": [{"type": "text", "text": "\n".join(lines)}] + image_parts,
        }

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
        if prompt_version == "v2":
            action_match = re.search(
                r"^\s*Action\s*:\s*Final\s*Answer\s*$",
                content,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            if action_match:
                tail = content[action_match.end():]
                action_input_match = re.search(
                    r"^\s*Action\s*Input\s*:\s*(.*)$",
                    tail,
                    flags=re.IGNORECASE | re.MULTILINE,
                )
                if action_input_match:
                    inline_answer = action_input_match.group(1).strip()
                    if inline_answer:
                        return inline_answer
                    trailing = tail[action_input_match.end():].strip()
                    if trailing:
                        return trailing
                    return ""

            v2_scaffold_line = r"^\s*(Question|Observation|Thought|Action|Action\s+Input)\s*:\s*.*$"
            if re.search(v2_scaffold_line, content, flags=re.IGNORECASE | re.MULTILINE):
                non_scaffold = re.sub(v2_scaffold_line, "", content, flags=re.IGNORECASE | re.MULTILINE).strip()
                if not non_scaffold:
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

    def _truncate_for_debug(self, value: Any, max_chars: int = 280) -> str:
        try:
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        except Exception:
            text = str(value)
        text = text.replace("\n", "\\n")
        if len(text) <= max_chars:
            return text
        return f"{text[:max_chars]}... (truncated)"

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def _run(self, question, **kwargs) -> list[dict]:
        """
        Run the native tool-calling loop.

        Flow: `chat/completions` -> `message.tool_calls` -> `/execute` ->
        append `role=tool` results (with `tool_call_id`) -> continue until a
        final answer is produced or `max_iterations` is reached.
        """
        debug_tool_loop = bool(kwargs.pop("debug_tool_loop", False))
        debug_char_limit = int(kwargs.pop("debug_char_limit", 280))
        chat_timeout_seconds = kwargs.pop("timeout_seconds", self.timeout)
        tool_choice = kwargs.pop("tool_choice", self.tool_choice)
        runtime_tools = kwargs.pop("tools", None)
        model_tools = self.json_schema
        if isinstance(runtime_tools, list):
            model_tools = runtime_tools
        elif runtime_tools is not None and debug_tool_loop:
            self.logger.info("[ToolLoop] Ignoring non-list runtime tools override; using agent tool schema.")

        self._reset_runtime_environment_hooks()
        self._build_image_registry_from_inputs(question)
        msgs = self.prepare_inputs(question)
        # add available tools to user prompt
        msgs = self._add_user_prompt_prefix(msgs)
        self._append_runtime_environment_messages(
            "after_initial_user_message",
            msgs=msgs,
            question=question,
        )

        # Use this for debugging the input messages
        # pretty_print_msgs(msgs)
        chat_history = []
        final_answer = None
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
            }

            if native_tool_calls:
                assistant_message = {
                    "role": "assistant",
                    "content": message.get("content") or "",
                    "tool_calls": [call["assistant_tool_call"] for call in native_tool_calls],
                }
                msgs.append(assistant_message)

                tool_image_user_messages: List[Dict[str, Any]] = []
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
                        resolved_args, image_resolution = self._resolve_tool_call_args(call["args"])
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
                    if debug_tool_loop:
                        self.logger.info(
                            f"[ToolLoop] result ok={normalized_payload.get('ok')} id={call['id']} error={self._truncate_for_debug(normalized_payload.get('error'), max_chars=debug_char_limit)}"
                        )

                    prepared_tool_payload = self._prepare_tool_message_payload(normalized_payload)
                    tool_message_content = self._build_tool_message_content(
                        normalized_payload,
                        prepared_payload=prepared_tool_payload,
                    )
                    if debug_tool_loop:
                        self.logger.info(
                            f"[ToolLoop] tool_message text={self._truncate_for_debug(tool_message_content, max_chars=debug_char_limit)}"
                        )
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": call["name"],
                        "content": tool_message_content,
                    }
                    msgs.append(tool_message)
                    tool_image_user_message = self._build_tool_image_user_message(
                        normalized_payload,
                        prepared_payload=prepared_tool_payload,
                    )
                    if tool_image_user_message:
                        tool_image_user_messages.append(tool_image_user_message)
                    turn_info["tool_responses"].append(normalized_payload)

                if tool_image_user_messages:
                    msgs.extend(tool_image_user_messages)
                    if debug_tool_loop:
                        image_count = sum(
                            1
                            for msg in tool_image_user_messages
                            for part in (msg.get("content") or [])
                            if isinstance(part, dict) and part.get("type") == "image_url"
                        )
                        self.logger.info(
                            f"[ToolLoop] appended user tool-image updates={len(tool_image_user_messages)} image_parts={image_count}"
                        )

                self._append_runtime_environment_messages(
                    "after_tool_turn",
                    msgs=msgs,
                    turn_idx=i,
                    tool_calls=native_tool_calls,
                    tool_payloads=turn_info["tool_responses"],
                )
                chat_history.append(turn_info)
                self.last_run_messages = cp.deepcopy(msgs)
                self.last_chat_history = cp.deepcopy(chat_history)
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
                    final_parsed_response = self.parse_response(final_raw_response)
                    final_answer = self._extract_final_answer_from_content(final_raw_response)

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
                    self.last_run_messages = cp.deepcopy(msgs)
                    self.last_chat_history = cp.deepcopy(chat_history)

                    if final_answer:
                        return final_answer, chat_history
                    return self.fail_msg, chat_history
                continue

            final_answer = self._extract_final_answer_from_content(raw_response)
            turn_info["final_answer"] = final_answer
            chat_history.append(turn_info)
            self.last_run_messages = cp.deepcopy(msgs)
            self.last_chat_history = cp.deepcopy(chat_history)
            if debug_tool_loop:
                self.logger.info(
                    f"[ToolLoop] final_answer={self._truncate_for_debug(final_answer, max_chars=debug_char_limit)}"
                )
            if final_answer:
                return final_answer, chat_history

            msgs.append({"role": "assistant", "content": raw_response})
            self.last_run_messages = cp.deepcopy(msgs)
            self.last_chat_history = cp.deepcopy(chat_history)

        self.last_run_messages = cp.deepcopy(msgs)
        self.last_chat_history = cp.deepcopy(chat_history)
        return self.fail_msg, chat_history

    def generate(self, message, **kwargs1):
        """Generate one final answer using the native tool-calling loop."""

        raw_message = cp.deepcopy(message)
        message = cp.deepcopy(message)
        if self.check_content(message) == 'listdict':
            message = self.preprocess_message_with_role(message)
        assert self.check_content(message) in ['str', 'dict', 'liststr', 'listdict'], f'Invalid input type: {message}'
        message = self.preproc_content(message)
        assert message is not None and self.check_content(message) == 'listdict'
        for item in message:
            assert item['type'] in self.allowed_types, f'Invalid input type: {item["type"]}'

        # merge kwargs
        kwargs = cp.deepcopy(self.default_kwargs)
        kwargs.update(kwargs1)
        kwargs = filter_kwargs_for_endpoint(kwargs, "chat")

        # Run the ReAct-style loop
        answer, chat_history = self._run(message, **kwargs)

        final_answer = self.fail_msg if answer in ['', None] else answer
        trajectory = {
            "input_messages": raw_message,
            "chat_history": chat_history,
            "final_answer": final_answer,
        }
        return {"prediction": final_answer, "extra_records": {"trajectory": trajectory}}
