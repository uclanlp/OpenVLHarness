import os
import requests
import numpy as np
import json
import time
import random as rd
import copy as cp
from .utils import _get_json, pretty_print_msgs
from .constant import DEFAULT_SYSTEM_PROMPT_V1, DEFAULT_USER_PROMPT_PREFIX_V1
from ..utils.api_utils import chat_completion_request, filter_kwargs_for_endpoint, tool_execute_request
from typing import Any, Dict, List, Optional
import re

try:
    from ..smp import get_logger, encode_image_to_base64, parse_file
except Exception:  # pragma: no cover - optional heavy deps may be missing for lightweight usage/tests.
    import base64
    import logging
    from io import BytesIO

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

    def parse_file(value):
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

        # check if server is reachable
        response = requests.get(base + "/v1/models", timeout=20)
        if response.status_code != 200:
            raise ConnectionError(f"Unable to connect to the server at {base}. Please check the server status.")
        
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
                ret_code, answer, log = self.chat_inner(messages, **kwargs)
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
            role = data.pop('role', 'user')
            if role == 'system':
                system_prompt += data['value'] + '\n'
            else:
                new_message.append(data)

        if system_prompt != '':
            if self.system_prompt is None:
                self.system_prompt = system_prompt
            else:
                if system_prompt not in self.system_prompt:
                    self.system_prompt += '\n' + system_prompt
        return new_message

    def generate(self, message, **kwargs1):
        """The main function to generate the answer. Will call `generate_inner` with the preprocessed input messages.

        Args:
            message: raw input messages.

        Returns:
            str: The generated answer of the Failed Message if failed to obtain answer.
        """
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
        # a very small random delay [0s - 0.5s]
        T = rd.random() * 0.5
        time.sleep(T)

        for i in range(self.retry):
            try:
                ret_code, answer, log = self.generate_inner(message, **kwargs)
                if ret_code == 0 and self.fail_msg not in answer and answer != '':
                    if self.verbose:
                        print(answer)
                    # Return both answer and extra_records
                    # Add for agent evaluation
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

        return self.fail_msg if answer in ['', None] else answer
    

class BaseAgentWithTools(BaseAgent):
    _DROP_VALUE = object()
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
                max_iterations: int = 15, 
                **kwargs):
        super().__init__(**kwargs)
        self.tool_base = tool_base
        self.tools = _get_json(self.tool_base, "/tools")
        self.logger.info(f"Available Tools: {list(self.tools.keys())}")
        self.json_schema = _get_json(self.tool_base, "/jsonschema")
        self.max_iterations = max_iterations
        
        if tool_list is not None:
            self.tools = {name: tool for name, tool in self.tools.items() if name in tool_list}
            self.json_schema = [x for x in self.json_schema if x['function']['name'] in tool_list]
        
        if not isinstance(self.tools, dict):
            raise ValueError(f"Failed to get tools from {self.tool_base}/tools")
        self.system_prompt = self._construct_system_message()
        self.last_run_messages = []
        self.last_chat_history = []

    def _construct_system_message(self) -> str:
        return DEFAULT_SYSTEM_PROMPT_V1
    
    def _add_user_prompt_prefix(self, msgs) -> str:
        if not msgs or msgs[-1].get("role") != "user":
            return msgs

        last_content = msgs[-1].get("content")
        if isinstance(last_content, list):
            for item in last_content:
                if isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                    item["text"] = DEFAULT_USER_PROMPT_PREFIX_V1.format(user_query=item["text"])
                    break
        elif isinstance(last_content, str):
            msgs[-1]["content"] = DEFAULT_USER_PROMPT_PREFIX_V1.format(user_query=last_content)
        elif isinstance(last_content, dict) and "text" in last_content:
            last_content["text"] = DEFAULT_USER_PROMPT_PREFIX_V1.format(user_query=last_content["text"])
        return msgs
    
    def _extract_tagged_content(self, text: str, tag: str) -> str:
        """Extract content between XML-style tags."""
        pattern = rf"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
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

            if isinstance(raw_args, dict):
                args_dict = raw_args
                args_text = json.dumps(raw_args, ensure_ascii=False)
            elif isinstance(raw_args, str):
                args_text = raw_args
                if raw_args.strip():
                    try:
                        parsed_args = json.loads(raw_args)
                    except Exception:
                        parsed_args = {}
                    if isinstance(parsed_args, dict):
                        args_dict = parsed_args
            else:
                args_text = "{}"
                args_dict = {}

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
                }
            )
        return normalized

    def _normalize_execute_payload(self, payload, tool_name: str, tool_call_id: str) -> Dict[str, Any]:
        if isinstance(payload, dict):
            if "ok" in payload and ("result" in payload or "error" in payload):
                return {
                    "ok": bool(payload.get("ok")),
                    "tool_name": payload.get("tool_name", tool_name),
                    "result": payload.get("result"),
                    "error": payload.get("error"),
                    "tool_call_id": payload.get("tool_call_id", tool_call_id),
                    "meta": payload.get("meta", {}),
                }
            if "result" in payload and ("tool_name" in payload or "error" in payload):
                return {
                    "ok": payload.get("error") in (None, "", {}),
                    "tool_name": payload.get("tool_name", tool_name),
                    "result": payload.get("result"),
                    "error": payload.get("error"),
                    "tool_call_id": payload.get("tool_call_id", tool_call_id),
                    "meta": payload.get("meta", {}),
                }
            return {
                "ok": True,
                "tool_name": tool_name,
                "result": payload,
                "error": None,
                "tool_call_id": tool_call_id,
                "meta": {},
            }

        return {
            "ok": True,
            "tool_name": tool_name,
            "result": payload,
            "error": None,
            "tool_call_id": tool_call_id,
            "meta": {},
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

    def _build_tool_message_content(self, payload: Dict[str, Any]):
        result = payload.get("result")
        image_parts, compact_result = self._extract_image_parts(result)
        compact_payload = {
            "ok": payload.get("ok"),
            "result": compact_result if isinstance(result, dict) else result,
            "error": payload.get("error"),
        }

        try:
            text_payload = json.dumps(compact_payload, ensure_ascii=False)
        except Exception:
            text_payload = str(compact_payload)

        if not image_parts:
            return text_payload

        return [{"type": "text", "text": text_payload}] + image_parts

    def _extract_final_answer_from_content(self, content: str) -> str:
        if content is None:
            return ""
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except Exception:
                content = str(content)

        cleaned = re.sub(r"<perception>.*?</perception>", "", content, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = cleaned.strip()
        if cleaned.lower().startswith("final answer:"):
            cleaned = cleaned.split(":", 1)[1].strip()
        return cleaned or content.strip()

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
        Run the ReAct-style loop with OpenAI Function Calling.
        """
        debug_tool_loop = bool(kwargs.pop("debug_tool_loop", False))
        debug_char_limit = int(kwargs.pop("debug_char_limit", 280))

        msgs = self.prepare_inputs(question)
        # add available tools to user prompt
        msgs = self._add_user_prompt_prefix(msgs)

        # Use this for debugging the input messages
        # pretty_print_msgs(msgs)
        chat_history = []
        final_answer = "None"
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
                tools=self.json_schema,
                tool_choice="auto",
                **kwargs,
            )
            message = (response.get("choices") or [{}])[0].get("message", {})
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

                for call in native_tool_calls:
                    if debug_tool_loop:
                        self.logger.info(
                            f"[ToolLoop] execute {call['name']} id={call['id']} args={self._truncate_for_debug(call['args'], max_chars=debug_char_limit)}"
                        )
                    execute_request = {
                        "tool_name": call["name"],
                        "args": call["args"],
                        "tool_call_id": call["id"],
                    }
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
                    if debug_tool_loop:
                        self.logger.info(
                            f"[ToolLoop] result ok={normalized_payload.get('ok')} id={call['id']} error={self._truncate_for_debug(normalized_payload.get('error'), max_chars=debug_char_limit)}"
                        )

                    tool_message_content = self._build_tool_message_content(normalized_payload)
                    if debug_tool_loop:
                        if isinstance(tool_message_content, list):
                            image_count = sum(
                                1
                                for part in tool_message_content
                                if isinstance(part, dict) and part.get("type") == "image_url"
                            )
                            self.logger.info(
                                f"[ToolLoop] tool_message multimodal parts={len(tool_message_content)} image_parts={image_count}"
                            )
                        else:
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
                    turn_info["tool_responses"].append(normalized_payload)

                chat_history.append(turn_info)
                self.last_run_messages = cp.deepcopy(msgs)
                self.last_chat_history = cp.deepcopy(chat_history)
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
        return final_answer, chat_history

    def generate(self, message, **kwargs1):

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

        return self.fail_msg if answer in ['', None] else answer
