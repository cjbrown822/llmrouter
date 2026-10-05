"""
OpenClaw Router Server
======================
OpenAI-compatible API server with intelligent LLM routing.

Usage:
    llmrouter serve --config configs/openclaw_example.yaml

Or directly:
    python server.py --config config.yaml
"""

import json
import hashlib
import os
import re
import sys
import time
import uuid
from collections import OrderedDict
from typing import AsyncGenerator, AsyncIterator, Optional, Dict, Any, List, Tuple

# Check dependencies
try:
    from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    import httpx
    import uvicorn
except ImportError:
    print("Please install: pip install fastapi uvicorn httpx pydantic")
    sys.exit(1)

# Handle both relative and direct imports
try:
    from .config import OpenClawConfig, LLMConfig, MODELS_WITHOUT_SYSTEM_ROLE, MODEL_CONTEXT_LIMITS
    from .routers import OpenClawRouter, _safe_log
    from .media import process_multimodal_content, MediaConfig
except ImportError:
    from config import OpenClawConfig, LLMConfig, MODELS_WITHOUT_SYSTEM_ROLE, MODEL_CONTEXT_LIMITS
    from routers import OpenClawRouter, _safe_log
    from media import process_multimodal_content, MediaConfig


# ============================================================
# Request/Response Models
# ============================================================

class Message(BaseModel):
    role: str
    content: Optional[Any] = None  # Can be string or list (multimodal)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = 4096
    stream: Optional[bool] = False
    user: Optional[str] = None  # Optional user id (used for memory scoping if enabled)
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    stream_options: Optional[Dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = None


class ResponsesRequest(BaseModel):
    """Subset of the Responses API request accepted by Codex.

    Unknown fields are intentionally ignored by Pydantic so newer Codex clients
    can add metadata without breaking older router deployments.
    """

    model: str = "auto"
    input: Any = ""
    instructions: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    stream: bool = False
    store: Optional[bool] = False
    user: Optional[str] = None
    safety_identifier: Optional[str] = None
    prompt_cache_key: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    reasoning: Optional[Dict[str, Any]] = None
    text: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    previous_response_id: Optional[str] = None
    truncation: Optional[str] = None


# ============================================================
# Message Processing
# ============================================================

def normalize_content(content: Any) -> str:
    """Convert multimodal content to plain string"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") in {"image_url", "image"}:
                    text_parts.append("[Image input]")
                elif part.get("type") in {"audio", "input_audio"}:
                    text_parts.append("[Audio input]")
                elif part.get("type") == "video":
                    text_parts.append("[Video input]")
                elif "text" in part:
                    text_parts.append(part.get("text", ""))
            elif isinstance(part, str):
                text_parts.append(part)
        return "\n".join(text_parts)
    return str(content) if content else ""


def normalize_messages(messages: List[Dict], model_id: str = "") -> List[Dict]:
    """Normalize message format for compatibility"""
    normalized = []
    system_content = ""

    for msg in messages:
        role = msg.get("role", "user")
        content = normalize_content(msg.get("content", ""))
        normalized_msg = {"role": role, "content": content}

        if msg.get("tool_calls") is not None:
            normalized_msg["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id") is not None:
            normalized_msg["tool_call_id"] = msg["tool_call_id"]
        if msg.get("function_call") is not None:
            normalized_msg["function_call"] = msg["function_call"]

        if role == "system":
            system_content = content
        else:
            normalized.append(normalized_msg)

    # Handle models without system role support
    if system_content and model_id in MODELS_WITHOUT_SYSTEM_ROLE:
        if normalized and normalized[0]["role"] == "user":
            normalized[0]["content"] = f"[System Instructions]\n{system_content}\n\n[User Message]\n{normalized[0]['content']}"
        else:
            normalized.insert(0, {"role": "user", "content": f"[System Instructions]\n{system_content}"})
    elif system_content:
        normalized.insert(0, {"role": "system", "content": system_content})

    return normalized


def estimate_tokens(text: str) -> int:
    """Estimate token count (approx 4 chars = 1 token)"""
    return (len(text) + 3) // 4


CONTEXT_SAFETY_TOKENS = 100
PRICE_TOKEN_UNIT = 1_000_000


def estimate_request_tokens(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Estimate prompt tokens, including tool schemas and tool-call history."""
    parts: List[str] = []
    for message in messages:
        parts.append(str(message.get("role") or ""))
        parts.append(normalize_content(message.get("content")))
        if message.get("tool_calls") is not None:
            parts.append(json.dumps(message["tool_calls"], separators=(",", ":"), default=str))
        if message.get("function_call") is not None:
            parts.append(json.dumps(message["function_call"], separators=(",", ":"), default=str))
        if message.get("tool_call_id") is not None:
            parts.append(str(message["tool_call_id"]))
    if tools:
        parts.append(json.dumps(tools, separators=(",", ":"), default=str))
    return estimate_tokens(" ".join(parts))


def _configured_context_limit(llm: LLMConfig) -> int:
    configured = int(llm.context_limit or 0)
    if configured > 0:
        return configured
    return int(MODEL_CONTEXT_LIMITS.get(llm.model_id, 32768))


def _eligible_models_for_request(
    config: OpenClawConfig,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[str], int]:
    input_tokens = estimate_request_tokens(messages, tools)
    eligible = [
        name
        for name, llm in config.llms.items()
        if input_tokens + CONTEXT_SAFETY_TOKENS < _configured_context_limit(llm)
    ]
    return eligible, input_tokens


def adjust_max_tokens(
    messages: List[Dict[str, Any]],
    llm: LLMConfig,
    requested_max: Optional[int],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """Clamp output tokens to the selected model's configured limits."""
    context_limit = _configured_context_limit(llm)
    input_tokens = estimate_request_tokens(messages, tools)
    available = context_limit - input_tokens - CONTEXT_SAFETY_TOKENS
    if available < 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Input requires approximately {input_tokens} tokens, which exceeds "
                f"the configured context limit for '{llm.name}' ({context_limit})."
            ),
        )

    configured_max = max(1, int(llm.max_tokens or 4096))
    desired_max = configured_max if requested_max is None else int(requested_max)
    if desired_max < 1:
        raise HTTPException(status_code=400, detail="max_tokens must be greater than zero")
    result = min(desired_max, configured_max, available)

    # NVIDIA API limits max_tokens to 1024
    if llm.model_id in MODELS_WITHOUT_SYSTEM_ROLE:
        result = min(result, 1024)

    return result


def calculate_usage_cost(llm: LLMConfig, usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Calculate request cost using configured USD-per-million-token prices."""
    if not isinstance(usage, dict) or not usage:
        return None

    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    input_cost = input_tokens * float(llm.input_price or 0.0) / PRICE_TOKEN_UNIT
    output_cost = output_tokens * float(llm.output_price or 0.0) / PRICE_TOKEN_UNIT
    return {
        "model": llm.name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 12),
        "output_cost_usd": round(output_cost, 12),
        "total_cost_usd": round(input_cost + output_cost, 12),
        "price_unit": "usd_per_1m_tokens",
    }


def _log_usage_cost(llm: LLMConfig, usage: Optional[Dict[str, Any]]) -> None:
    record = calculate_usage_cost(llm, usage)
    if record is not None:
        _safe_log(f"[Usage] {json.dumps(record, separators=(',', ':'), sort_keys=True)}")


def clean_response(result: Dict) -> Dict:
    """Clean response for OpenAI compatibility"""
    usage = _clean_usage(result.get("usage"))

    cleaned = {
        "id": result.get("id", ""),
        "object": result.get("object", "chat.completion"),
        "model": result.get("model", ""),
        "choices": [],
        "usage": usage
    }

    for choice in result.get("choices", []):
        cleaned_choice = {
            "index": choice.get("index", 0),
            "finish_reason": choice.get("finish_reason", "stop")
        }
        if "message" in choice:
            msg = choice["message"]
            cleaned_choice["message"] = {
                "role": msg.get("role", "assistant"),
                "content": msg.get("content")
            }
            if msg.get("tool_calls") is not None:
                cleaned_choice["message"]["tool_calls"] = msg["tool_calls"]
            if msg.get("function_call") is not None:
                cleaned_choice["message"]["function_call"] = msg["function_call"]
        cleaned["choices"].append(cleaned_choice)

    return cleaned


def _message_has_tool_calls(message: Optional[Dict[str, Any]]) -> bool:
    return bool(message and (message.get("tool_calls") or message.get("function_call")))


def _delta_has_tool_calls(delta: Optional[Dict[str, Any]]) -> bool:
    return bool(delta and (delta.get("tool_calls") or delta.get("function_call")))


def _model_attribution_suffix(config: OpenClawConfig, selected_model: str) -> str:
    llm_config = config.llms.get(selected_model)
    model_id = llm_config.model_id if llm_config and llm_config.model_id else selected_model
    return f"\n\n[model: {model_id}]"


def _model_suffix_stream_chunk(
    template: Optional[Dict[str, Any]],
    suffix: str,
) -> str:
    template = template or {}
    source_choices = template.get("choices") or [{}]
    payload = {
        "id": template.get("id", ""),
        "object": template.get("object", "chat.completion.chunk"),
        "choices": [
            {
                "index": source_choices[0].get("index", 0),
                "delta": {"content": suffix},
                "finish_reason": None,
            }
        ],
    }
    for key in ("created", "model", "system_fingerprint"):
        if key in template:
            payload[key] = template[key]
    return f"data: {json.dumps(payload)}\n\n"


def _clean_usage_value(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            cleaned_item = _clean_usage_value(item)
            if cleaned_item is not None:
                cleaned[key] = cleaned_item
        return cleaned
    if isinstance(value, list):
        cleaned = []
        for item in value:
            cleaned_item = _clean_usage_value(item)
            if cleaned_item is not None:
                cleaned.append(cleaned_item)
        return cleaned
    if value is None:
        return None
    return value


def _clean_usage(usage_raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not usage_raw:
        return {}
    if not isinstance(usage_raw, dict):
        return {}
    cleaned_usage = _clean_usage_value(usage_raw)
    return cleaned_usage if isinstance(cleaned_usage, dict) else {}


def _merge_stream_options(stream_options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = dict(stream_options or {})
    merged.setdefault("include_usage", True)
    return merged


def clean_streaming_chunk(chunk: Dict) -> Optional[Dict]:
    """Clean streaming chunk for OpenAI compatibility"""
    choices = chunk.get("choices", [])
    usage = _clean_usage(chunk.get("usage"))
    if not choices and not usage:
        return None

    cleaned = {
        "id": chunk.get("id", ""),
        "object": chunk.get("object", "chat.completion.chunk"),
        "choices": []
    }
    if "model" in chunk:
        cleaned["model"] = chunk["model"]
    if usage:
        cleaned["usage"] = usage

    for choice in choices:
        finish_reason = choice.get("finish_reason")
        cleaned_choice = {
            "index": choice.get("index", 0),
            "finish_reason": finish_reason
        }

        if "delta" in choice:
            delta = choice["delta"]
            if finish_reason == "stop":
                cleaned_choice["delta"] = {}
            else:
                cleaned_delta = {}
                if "role" in delta:
                    cleaned_delta["role"] = delta["role"]
                if "content" in delta:
                    cleaned_delta["content"] = delta["content"]
                if "tool_calls" in delta:
                    cleaned_delta["tool_calls"] = delta["tool_calls"]
                if "function_call" in delta:
                    cleaned_delta["function_call"] = delta["function_call"]
                cleaned_choice["delta"] = cleaned_delta
        else:
            cleaned_choice["delta"] = {}

        cleaned["choices"].append(cleaned_choice)

    return cleaned


LOCAL_PROVIDER_HINTS = {
    "sglang",
    "vllm",
    "llama.cpp",
    "llama_cpp",
    "lmstudio",
    "lm_studio",
    "huggingface_cli",
}


def _is_local_base_url(base_url: str) -> bool:
    if not base_url:
        return False
    lower = base_url.lower()
    return (
        "localhost" in lower
        or "127.0.0.1" in lower
        or lower.startswith("http://0.0.0.0")
    )


def _resolve_auth_mode(provider: str, base_url: str, auth_mode: str = "auto", local: Optional[bool] = None) -> str:
    mode = (auth_mode or "auto").strip().lower()
    if mode in ("none", "bearer"):
        return mode

    provider_norm = (provider or "").strip().lower()
    is_local = bool(local) if local is not None else _is_local_base_url(base_url)
    if provider_norm in LOCAL_PROVIDER_HINTS or is_local:
        return "none"
    return "bearer"


def _build_chat_url(base_url: str, chat_path: str) -> str:
    path = (chat_path or "/chat/completions").strip()
    if not path.startswith("/"):
        path = "/" + path
    return f"{(base_url or '').rstrip('/')}{path}"


def _build_fallback_chain(
    selected_model: str,
    available_models: List[str],
    fallback_models: List[str],
) -> List[str]:
    """Ordered, de-duplicated list of models to try: the routed model first,
    then any configured fallbacks that are actually available."""
    chain = [selected_model]
    for name in fallback_models:
        if name in available_models and name not in chain:
            chain.append(name)
    return chain


async def _prepend_chunk(first_chunk: Optional[str], rest: AsyncGenerator) -> AsyncGenerator:
    """Re-attach a chunk already pulled off `rest` back onto the stream."""
    if first_chunk is not None:
        yield first_chunk
    async for item in rest:
        yield item


# ============================================================
# Responses API compatibility (used by Codex custom providers)
# ============================================================

# chat-safe name -> (Responses tool kind, optional namespace, original name)
ToolNameMap = Dict[str, Tuple[str, Optional[str], str]]


def _new_api_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _chat_safe_tool_name(name: str, used: ToolNameMap) -> str:
    """Return a unique Chat Completions-compatible function name."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", name or "tool")
    if len(cleaned) > 64:
        suffix = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:10]
        cleaned = f"{cleaned[:53]}_{suffix}"

    candidate = cleaned or "tool"
    counter = 2
    while candidate in used:
        suffix = f"_{counter}"
        candidate = f"{cleaned[:64 - len(suffix)]}{suffix}"
        counter += 1
    return candidate


def _function_tool_to_chat(
    tool: Dict[str, Any],
    namespace: Optional[str],
    name_map: ToolNameMap,
) -> Optional[Dict[str, Any]]:
    if tool.get("type") != "function" or not tool.get("name"):
        return None

    original_name = str(tool["name"])
    qualified_name = f"{namespace}__{original_name}" if namespace else original_name
    chat_name = _chat_safe_tool_name(qualified_name, name_map)
    name_map[chat_name] = ("function", namespace, original_name)

    function: Dict[str, Any] = {
        "name": chat_name,
        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
    }
    if tool.get("description") is not None:
        function["description"] = tool["description"]
    if tool.get("strict") is not None:
        function["strict"] = tool["strict"]
    return {"type": "function", "function": function}


def _custom_tool_to_chat(
    tool: Dict[str, Any],
    namespace: Optional[str],
    name_map: ToolNameMap,
) -> Optional[Dict[str, Any]]:
    """Wrap a Responses freeform custom tool in a JSON function argument."""
    if tool.get("type") != "custom" or not tool.get("name"):
        return None

    original_name = str(tool["name"])
    qualified_name = f"{namespace}__{original_name}" if namespace else original_name
    chat_name = _chat_safe_tool_name(qualified_name, name_map)
    name_map[chat_name] = ("custom", namespace, original_name)

    description = str(tool.get("description") or f"Run the {original_name} custom tool.")
    description += " Pass the custom tool's complete raw input in the `input` string."
    return {
        "type": "function",
        "function": {
            "name": chat_name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Complete raw input for the custom tool.",
                    }
                },
                "required": ["input"],
                "additionalProperties": False,
            },
        },
    }


def _tool_search_to_chat(tool: Dict[str, Any], name_map: ToolNameMap) -> Dict[str, Any]:
    """Expose Codex's client-executed tool search to Chat backends."""
    chat_name = _chat_safe_tool_name("tool_search", name_map)
    name_map[chat_name] = ("tool_search", None, "tool_search")
    return {
        "type": "function",
        "function": {
            "name": chat_name,
            "description": tool.get("description") or "Search for additional tools by capability.",
            "parameters": tool.get("parameters")
            or {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


def responses_tools_to_chat(
    tools: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[List[Dict[str, Any]]], ToolNameMap]:
    """Flatten Responses function/namespace tools for Chat Completions backends."""
    if not tools:
        return None, {}

    converted: List[Dict[str, Any]] = []
    name_map: ToolNameMap = {}
    for tool in tools:
        tool_type = tool.get("type")
        if tool_type == "function":
            converted_tool = _function_tool_to_chat(tool, None, name_map)
            if converted_tool:
                converted.append(converted_tool)
        elif tool_type == "custom":
            converted_tool = _custom_tool_to_chat(tool, None, name_map)
            if converted_tool:
                converted.append(converted_tool)
        elif tool_type == "tool_search":
            converted.append(_tool_search_to_chat(tool, name_map))
        elif tool_type == "namespace":
            namespace = str(tool.get("name") or "namespace")
            for child in tool.get("tools") or []:
                if child.get("type") == "custom":
                    converted_tool = _custom_tool_to_chat(child, namespace, name_map)
                else:
                    converted_tool = _function_tool_to_chat(child, namespace, name_map)
                if converted_tool:
                    converted.append(converted_tool)

    return converted or None, name_map


def _responses_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)

    parts: List[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            continue

        part_type = part.get("type")
        if part_type in {"input_text", "output_text", "text", "reasoning_text"}:
            parts.append(str(part.get("text") or ""))
        elif part_type == "refusal":
            parts.append(str(part.get("refusal") or ""))
        elif part_type == "input_image":
            image_ref = part.get("image_url") or part.get("file_id") or "image"
            parts.append(f"[Image input: {image_ref}]")
        elif part_type == "input_file":
            file_ref = part.get("filename") or part.get("file_id") or part.get("file_url") or "file"
            parts.append(f"[File input: {file_ref}]")
    return "\n".join(part for part in parts if part)


def _responses_content_to_chat(content: Any) -> Any:
    """Preserve supported Responses media for the existing media pipeline."""
    if not isinstance(content, list):
        return _responses_content_to_text(content)

    chat_parts: List[Dict[str, Any]] = []
    has_media = False
    for part in content:
        if isinstance(part, str):
            chat_parts.append({"type": "text", "text": part})
            continue
        if not isinstance(part, dict):
            continue

        part_type = str(part.get("type") or "")
        if part_type in {"input_text", "output_text", "text", "reasoning_text"}:
            text = str(part.get("text") or "")
            if text:
                chat_parts.append({"type": "text", "text": text})
        elif part_type == "refusal":
            text = str(part.get("refusal") or "")
            if text:
                chat_parts.append({"type": "text", "text": text})
        elif part_type == "input_image":
            has_media = True
            image_url = part.get("image_url")
            image_ref = part.get("file_id") or "image"
            if image_url and not str(image_url).startswith("data:"):
                image_ref = image_url
            chat_parts.append({"type": "text", "text": f"[Image input: {image_ref}]"})
            if image_url:
                image_value: Dict[str, Any] = {"url": str(image_url)}
                if part.get("detail") is not None:
                    image_value["detail"] = part["detail"]
                chat_parts.append({"type": "image_url", "image_url": image_value})
        elif part_type == "input_audio":
            has_media = True
            audio = part.get("input_audio") if isinstance(part.get("input_audio"), dict) else part
            data = audio.get("data") if isinstance(audio, dict) else None
            chat_parts.append({"type": "text", "text": "[Audio input]"})
            if data:
                chat_parts.append(
                    {
                        "type": "input_audio",
                        "data": data,
                        "mime_type": audio.get("mime_type") or audio.get("format") or "audio/mp3",
                    }
                )
        elif part_type == "input_file":
            file_ref = (
                part.get("filename")
                or part.get("file_id")
                or part.get("file_url")
                or "file"
            )
            chat_parts.append({"type": "text", "text": f"[File input: {file_ref}]"})

    if not has_media:
        return "\n".join(
            str(part.get("text") or "")
            for part in chat_parts
            if part.get("type") == "text" and part.get("text")
        )
    return chat_parts


def _mapped_chat_tool_name(
    name: str,
    namespace: Optional[str],
    name_map: ToolNameMap,
    kind: str = "function",
) -> str:
    for chat_name, mapped in name_map.items():
        if mapped == (kind, namespace, name):
            return chat_name
    # This can happen when a client resends history but omits the old tool list.
    qualified = f"{namespace}__{name}" if namespace else name
    return re.sub(r"[^a-zA-Z0-9_-]", "_", qualified)[:64] or "tool"


def _responses_tool_choice_to_chat(choice: Any, name_map: ToolNameMap) -> Any:
    if not isinstance(choice, dict):
        return choice
    choice_type = str(choice.get("type") or "")
    if choice_type not in {"function", "custom"} or not choice.get("name"):
        # Chat Completions has no equivalent for namespace/allowed-tools choices.
        return "auto"
    name = _mapped_chat_tool_name(
        str(choice["name"]),
        str(choice["namespace"]) if choice.get("namespace") else None,
        name_map,
        choice_type,
    )
    return {"type": "function", "function": {"name": name}}


def _append_chat_tool_call(messages: List[Dict[str, Any]], tool_call: Dict[str, Any]) -> None:
    if messages and messages[-1].get("role") == "assistant" and messages[-1].get("tool_calls"):
        messages[-1]["tool_calls"].append(tool_call)
    else:
        messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})


def _tool_output_to_text(output: Any) -> str:
    if isinstance(output, (dict, list)):
        if isinstance(output, list) and all(isinstance(part, (dict, str)) for part in output):
            text = _responses_content_to_text(output)
            if text:
                return text
        return json.dumps(output, separators=(",", ":"))
    return _responses_content_to_text(output)


def responses_request_to_chat(
    request: ResponsesRequest,
) -> Tuple[ChatRequest, ToolNameMap]:
    """Translate a Responses request into the existing Chat Completions path."""
    chat_tools, name_map = responses_tools_to_chat(request.tools)
    messages: List[Dict[str, Any]] = []
    system_parts: List[str] = []
    if request.instructions:
        system_parts.append(request.instructions)

    input_items = request.input
    if isinstance(input_items, str):
        input_items = [{"type": "message", "role": "user", "content": input_items}]
    elif isinstance(input_items, dict):
        input_items = [input_items]
    elif not isinstance(input_items, list):
        input_items = []

    for item in input_items:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type", "message")
        if item_type == "message":
            role = str(item.get("role") or "user")
            if role in {"system", "developer"}:
                content = _responses_content_to_text(item.get("content"))
                if content:
                    system_parts.append(content)
            else:
                content = _responses_content_to_chat(item.get("content"))
                messages.append({"role": role, "content": content})
        elif item_type in {"function_call", "custom_tool_call", "tool_search_call"}:
            kind = {
                "function_call": "function",
                "custom_tool_call": "custom",
                "tool_search_call": "tool_search",
            }[item_type]
            name = str(item.get("name") or "tool")
            if kind == "tool_search":
                name = "tool_search"
            namespace = str(item["namespace"]) if item.get("namespace") else None
            if kind == "custom":
                arguments = json.dumps({"input": str(item.get("input") or "")})
            elif kind == "tool_search":
                raw_arguments = item.get("arguments") or {}
                arguments = (
                    raw_arguments
                    if isinstance(raw_arguments, str)
                    else json.dumps(raw_arguments, separators=(",", ":"))
                )
            else:
                arguments = str(item.get("arguments") or "{}")
            tool_call = {
                "id": str(item.get("call_id") or item.get("id") or _new_api_id("call")),
                "type": "function",
                "function": {
                    "name": _mapped_chat_tool_name(name, namespace, name_map, kind),
                    "arguments": arguments,
                },
            }
            _append_chat_tool_call(messages, tool_call)
        elif item_type in {"function_call_output", "custom_tool_call_output", "tool_search_output"}:
            output = (
                item.get("tools", item.get("output"))
                if item_type == "tool_search_output"
                else item.get("output")
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or item.get("id") or ""),
                    "content": _tool_output_to_text(output),
                }
            )

    if system_parts:
        messages.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    if not messages:
        messages.append({"role": "user", "content": ""})

    return (
        ChatRequest(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_output_tokens if request.max_output_tokens is not None else 4096,
            stream=request.stream,
            user=request.user or request.safety_identifier,
            tools=chat_tools,
            tool_choice=_responses_tool_choice_to_chat(request.tool_choice, name_map),
            parallel_tool_calls=request.parallel_tool_calls,
        ),
        name_map,
    )


def _responses_usage(chat_usage: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not chat_usage:
        return None
    input_details = chat_usage.get("prompt_tokens_details") or {}
    output_details = chat_usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": int(chat_usage.get("prompt_tokens") or 0),
        "input_tokens_details": {
            "cached_tokens": int(input_details.get("cached_tokens") or chat_usage.get("cache_read_input_tokens") or 0),
            "cache_write_tokens": int(chat_usage.get("cache_creation_input_tokens") or 0),
        },
        "output_tokens": int(chat_usage.get("completion_tokens") or 0),
        "output_tokens_details": {
            "reasoning_tokens": int(output_details.get("reasoning_tokens") or 0),
        },
        "total_tokens": int(chat_usage.get("total_tokens") or 0),
    }


def _response_object(
    request: ResponsesRequest,
    response_id: str,
    created_at: int,
    model: str,
    status: str,
    output: List[Dict[str, Any]],
    usage: Optional[Dict[str, Any]],
    *,
    error: Optional[Dict[str, Any]] = None,
    incomplete_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "error": error,
        "incomplete_details": incomplete_details,
        "instructions": None,
        "max_output_tokens": request.max_output_tokens,
        "model": model,
        "output": output,
        "parallel_tool_calls": request.parallel_tool_calls if request.parallel_tool_calls is not None else True,
        "previous_response_id": request.previous_response_id,
        "reasoning": request.reasoning or {"effort": None, "summary": None},
        "store": bool(request.store),
        "temperature": request.temperature,
        "text": request.text or {"format": {"type": "text"}},
        "tool_choice": request.tool_choice or "auto",
        "tools": [],
        "top_p": request.top_p,
        "truncation": request.truncation or "disabled",
        "usage": usage,
        "user": request.user,
        "metadata": request.metadata or {},
    }


def _decode_tool_name(chat_name: str, name_map: ToolNameMap) -> Tuple[str, Optional[str], str]:
    return name_map.get(chat_name, ("function", None, chat_name))


def _custom_input_from_chat_arguments(arguments: str) -> str:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return arguments
    if isinstance(parsed, dict) and "input" in parsed:
        value = parsed["input"]
        return value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
    return arguments


def _tool_search_arguments(arguments: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        return {"query": arguments}
    return parsed if isinstance(parsed, dict) else {"query": str(parsed)}


def _chat_tool_call_to_response_item(
    tool_call: Dict[str, Any],
    name_map: ToolNameMap,
) -> Dict[str, Any]:
    function = tool_call.get("function") or {}
    kind, namespace, name = _decode_tool_name(str(function.get("name") or "tool"), name_map)
    call_id = str(tool_call.get("id") or _new_api_id("call"))
    arguments = str(function.get("arguments") or "{}")
    if kind == "custom":
        item = {
            "id": _new_api_id("ctc"),
            "type": "custom_tool_call",
            "status": "completed",
            "call_id": call_id,
            "name": name,
            "input": _custom_input_from_chat_arguments(arguments),
        }
    elif kind == "tool_search":
        item = {
            "id": _new_api_id("tsc"),
            "type": "tool_search_call",
            "status": "completed",
            "call_id": call_id,
            "execution": "client",
            "arguments": _tool_search_arguments(arguments),
        }
    else:
        item = {
            "id": _new_api_id("fc"),
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }
    if namespace:
        item["namespace"] = namespace
    return item


def chat_response_to_responses(
    result: Dict[str, Any],
    request: ResponsesRequest,
    name_map: ToolNameMap,
    *,
    strip_model_prefix: bool = False,
) -> Dict[str, Any]:
    choice = (result.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    output: List[Dict[str, Any]] = []
    content = message.get("content")
    if content:
        text = str(content)
        if strip_model_prefix:
            text = re.sub(r"^\[[\w\-.]+\]\s*", "", text)
        output.append(
            {
                "id": _new_api_id("msg"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": text, "annotations": [], "logprobs": []}
                ],
            }
        )
    for tool_call in message.get("tool_calls") or []:
        output.append(_chat_tool_call_to_response_item(tool_call, name_map))

    finish_reason = choice.get("finish_reason")
    incomplete = finish_reason in {"length", "content_filter"}
    status = "incomplete" if incomplete else "completed"
    return _response_object(
        request,
        _new_api_id("resp"),
        int(time.time()),
        str(result.get("model") or request.model),
        status,
        output,
        _responses_usage(result.get("usage")),
        incomplete_details={"reason": "max_output_tokens"} if finish_reason == "length" else None,
    )


def _responses_sse(event: Dict[str, Any]) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"


async def _chat_sse_payloads(body_iterator: AsyncIterator[Any]) -> AsyncGenerator:
    buffer = ""
    async for raw_chunk in body_iterator:
        if isinstance(raw_chunk, bytes):
            raw_chunk = raw_chunk.decode("utf-8", errors="replace")
        buffer += str(raw_chunk)
        while "\n\n" in buffer:
            block, buffer = buffer.split("\n\n", 1)
            data_lines = [line[5:].lstrip() for line in block.splitlines() if line.startswith("data:")]
            if not data_lines:
                continue
            data = "\n".join(data_lines)
            if data == "[DONE]":
                yield None
            else:
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    continue

    if buffer.strip():
        data_lines = [line[5:].lstrip() for line in buffer.splitlines() if line.startswith("data:")]
        if data_lines:
            data = "\n".join(data_lines)
            if data == "[DONE]":
                yield None
            else:
                try:
                    yield json.loads(data)
                except json.JSONDecodeError:
                    return


async def chat_stream_to_responses(
    body_iterator: AsyncIterator[Any],
    request: ResponsesRequest,
    name_map: ToolNameMap,
    *,
    strip_model_prefix: bool = False,
) -> AsyncGenerator[str, None]:
    """Translate Chat Completions SSE chunks into Responses API SSE events."""
    response_id = _new_api_id("resp")
    created_at = int(time.time())
    model = request.model
    sequence = 0
    next_output_index = 0
    finish_reason: Optional[str] = None
    chat_usage: Optional[Dict[str, Any]] = None
    final_output: List[Tuple[int, Dict[str, Any]]] = []

    text_state: Optional[Dict[str, Any]] = None
    tool_states: Dict[int, Dict[str, Any]] = {}
    first_text_delta = True

    def event(event_type: str, **fields: Any) -> Dict[str, Any]:
        nonlocal sequence
        value = {"type": event_type, **fields, "sequence_number": sequence}
        sequence += 1
        return value

    def start_text_item() -> List[Dict[str, Any]]:
        nonlocal text_state, next_output_index
        if text_state is not None:
            return []
        text_state = {
            "id": _new_api_id("msg"),
            "output_index": next_output_index,
            "text": "",
            "done": False,
        }
        next_output_index += 1
        item = {
            "id": text_state["id"],
            "type": "message",
            "status": "in_progress",
            "role": "assistant",
            "content": [],
        }
        part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
        return [
            event(
                "response.output_item.added",
                output_index=text_state["output_index"],
                item=item,
            ),
            event(
                "response.content_part.added",
                item_id=text_state["id"],
                output_index=text_state["output_index"],
                content_index=0,
                part=part,
            ),
        ]

    def finish_text_item() -> List[Dict[str, Any]]:
        if text_state is None or text_state["done"]:
            return []
        text_state["done"] = True
        part = {
            "type": "output_text",
            "text": text_state["text"],
            "annotations": [],
            "logprobs": [],
        }
        item = {
            "id": text_state["id"],
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [part],
        }
        final_output.append((text_state["output_index"], item))
        return [
            event(
                "response.output_text.done",
                item_id=text_state["id"],
                output_index=text_state["output_index"],
                content_index=0,
                text=text_state["text"],
                logprobs=[],
            ),
            event(
                "response.content_part.done",
                item_id=text_state["id"],
                output_index=text_state["output_index"],
                content_index=0,
                part=part,
            ),
            event(
                "response.output_item.done",
                output_index=text_state["output_index"],
                item=item,
            ),
        ]

    def start_tool_item(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        nonlocal next_output_index
        if state.get("started"):
            return []
        kind, namespace, name = _decode_tool_name(state.get("chat_name") or "tool", name_map)
        state.update(
            {
                "started": True,
                "kind": kind,
                "namespace": namespace,
                "name": name,
                "item_id": _new_api_id(
                    "ctc" if kind == "custom" else "tsc" if kind == "tool_search" else "fc"
                ),
                "output_index": next_output_index,
            }
        )
        next_output_index += 1
        if kind == "custom":
            item: Dict[str, Any] = {
                "id": state["item_id"],
                "type": "custom_tool_call",
                "status": "in_progress",
                "call_id": state["call_id"],
                "name": state["name"],
                "input": "",
            }
        elif kind == "tool_search":
            item = {
                "id": state["item_id"],
                "type": "tool_search_call",
                "status": "in_progress",
                "call_id": state["call_id"],
                "execution": "client",
                "arguments": {},
            }
        else:
            item = {
                "id": state["item_id"],
                "type": "function_call",
                "status": "in_progress",
                "call_id": state["call_id"],
                "name": state["name"],
                "arguments": "",
            }
        if state["namespace"]:
            item["namespace"] = state["namespace"]
        return [
            event(
                "response.output_item.added",
                output_index=state["output_index"],
                item=item,
            )
        ]

    def finish_tool_item(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not state.get("started") or state.get("done"):
            return []
        state["done"] = True
        kind = state.get("kind", "function")
        if kind == "custom":
            custom_input = _custom_input_from_chat_arguments(state["arguments"])
            item: Dict[str, Any] = {
                "id": state["item_id"],
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": state["call_id"],
                "name": state["name"],
                "input": custom_input,
            }
            done_events = []
            if custom_input:
                done_events.append(
                    event(
                        "response.custom_tool_call_input.delta",
                        item_id=state["item_id"],
                        output_index=state["output_index"],
                        delta=custom_input,
                    )
                )
            done_events.append(
                event(
                    "response.custom_tool_call_input.done",
                    item_id=state["item_id"],
                    output_index=state["output_index"],
                    input=custom_input,
                )
            )
        elif kind == "tool_search":
            item = {
                "id": state["item_id"],
                "type": "tool_search_call",
                "status": "completed",
                "call_id": state["call_id"],
                "execution": "client",
                "arguments": _tool_search_arguments(state["arguments"]),
            }
            done_events = []
        else:
            item = {
                "id": state["item_id"],
                "type": "function_call",
                "status": "completed",
                "call_id": state["call_id"],
                "name": state["name"],
                "arguments": state["arguments"],
            }
            done_events = [
                event(
                    "response.function_call_arguments.done",
                    item_id=state["item_id"],
                    output_index=state["output_index"],
                    arguments=state["arguments"],
                )
            ]
        if state.get("namespace"):
            item["namespace"] = state["namespace"]
        final_output.append((state["output_index"], item))
        return done_events + [
            event(
                "response.output_item.done",
                output_index=state["output_index"],
                item=item,
            ),
        ]

    initial_response = _response_object(
        request,
        response_id,
        created_at,
        model,
        "in_progress",
        [],
        None,
    )
    yield _responses_sse(event("response.created", response=initial_response))
    yield _responses_sse(event("response.in_progress", response=initial_response))

    async for payload in _chat_sse_payloads(body_iterator):
        if payload is None:
            # Drain the wrapped Chat stream so its HTTP context and async
            # generator close cleanly before the Responses stream completes.
            continue
        if payload.get("error"):
            error_value = payload["error"]
            error_message = error_value.get("message") if isinstance(error_value, dict) else str(error_value)
            failed_response = _response_object(
                request,
                response_id,
                created_at,
                model,
                "failed",
                [item for _, item in sorted(final_output)],
                _responses_usage(chat_usage),
                error={"code": "upstream_error", "message": error_message or "Upstream model failed"},
            )
            yield _responses_sse(event("response.failed", response=failed_response))
            return

        if payload.get("model"):
            model = str(payload["model"])
        if payload.get("usage"):
            chat_usage = payload["usage"]

        for choice in payload.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if content is not None:
                content = str(content)
                if first_text_delta and strip_model_prefix:
                    content = re.sub(r"^\[[\w\-.]+\]\s*", "", content)
                if content:
                    first_text_delta = False
                    for item_event in start_text_item():
                        yield _responses_sse(item_event)
                    assert text_state is not None
                    text_state["text"] += content
                    yield _responses_sse(
                        event(
                            "response.output_text.delta",
                            item_id=text_state["id"],
                            output_index=text_state["output_index"],
                            content_index=0,
                            delta=content,
                            logprobs=[],
                        )
                    )

            streamed_tool_calls = list(delta.get("tool_calls") or [])
            if delta.get("function_call"):
                streamed_tool_calls.append(
                    {
                        "index": 0,
                        "id": delta.get("id"),
                        "type": "function",
                        "function": delta["function_call"],
                    }
                )

            for tool_call in streamed_tool_calls:
                tool_index = int(tool_call.get("index") or 0)
                state = tool_states.setdefault(
                    tool_index,
                    {
                        "call_id": str(tool_call.get("id") or _new_api_id("call")),
                        "chat_name": "",
                        "arguments": "",
                        "started": False,
                        "done": False,
                    },
                )
                if tool_call.get("id"):
                    state["call_id"] = str(tool_call["id"])
                function = tool_call.get("function") or {}
                name_delta = function.get("name")
                if name_delta:
                    state["chat_name"] += str(name_delta)
                arguments_delta = str(function.get("arguments") or "")
                had_started = bool(state.get("started"))
                state["arguments"] += arguments_delta

                # Function names may themselves be split across Chat SSE chunks.
                # The first argument fragment marks the point at which providers
                # have finished emitting the name; calls with no arguments start
                # in the finalization loop below.
                if state["chat_name"] and arguments_delta and not state["started"]:
                    for item_event in start_tool_item(state):
                        yield _responses_sse(item_event)
                if state["started"] and state.get("kind") == "function":
                    emitted_arguments = arguments_delta if had_started else state["arguments"]
                    if emitted_arguments:
                        yield _responses_sse(
                            event(
                                "response.function_call_arguments.delta",
                                item_id=state["item_id"],
                                output_index=state["output_index"],
                                delta=emitted_arguments,
                            )
                        )

            if choice.get("finish_reason"):
                finish_reason = str(choice["finish_reason"])

    for item_event in finish_text_item():
        yield _responses_sse(item_event)
    for _, state in sorted(tool_states.items()):
        if state.get("chat_name") and not state.get("started"):
            for item_event in start_tool_item(state):
                yield _responses_sse(item_event)
        for item_event in finish_tool_item(state):
            yield _responses_sse(item_event)

    output = [item for _, item in sorted(final_output, key=lambda pair: pair[0])]
    is_incomplete = finish_reason in {"length", "content_filter"}
    status = "incomplete" if is_incomplete else "completed"
    completed_response = _response_object(
        request,
        response_id,
        created_at,
        model,
        status,
        output,
        _responses_usage(chat_usage),
        incomplete_details={"reason": "max_output_tokens"} if finish_reason == "length" else None,
    )
    event_type = "response.incomplete" if is_incomplete else "response.completed"
    yield _responses_sse(event(event_type, response=completed_response))


# ============================================================
# LLM Backend
# ============================================================

class LLMBackend:
    """LLM API caller"""

    def __init__(self, config: OpenClawConfig):
        self.config = config

    async def call(self, llm_name: str, messages: List[Dict], max_tokens: int = 4096,
                   temperature: Optional[float] = None, stream: bool = False,
                   tools: Optional[List[Dict[str, Any]]] = None,
                   tool_choice: Optional[Any] = None,
                   stream_options: Optional[Dict[str, Any]] = None,
                   parallel_tool_calls: Optional[bool] = None,
                   top_p: Optional[float] = None):
        """Call LLM API"""
        if llm_name not in self.config.llms:
            raise HTTPException(status_code=404, detail=f"LLM '{llm_name}' not found")

        llm_config = self.config.llms[llm_name]
        api_key = self.config.get_api_key(llm_config.provider, llm_config)

        if stream:
            return self._call_streaming(
                llm_config,
                messages,
                max_tokens,
                temperature,
                api_key,
                tools,
                tool_choice,
                stream_options,
                parallel_tool_calls,
                top_p,
            )
        else:
            return await self._call_sync(
                llm_config,
                messages,
                max_tokens,
                temperature,
                api_key,
                tools,
                tool_choice,
                parallel_tool_calls,
                top_p,
            )

    async def _call_sync(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                         temperature: Optional[float], api_key: Optional[str],
                         tools: Optional[List[Dict[str, Any]]] = None,
                         tool_choice: Optional[Any] = None,
                         parallel_tool_calls: Optional[bool] = None,
                         top_p: Optional[float] = None) -> Dict:
        """Synchronous API call"""
        normalized = normalize_messages(messages, llm.model_id)
        adjusted_max = adjust_max_tokens(normalized, llm, max_tokens, tools)
        auth_mode = _resolve_auth_mode(llm.provider, llm.base_url, llm.auth_mode, llm.local)
        chat_url = _build_chat_url(llm.base_url, llm.chat_path)


        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if auth_mode == "bearer" and api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            body = {
                "model": llm.model_id,
                "messages": normalized,
                "max_tokens": adjusted_max,
            }
            if temperature is not None:
                body["temperature"] = temperature
            if top_p is not None:
                body["top_p"] = top_p
            if tools is not None:
                body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
            if parallel_tool_calls is not None:
                body["parallel_tool_calls"] = parallel_tool_calls

            resp = await client.post(
                chat_url,
                headers=headers,
                json=body,
                timeout=120.0
            )

            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])

            result = clean_response(resp.json())
            _log_usage_cost(llm, result.get("usage"))
            return result

    async def _call_streaming(self, llm: LLMConfig, messages: List[Dict], max_tokens: int,
                          temperature: Optional[float], api_key: Optional[str],
                          tools: Optional[List[Dict[str, Any]]] = None,
                          tool_choice: Optional[Any] = None,
                          stream_options: Optional[Dict[str, Any]] = None,
                          parallel_tool_calls: Optional[bool] = None,
                          top_p: Optional[float] = None) -> AsyncGenerator:
        """Streaming API call"""
        normalized = normalize_messages(messages, llm.model_id)
        adjusted_max = adjust_max_tokens(normalized, llm, max_tokens, tools)
        auth_mode = _resolve_auth_mode(llm.provider, llm.base_url, llm.auth_mode, llm.local)
        chat_url = _build_chat_url(llm.base_url, llm.chat_path)

        async with httpx.AsyncClient() as client:
            headers = {"Content-Type": "application/json"}
            if auth_mode == "bearer" and api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            body = {
                "model": llm.model_id,
                "messages": normalized,
                "max_tokens": adjusted_max,
                "stream": True,
                "stream_options": _merge_stream_options(stream_options),
            }
            if temperature is not None:
                body["temperature"] = temperature
            if top_p is not None:
                body["top_p"] = top_p
            if tools is not None:
                body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
            if parallel_tool_calls is not None:
                body["parallel_tool_calls"] = parallel_tool_calls

            async with client.stream(
                "POST",
                chat_url,
                headers=headers,
                json=body,
                timeout=120.0
            ) as resp:
                if resp.status_code != 200:
                    error = await resp.aread()
                    raise HTTPException(status_code=resp.status_code, detail=error.decode()[:200])

                latest_usage: Optional[Dict[str, Any]] = None
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            _log_usage_cost(llm, latest_usage)
                        elif payload:
                            try:
                                event = json.loads(payload)
                                usage = event.get("usage")
                                if isinstance(usage, dict) and usage:
                                    latest_usage = usage
                            except json.JSONDecodeError:
                                pass
                        yield line + "\n\n"


# ============================================================
# FastAPI App Factory
# ============================================================

def create_app(config: OpenClawConfig = None, config_path: str = None) -> FastAPI:
    """Create FastAPI application"""
    if config is None and config_path:
        config = OpenClawConfig.from_yaml(config_path)
    elif config is None:
        config = OpenClawConfig()

    app = FastAPI(
        title="OpenClaw Router",
        description="OpenAI-compatible API with intelligent LLM routing",
        version="1.0.0"
    )

    # Initialize components
    router = OpenClawRouter(config)
    backend = LLMBackend(config)
    # Codex can make several model requests for one user turn while executing
    # tools. Keep the selected backend stable for that loop, but allow the next
    # distinct user query in the same thread to be routed again.
    codex_route_cache: OrderedDict[Tuple[str, str, str, str], str] = OrderedDict()
    codex_route_cache_limit = 1024

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "strategy": config.router.strategy,
            "llms": list(config.llms.keys())
        }

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {"id": name, "object": "model", "description": llm.description}
                for name, llm in config.llms.items()
            ] + [{"id": "auto", "object": "model", "description": "Auto router"}]
        }

    async def _serve_chat_completions(
        request: ChatRequest,
        codex_cache_context: Optional[Tuple[str, str, str]] = None,
    ):
        print(f"============\n")
        messages = []
        for message in request.messages:
            message_payload = {
                "role": message.role,
                "content": message.content,
            }
            if message.tool_calls is not None:
                message_payload["tool_calls"] = message.tool_calls
            if message.tool_call_id is not None:
                message_payload["tool_call_id"] = message.tool_call_id
            if message.function_call is not None:
                message_payload["function_call"] = message.function_call
            messages.append(message_payload)

        # Extract user query for routing (with optional media understanding)
        user_query = ""
        media_description = None

        # Find and process the last user message
        last_user_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                last_user_idx = i
                break

        if last_user_idx is not None:
            raw_content = messages[last_user_idx]["content"]

            # Process multimodal content if media is enabled
            # Supports both OpenAI format (list) and OpenClaw format (string with [media attached:...])
            if config.media.enabled:
                # Use together API key as fallback
                together_key = config.api_keys.get("together")
                processed_text, media_desc = await process_multimodal_content(
                    raw_content, config.media, fallback_key=together_key
                )
                user_query = processed_text[:500]
                media_description = media_desc
                if media_desc:
                    print(f"[Media] Processed: {media_desc[:80]}...")
                    # IMPORTANT: Replace the message content with processed text
                    # so LLM sees the image description instead of [media attached: ...]
                    messages[last_user_idx]["content"] = processed_text
            else:
                user_query = normalize_content(raw_content)[:500]

        if not user_query:
            user_query = "general query"

        # Constrain routing and fallback to models that can fit this request.
        available_models = list(config.llms.keys())
        eligible_models, estimated_input_tokens = _eligible_models_for_request(
            config,
            messages,
            request.tools,
        )
        if not eligible_models:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Input requires approximately {estimated_input_tokens} tokens; "
                    "no configured model has enough context."
                ),
            )

        if request.model == "auto" or request.model not in available_models:
            selected_model = None
            cache_key = None
            if codex_cache_context is not None:
                identity, prompt_cache_key, advertised_model = codex_cache_context
                query_digest = hashlib.sha256(user_query.encode("utf-8")).hexdigest()
                cache_key = (identity, advertised_model, prompt_cache_key, query_digest)
                cached_model = codex_route_cache.get(cache_key)
                if cached_model in eligible_models:
                    selected_model = cached_model
                    codex_route_cache.move_to_end(cache_key)
                    print(f"[Router] Reusing Codex turn route -> {selected_model}")

            if selected_model is None:
                selected_model = await router.select_model(
                    user_query,
                    user=request.user,
                    candidate_models=eligible_models,
                )
                if cache_key is not None:
                    codex_route_cache[cache_key] = selected_model
                    codex_route_cache.move_to_end(cache_key)
                    while len(codex_route_cache) > codex_route_cache_limit:
                        codex_route_cache.popitem(last=False)
                    print(f"[Router] Codex query: '{user_query}' -> {selected_model}")
                else:
                    print(f"[Router] Query: '{user_query}' -> {selected_model}")
        else:
            if request.model not in eligible_models:
                context_limit = _configured_context_limit(config.llms[request.model])
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Input requires approximately {estimated_input_tokens} tokens, "
                        f"which exceeds the configured context limit for "
                        f"'{request.model}' ({context_limit})."
                    ),
                )
            selected_model = request.model
            print(f"[Specified] Query: '{user_query}' -> {selected_model}")

        fallback_chain = _build_fallback_chain(
            selected_model, eligible_models, config.router.fallback_models
        )

        # Handle streaming
        if request.stream:
            async def generate():
                nonlocal selected_model
                prefix_sent = False
                content_buffer = ""
                buffered_chunks = []
                suffix_sent = False
                stream_saw_text = False
                stream_saw_tool_calls = False
                suffix_template: Optional[Dict[str, Any]] = None

                def flush_buffered_prefix() -> Optional[str]:
                    nonlocal prefix_sent, content_buffer, buffered_chunks
                    if not buffered_chunks or prefix_sent:
                        return None

                    content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                    first = buffered_chunks[0]
                    try:
                        first_json = first[6:] if first.startswith("data: ") else first
                        first_data = json.loads(first_json.strip())
                        if first_data.get("choices") and first_data["choices"][0].get("delta"):
                            first_data["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                            prefix_sent = True
                            buffered_chunks = []
                            return f"data: {json.dumps(first_data)}\n\n"
                    except:
                        pass
                    return None

                try:
                    prefix_disabled = False

                    stream_gen = None
                    first_chunk = None
                    last_error = None
                    for i, candidate in enumerate(fallback_chain):
                        try:
                            candidate_gen = await backend.call(
                                candidate, messages, request.max_tokens,
                                request.temperature, stream=True,
                                tools=request.tools,
                                tool_choice=request.tool_choice,
                                stream_options=request.stream_options,
                                parallel_tool_calls=request.parallel_tool_calls,
                                top_p=request.top_p,
                            )
                            # Probe the connection before committing to this
                            # candidate, so a failed attempt never reaches the
                            # client and we can still fall back.
                            try:
                                first_chunk = await candidate_gen.__anext__()
                            except StopAsyncIteration:
                                first_chunk = None
                            stream_gen = candidate_gen
                            selected_model = candidate
                            if i > 0:
                                print(f"[Fallback] '{fallback_chain[0]}' failed; streaming from '{candidate}' instead.")
                            break
                        except Exception as error:
                            last_error = error
                            print(f"[Fallback] Model '{candidate}' failed: {error}")

                    if stream_gen is None:
                        yield f'data: {json.dumps({"error": str(last_error)})}\n\n'
                        yield "data: [DONE]\n\n"
                        return

                    combined_stream = _prepend_chunk(first_chunk, stream_gen)
                    async for chunk in combined_stream:
                        if config.show_model_suffix:
                            if "[DONE]" in chunk:
                                if stream_saw_text and not stream_saw_tool_calls and not suffix_sent:
                                    yield _model_suffix_stream_chunk(
                                        suffix_template,
                                        _model_attribution_suffix(config, selected_model),
                                    )
                                    suffix_sent = True
                                yield chunk
                                continue

                            try:
                                json_str = chunk[6:] if chunk.startswith("data: ") else chunk
                                data = json.loads(json_str.strip())
                            except Exception:
                                yield chunk
                                continue

                            choices = data.get("choices") or []
                            if choices:
                                suffix_template = data
                            for choice in choices:
                                delta = choice.get("delta") or {}
                                if delta.get("content"):
                                    stream_saw_text = True
                                if _delta_has_tool_calls(delta) or choice.get("finish_reason") in {
                                    "tool_calls",
                                    "function_call",
                                }:
                                    stream_saw_tool_calls = True

                            terminal_chunk = any(choice.get("finish_reason") for choice in choices)
                            if (
                                terminal_chunk
                                and stream_saw_text
                                and not stream_saw_tool_calls
                                and not suffix_sent
                            ):
                                yield _model_suffix_stream_chunk(
                                    suffix_template,
                                    _model_attribution_suffix(config, selected_model),
                                )
                                suffix_sent = True

                            yield chunk
                            continue

                        if not config.show_model_prefix:
                            yield chunk
                            continue

                        if prefix_disabled:
                            if "[DONE]" in chunk:
                                yield chunk
                                continue
                            try:
                                json_str = chunk[6:] if chunk.startswith("data: ") else chunk
                                data = json.loads(json_str.strip())
                                cleaned = clean_streaming_chunk(data)
                                if cleaned:
                                    yield f"data: {json.dumps(cleaned)}\n\n"
                                    continue
                            except:
                                pass
                            yield chunk
                            continue

                        # Add model prefix to first content chunk
                        if "[DONE]" in chunk:
                            # Flush buffer before DONE
                            if buffered_chunks and not prefix_sent:
                                flushed_chunk = flush_buffered_prefix()
                                if flushed_chunk:
                                    yield flushed_chunk
                            yield chunk
                        else:
                            try:
                                json_str = chunk[6:] if chunk.startswith("data: ") else chunk
                                data = json.loads(json_str.strip())
                                cleaned = clean_streaming_chunk(data)

                                if cleaned:
                                    if cleaned.get("usage") and not cleaned.get("choices"):
                                        if buffered_chunks and not prefix_sent:
                                            flushed_chunk = flush_buffered_prefix()
                                            if flushed_chunk:
                                                yield flushed_chunk
                                        yield f"data: {json.dumps(cleaned)}\n\n"
                                        continue

                                    choices = cleaned.get("choices", [])
                                    if choices and "delta" in choices[0]:
                                        delta = choices[0]["delta"]

                                        if _delta_has_tool_calls(delta):
                                            if buffered_chunks and not prefix_sent:
                                                for buffered_chunk in buffered_chunks:
                                                    try:
                                                        buffered_json = buffered_chunk[6:] if buffered_chunk.startswith("data: ") else buffered_chunk
                                                        buffered_data = json.loads(buffered_json.strip())
                                                        buffered_cleaned = clean_streaming_chunk(buffered_data)
                                                        if buffered_cleaned:
                                                            yield f"data: {json.dumps(buffered_cleaned)}\n\n"
                                                        else:
                                                            yield buffered_chunk
                                                    except:
                                                        yield buffered_chunk
                                                buffered_chunks = []
                                            prefix_disabled = True
                                            yield f"data: {json.dumps(cleaned)}\n\n"
                                            continue

                                        content = delta.get("content", "")

                                        if not prefix_sent:
                                            content_buffer += content
                                            buffered_chunks.append(chunk)

                                            if len(content_buffer) > 30 or (content_buffer and not content_buffer.startswith("[")):
                                                content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                                                first = buffered_chunks[0]
                                                first_data = json.loads(first[6:] if first.startswith("data: ") else first)
                                                if first_data.get("choices") and first_data["choices"][0].get("delta"):
                                                    first_data["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                                                    yield f"data: {json.dumps(first_data)}\n\n"
                                                    prefix_sent = True
                                                    buffered_chunks = []
                                        else:
                                            yield f"data: {json.dumps(cleaned)}\n\n"
                                    else:
                                        if prefix_sent:
                                            yield f"data: {json.dumps(cleaned)}\n\n"
                            except:
                                yield chunk
                except Exception as e:
                    print(f"[Stream Error] {type(e).__name__}: {e}")
                    yield f'data: {json.dumps({"error": str(e)})}\n\n'

            return StreamingResponse(generate(), media_type="text/event-stream")

        else:
            result = None
            last_error = None
            for i, candidate in enumerate(fallback_chain):
                try:
                    result = await backend.call(
                        candidate, messages, request.max_tokens,
                        request.temperature, stream=False,
                        tools=request.tools,
                        tool_choice=request.tool_choice,
                        parallel_tool_calls=request.parallel_tool_calls,
                        top_p=request.top_p,
                    )
                    selected_model = candidate
                    if i > 0:
                        print(f"[Fallback] '{fallback_chain[0]}' failed; served by '{candidate}' instead.")
                    break
                except Exception as error:
                    last_error = error
                    print(f"[Fallback] Model '{candidate}' failed: {error}")

            if result is None:
                raise last_error

            # Add model attribution only to completed text answers. Tool-call
            # turns remain machine-readable for the client's tool loop.
            if config.show_model_suffix and result.get("choices"):
                message = result["choices"][0].get("message", {})
                content = message.get("content")
                if content and not _message_has_tool_calls(message):
                    suffix = _model_attribution_suffix(config, selected_model)
                    content = re.sub(r"\s*\[model:\s*[^\]\n]+\]\s*$", "", str(content))
                    message["content"] = f"{content.rstrip()}{suffix}"
            elif config.show_model_prefix and result.get("choices"):
                message = result["choices"][0].get("message", {})
                content = message.get("content")
                if content and not _message_has_tool_calls(message):
                    # Remove any existing prefix
                    content = re.sub(r'^\[[\w\-\.]+\]\s*', '', content)
                    message["content"] = f"[{selected_model}] {content}"

            result["model"] = selected_model
            return result

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        return await _serve_chat_completions(request)

    @app.post("/v1/responses")
    async def responses(request: ResponsesRequest):
        """Codex-compatible Responses API adapter.

        Routing and upstream retries stay on the existing Chat Completions path;
        this endpoint only translates request, tool, response, and SSE shapes.
        """
        chat_request, tool_name_map = responses_request_to_chat(request)

        available_models = list(config.llms.keys())
        should_route = request.model == "auto" or request.model not in available_models
        cache_context = None
        if should_route and request.prompt_cache_key:
            identity = request.user or request.safety_identifier or "anonymous"
            cache_context = (identity, request.prompt_cache_key, request.model)

        result = await _serve_chat_completions(chat_request, cache_context)

        if request.stream:
            if not isinstance(result, StreamingResponse):
                raise HTTPException(status_code=500, detail="Expected a streaming upstream response")
            return StreamingResponse(
                chat_stream_to_responses(
                    result.body_iterator,
                    request,
                    tool_name_map,
                    strip_model_prefix=config.show_model_prefix and not config.show_model_suffix,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        if not isinstance(result, dict):
            raise HTTPException(status_code=500, detail="Expected a JSON upstream response")
        return chat_response_to_responses(
            result,
            request,
            tool_name_map,
            strip_model_prefix=config.show_model_prefix and not config.show_model_suffix,
        )

    @app.post("/v1/analytics/codex/turn-costs", status_code=204)
    async def codex_turn_costs():
        """Accept optional Codex cost telemetry without persisting it."""
        return Response(status_code=204)

    @app.get("/")
    async def root():
        return {
            "name": "OpenClaw Router",
            "version": "1.0.0",
            "strategy": config.router.strategy,
            "llms": list(config.llms.keys()),
            "endpoints": {
                "chat": "POST /v1/chat/completions",
                "responses": "POST /v1/responses",
                "models": "GET /v1/models",
                "health": "GET /health"
            }
        }

    @app.get("/routers")
    async def list_routers():
        """List available routing strategies"""
        return {
            "available_routers": router.get_available_routers(),
            "current": config.router.strategy
        }

    @app.websocket("/v1/chat/ws")
    async def chat_websocket(websocket: WebSocket):
        """WebSocket endpoint for real-time streaming"""
        await websocket.accept()
        try:
            # Receive request
            data = await websocket.receive_json()
            request = ChatRequest(**data)
            messages = [{"role": m.role, "content": m.content} for m in request.messages]

            # Extract user query for routing
            user_query = ""
            last_user_idx = None
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    last_user_idx = i
                    break

            if last_user_idx is not None:
                raw_content = messages[last_user_idx]["content"]
                if config.media.enabled:
                    together_key = config.api_keys.get("together")
                    processed_text, _ = await process_multimodal_content(
                        raw_content, config.media, fallback_key=together_key
                    )
                    user_query = processed_text[:500]
                    messages[last_user_idx]["content"] = processed_text
                else:
                    user_query = normalize_content(raw_content)[:500]

            if not user_query:
                user_query = "general query"

            # Select model
            available_models = list(config.llms.keys())
            eligible_models, estimated_input_tokens = _eligible_models_for_request(
                config,
                messages,
                request.tools,
            )
            if not eligible_models:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Input requires approximately {estimated_input_tokens} tokens; "
                        "no configured model has enough context."
                    ),
                )
            if request.model == "auto" or request.model not in available_models:
                selected_model = await router.select_model(
                    user_query,
                    user=request.user,
                    candidate_models=eligible_models,
                )
                _safe_log(f"[WS Router] Query: '{user_query[:50]}...' -> {selected_model}")
            else:
                if request.model not in eligible_models:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Input exceeds the configured context limit for '{request.model}'.",
                    )
                selected_model = request.model

            # Call LLM backend in streaming mode
            prefix_sent = False
            content_buffer = ""
            buffered_chunks = []
            suffix_sent = False
            stream_saw_text = False
            stream_saw_tool_calls = False
            suffix_template: Optional[Dict[str, Any]] = None

            stream_gen = await backend.call(
                selected_model, messages, request.max_tokens,
                request.temperature,
                stream=True,
                stream_options=request.stream_options,
                tools=request.tools,
                tool_choice=request.tool_choice,
                parallel_tool_calls=request.parallel_tool_calls,
                top_p=request.top_p,
            )

            async for chunk in stream_gen:
                if config.show_model_suffix:
                    if "[DONE]" in chunk:
                        if stream_saw_text and not stream_saw_tool_calls and not suffix_sent:
                            await websocket.send_text(
                                _model_suffix_stream_chunk(
                                    suffix_template,
                                    _model_attribution_suffix(config, selected_model),
                                )
                            )
                            suffix_sent = True
                        await websocket.send_text(chunk)
                        continue

                    try:
                        json_str = chunk[6:] if chunk.startswith("data: ") else chunk
                        data_chunk = json.loads(json_str.strip())
                    except Exception:
                        await websocket.send_text(chunk)
                        continue

                    choices = data_chunk.get("choices") or []
                    if choices:
                        suffix_template = data_chunk
                    for choice in choices:
                        delta = choice.get("delta") or {}
                        if delta.get("content"):
                            stream_saw_text = True
                        if _delta_has_tool_calls(delta) or choice.get("finish_reason") in {
                            "tool_calls",
                            "function_call",
                        }:
                            stream_saw_tool_calls = True

                    terminal_chunk = any(choice.get("finish_reason") for choice in choices)
                    if (
                        terminal_chunk
                        and stream_saw_text
                        and not stream_saw_tool_calls
                        and not suffix_sent
                    ):
                        await websocket.send_text(
                            _model_suffix_stream_chunk(
                                suffix_template,
                                _model_attribution_suffix(config, selected_model),
                            )
                        )
                        suffix_sent = True

                    await websocket.send_text(chunk)
                    continue

                if not config.show_model_prefix:
                    await websocket.send_text(chunk)
                    continue

                if "[DONE]" in chunk:
                    if buffered_chunks and not prefix_sent:
                        content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                        first = buffered_chunks[0]
                        try:
                            data_chunk = json.loads(first[6:]) if first.startswith("data: ") else {}
                            if data_chunk.get("choices") and data_chunk["choices"][0].get("delta"):
                                data_chunk["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                                await websocket.send_text(f"data: {json.dumps(data_chunk)}\n\n")
                        except:
                            pass
                    await websocket.send_text(chunk)
                else:
                    try:
                        json_str = chunk[6:] if chunk.startswith("data: ") else chunk
                        data_chunk = json.loads(json_str.strip())
                        cleaned = clean_streaming_chunk(data_chunk)

                        if cleaned:
                            if cleaned.get("usage") and not cleaned.get("choices"):
                                if buffered_chunks and not prefix_sent:
                                    content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                                    first = buffered_chunks[0]
                                    try:
                                        first_data = json.loads(first[6:] if first.startswith("data: ") else first)
                                        if first_data.get("choices") and first_data["choices"][0].get("delta"):
                                            first_data["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                                            await websocket.send_text(f"data: {json.dumps(first_data)}\n\n")
                                            prefix_sent = True
                                            buffered_chunks = []
                                    except:
                                        pass
                                await websocket.send_json(cleaned)
                                continue

                            choices = cleaned.get("choices", [])
                            if choices and "delta" in choices[0]:
                                content = choices[0]["delta"].get("content", "")

                                if not prefix_sent:
                                    content_buffer += content
                                    buffered_chunks.append(chunk)

                                    if len(content_buffer) > 30 or (content_buffer and not content_buffer.startswith("[")):
                                        content_buffer = re.sub(r'^\[[\w\-\.]+\]\s*', '', content_buffer)
                                        first = buffered_chunks[0]
                                        first_data = json.loads(first[6:] if first.startswith("data: ") else first)
                                        if first_data.get("choices") and first_data["choices"][0].get("delta"):
                                            first_data["choices"][0]["delta"]["content"] = f"[{selected_model}] " + content_buffer
                                            await websocket.send_text(f"data: {json.dumps(first_data)}\n\n")
                                            prefix_sent = True
                                            buffered_chunks = []
                                else:
                                    await websocket.send_json(cleaned)
                            else:
                                if prefix_sent:
                                    await websocket.send_json(cleaned)
                    except:
                        await websocket.send_text(chunk)

        except WebSocketDisconnect:
            _safe_log("[WS] Client disconnected")
        except Exception as e:
            _safe_log(f"[WS Error] {type(e).__name__}: {e}")
            try:
                await websocket.send_json({"error": str(e)})
            except:
                pass
        finally:
            try:
                await websocket.close()
            except:
                pass

    return app


def run_server(app: FastAPI = None, config_path: str = None, host: str = "0.0.0.0", port: int = 8000):
    """Run the server"""
    if app is None:
        app = create_app(config_path=config_path)

    print(f"""
============================================================
  OpenClaw Router
============================================================
  Server: http://{host}:{port}
  Chat:   http://{host}:{port}/v1/chat/completions
  Codex:  http://{host}:{port}/v1/responses
  Health: http://{host}:{port}/health
============================================================
""")

    uvicorn.run(app, host=host, port=port)


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenClaw Router Server")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()

    run_server(config_path=args.config, host=args.host, port=args.port)
