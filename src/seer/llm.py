"""OpenAI 兼容接口的 LLM 客户端：配置、流式调用与 JSON 解析。"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

SYSTEM_PROMPT = "You produce strict JSON for scientific preprocessing."
TRUNCATED = ("length", "content_filter")


OUTAGE = (60.0, 600.0, 4 * 3600.0)  # 连接故障的首次等待、单次上限与总时长（秒）


def unreachable(error: BaseException) -> bool:
    """连接层故障（断网、连接被重置、超时）：与回答内容无关，等待后重发同一请求即可。"""

    import urllib.error

    import httpx
    from openai import APIConnectionError

    return isinstance(error, APIConnectionError | httpx.TransportError | urllib.error.URLError | TimeoutError)


def patiently(call: Callable[[], Any]) -> Any:
    """执行 ``call``；连接故障时按 ``OUTAGE`` 指数退避重发，其余错误照常抛出。"""

    waited, pause = 0.0, OUTAGE[0]
    while True:
        try:
            return call()
        except Exception as error:
            if not unreachable(error) or waited >= OUTAGE[2]:
                raise
            time.sleep(pause)
            waited, pause = waited + pause, min(2.0 * pause, OUTAGE[1])


@dataclass(frozen=True)
class LLMConfig:
    model: str
    base_url: str
    api_key_env: str
    provider: str = "openai_compatible"
    temperature: float | None = None
    reasoning_effort: str | None = None
    max_tokens: int = 65536
    token_parameter: str = "max_tokens"
    timeout_seconds: float = 600.0
    stream: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> LLMConfig:
        """configs/llm.yaml 中的一个模型条目（``Registry.entry``），或与之等价的映射。"""

        if not str(payload.get("model", "")).strip() or not str(payload.get("base_url", "")).strip():
            raise ValueError("LLM model and base_url must be set; fill configs/env/api.local.env")
        if payload.get("provider", "openai_compatible") != "openai_compatible":
            raise ValueError(f"Unsupported LLM provider: {payload.get('provider')}")
        if not isinstance(payload.get("stream", True), bool):
            raise ValueError("stream must be true or false")
        if payload.get("token_parameter", "max_tokens") not in ("max_tokens", "max_completion_tokens"):
            raise ValueError("token_parameter must be max_tokens or max_completion_tokens")
        optional = {None, "", "null", "none"}
        return cls(
            # model_id：平台模型 ID 与登记名不同时填写
            model=str(payload.get("model_id") or payload["model"]).strip(),
            base_url=str(payload["base_url"]).strip(),
            api_key_env=str(payload["api_key_env"]),
            provider=str(payload.get("provider", "openai_compatible")),
            temperature=None if payload.get("temperature") in optional else float(payload["temperature"]),
            reasoning_effort=(
                None if payload.get("reasoning_effort") in optional else str(payload["reasoning_effort"])
            ),
            max_tokens=int(payload.get("max_tokens", 65536)),
            token_parameter=str(payload.get("token_parameter", "max_tokens")),
            timeout_seconds=float(payload.get("timeout_seconds", 600.0)),
            stream=bool(payload.get("stream", True)),
        )

    def public(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if key not in ("api_key_env", "base_url")}


@dataclass(frozen=True)
class LLMReply:
    text: str
    model: str
    finish_reason: str | None
    usage: dict[str, int]
    seconds: float


Caller = Callable[[str, LLMConfig], LLMReply]
_JSON_FENCE = re.compile(r"```(?:json)?[ \t]*\r?\n(.*?)\r?\n```", re.DOTALL)


def parse_json_response(content: str) -> dict[str, Any]:
    """解析严格 JSON；也接受整个响应被一个 Markdown 代码围栏包裹的情况。"""

    if not content or not content.strip():
        raise RuntimeError("LLM returned an empty response")
    fenced = _JSON_FENCE.fullmatch(content.strip())
    payload = json.loads(fenced.group(1) if fenced else content)
    if not isinstance(payload, dict):
        raise RuntimeError("LLM response must be a JSON object")
    return payload


def call_llm(prompt: str, config: LLMConfig) -> LLMReply:
    """发送一次请求，返回回答文本、API 报告的模型名、结束原因与 token 用量。"""

    if config.provider != "openai_compatible":
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
    from openai import APIError

    keys = os.environ.get(config.api_key_env, "").split(",")
    if not keys[0]:
        raise RuntimeError(f"Missing API key environment variable: {config.api_key_env}")
    request: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        config.token_parameter: config.max_tokens,
        "response_format": {"type": "json_object"},
        "stream": config.stream,
    }
    if config.stream:
        request["stream_options"] = {"include_usage": True}
    if config.temperature is not None:
        request["temperature"] = config.temperature
    if config.reasoning_effort is not None:
        request["reasoning_effort"] = config.reasoning_effort
    for number, key in enumerate(keys, 1):  # 逗号分隔的多个密钥按序尝试，服务端不可用时换下一个
        try:
            return _request(request, key, config)
        except APIError as error:
            if number == len(keys) or getattr(error, "status_code", 503) < 500:
                raise


def _request(request: dict[str, Any], api_key: str, config: LLMConfig) -> LLMReply:
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=config.base_url, timeout=config.timeout_seconds, max_retries=0)
    start = time.perf_counter()
    try:
        response = client.chat.completions.create(**request)
        if config.stream:
            content, model, finish, usage = _collect_stream(response, config.model)
        else:
            choice = response.choices[0]
            content, finish = choice.message.content or "", choice.finish_reason
            model, usage = str(getattr(response, "model", None) or config.model), response.usage
    finally:
        client.close()
    return LLMReply(content, model, finish, _usage(usage), time.perf_counter() - start)


def _collect_stream(response: Any, model: str) -> tuple[str, str, str | None, Any]:
    """收集流式响应中的回答文本；推理 token 不属于回答。用量在最后一个分块中。"""

    parts: list[str] = []
    finish = usage = None
    for chunk in response:
        model = getattr(chunk, "model", None) or model
        usage = getattr(chunk, "usage", None) or usage
        for choice in chunk.choices:
            parts.append(choice.delta.content or "")
            finish = choice.finish_reason or finish
    return "".join(parts), model, finish, usage


def _usage(usage: Any) -> dict[str, int]:
    details = getattr(usage, "completion_tokens_details", None)
    values = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "reasoning_tokens": getattr(details, "reasoning_tokens", None),
    }
    return {key: int(value) for key, value in values.items() if value is not None}
