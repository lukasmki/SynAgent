"""Convert traces into a supervised fine-tuning dataset.

Emits `{"messages": [...], "tools": [...]}` per line -- the format TRL, axolotl,
Unsloth, and OpenAI finetuning all consume, and one that converts mechanically to
Gemini's `contents`/`systemInstruction` shape.

Each `model_call` record is exactly one model call, so the target is unambiguous:
the recorded response is the assistant turn, everything before it is the prompt.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def _stringify(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            c if isinstance(c, str) else json.dumps(c, default=str) for c in content
        ]
        return "\n".join(parts)
    return json.dumps(content, default=str)


def _arguments(args: Any) -> str:
    """Tool call arguments as the JSON string the chat format expects."""
    if isinstance(args, str):
        return args
    return json.dumps(args if args is not None else {}, default=str)


def _assistant_message(
    response: dict[str, Any], *, include_thinking: bool
) -> dict[str, Any]:
    text: list[str] = []
    thinking: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for part in response.get("parts", []):
        kind = part.get("part_kind")
        if kind == "text":
            text.append(part.get("content") or "")
        elif kind == "thinking":
            thinking.append(part.get("content") or "")
        elif kind == "tool-call":
            tool_calls.append(
                {
                    "id": part.get("tool_call_id"),
                    "type": "function",
                    "function": {
                        "name": part.get("tool_name"),
                        "arguments": _arguments(part.get("args")),
                    },
                }
            )
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if include_thinking and thinking:
        message["reasoning_content"] = "".join(thinking)
    return message


def _to_chat_messages(
    messages: list[dict[str, Any]], *, include_thinking: bool
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.get("kind") == "response":
            out.append(_assistant_message(message, include_thinking=include_thinking))
            continue
        for part in message.get("parts", []):
            kind = part.get("part_kind")
            if kind == "system-prompt":
                out.append(
                    {"role": "system", "content": _stringify(part.get("content"))}
                )
            elif kind == "user-prompt":
                out.append({"role": "user", "content": _stringify(part.get("content"))})
            elif kind == "tool-return":
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": part.get("tool_call_id"),
                        "content": _stringify(part.get("content")),
                    }
                )
            elif kind == "retry-prompt":
                # A retry with a tool_call_id is the model being told its call was
                # rejected; without one it reaches the model as a user turn.
                if part.get("tool_call_id"):
                    out.append(
                        {
                            "role": "tool",
                            "tool_call_id": part["tool_call_id"],
                            "content": _stringify(part.get("content")),
                        }
                    )
                else:
                    out.append(
                        {"role": "user", "content": _stringify(part.get("content"))}
                    )
    return out


def _to_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description") or "",
                "parameters": tool.get("parameters_json_schema") or {},
            },
        }
        for tool in tools
    ]


def read_records(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Yield every record across the given trace files, skipping unreadable lines."""
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def export_sft(
    paths: Iterable[Path],
    out: Path,
    *,
    include_thinking: bool = False,
    only_successful: bool = False,
    agent: str | None = None,
    min_steps: int = 0,
) -> int:
    """Write a chat-format SFT dataset. Returns the number of samples written."""
    paths = list(paths)

    # First pass: run outcomes, so samples can be filtered by trajectory quality.
    status: dict[str, str] = {}
    steps: dict[str, int] = {}
    valid: dict[str, bool] = {}
    for record in read_records(paths):
        run_id = record.get("run_id")
        if not run_id:
            continue
        if record.get("record") == "run":
            status[run_id] = record.get("status", "ok")
            steps[run_id] = record.get("n_steps", 0)
        elif record.get("record") == "label" and "valid" in record:
            # Any invalid label for a run disqualifies it.
            valid[run_id] = valid.get(run_id, True) and bool(record["valid"])

    out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    written = 0

    with out.open("w", encoding="utf-8") as fh:
        for record in read_records(paths):
            if record.get("record") != "model_call" or record.get("error"):
                continue
            response = record.get("output")
            if not response:
                continue
            run_id = record.get("run_id") or ""
            if agent is not None and record.get("agent") != agent:
                continue
            if min_steps and steps.get(run_id, 0) < min_steps:
                continue
            if only_successful and (
                status.get(run_id) != "ok" or not valid.get(run_id, True)
            ):
                continue

            source = record.get("input") or {}
            messages = _to_chat_messages(
                source.get("messages") or [], include_thinking=include_thinking
            )
            if instructions := source.get("instructions"):
                messages.insert(0, {"role": "system", "content": instructions})
            messages.append(
                _assistant_message(response, include_thinking=include_thinking)
            )

            sample = {
                "messages": messages,
                "tools": _to_tools(source.get("tools") or []),
            }
            digest = hashlib.sha256(
                json.dumps(sample, sort_keys=True, default=str).encode()
            ).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            fh.write(json.dumps(sample, default=str) + "\n")
            written += 1

    return written
