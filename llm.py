"""
One switch for the model behind every lab in this folder.

    LLM_PROVIDER=groq     openai/gpt-oss-120b on Groq (needs GROQ_API_KEY)                  [default]
    LLM_PROVIDER=claude   Claude Haiku 5.5 through Claude Code's headless mode (`claude -p`),
                          signed in with a Claude subscription. No API key is involved.

    LLM_MODEL=...         optional: a different model id for the chosen provider

The course code was written for OpenAI (gpt-6-luna). Only the lines that create a model or a
client change; everything else in the labs is the course's own code.

What each provider can and can't do, and how this file papers over it:

* Groq speaks the OpenAI API, so `make_client()` is a real `openai.OpenAI` pointed at Groq and
  `make_llm()` is LangChain's `ChatOpenAI` with Groq's base URL. Tool calls and JSON mode are native.
* The Claude subscription is reachable only through Claude Code, which answers one prompt at a
  time. `ChatClaudeCode` turns a LangChain conversation into one prompt, and uses the CLI's
  asks for plain-text JSON for everything structured: JSON mode, `with_structured_output`, and tool
  calls, and parses it here. Tool calls are therefore *emulated*: the model is shown the tool list and
  must reply with a JSON object that either names tool calls or gives the answer. The agent loop around it (LangChain's
  `create_agent`, or a hand-written loop) is unchanged.
* Neither has an embeddings endpoint, so both use the local sentence-transformers model
  all-MiniLM-L6-v2 (384 numbers per text, runs on the CPU).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from functools import lru_cache
from types import SimpleNamespace
from typing import Any, Optional, Sequence

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODELS = {"groq": "openai/gpt-oss-120b", "claude": "claude-haiku-5-5"}
if PROVIDER not in DEFAULT_MODELS:
    raise ValueError(f"LLM_PROVIDER must be 'groq' or 'claude', not {PROVIDER!r}")
MODEL = os.getenv("LLM_MODEL") or DEFAULT_MODELS[PROVIDER]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# The course passes reasoning_effort="none" (answer at once). gpt-oss's lowest setting is "low";
# the Claude path ignores it.
FAST_EFFORT = "low"

# US$ per 1M tokens (input, output), list prices. The Claude subscription is not billed per token;
# its number is what the same tokens would cost on the API, for comparison only.
PRICES_PER_M = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),   # used when 120b's 200K-tokens-a-day free quota ran out
    "claude-haiku-5-5": (0.10, 0.50),
}


def price(model: str = MODEL) -> tuple[float, float]:
    """(input, output) US$ per 1M tokens for `model`, or (0, 0) if unknown."""
    return PRICES_PER_M.get(model, (0.0, 0.0))


def describe() -> str:
    """One line for the top of a lab's output."""
    via = "Groq" if PROVIDER == "groq" else "Claude Code (subscription)"
    return f"model: {MODEL} via {via}"


# ---------------------------------------------------------------------------
# The Claude Code CLI, used as a one-shot model call
# ---------------------------------------------------------------------------

# Claude Code adds the signed-in account's email, the OS, folder paths and today's date to every prompt, and a
# subscription login can't turn that off. The labs never gave the model any of it, so tell the model to ignore it.
HOST_NOTE = (
    "You are the language model inside a small application, answering its requests. Your context may contain "
    "information about an account, an email address, a computer, folders or the date: it comes from the hosting "
    "tool, not from this application's user. Never use it or mention it, and never guess the user's name from it."
)


def _find_claude_cli() -> str:
    found = os.getenv("CLAUDE_CLI") or shutil.which("claude")
    if found:
        return found
    # The VS Code extension ships the same binary; take the newest installed version.
    pattern = os.path.expanduser("~/.vscode/extensions/anthropic.claude-code-*/resources/native-binary/claude*")
    candidates = sorted(glob.glob(pattern), key=os.path.getmtime)
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("Claude Code CLI not found. Install it or set CLAUDE_CLI to its path.")


def _claude_once(system: str, prompt: str, schema: Optional[dict], model: str) -> dict:
    """One `claude -p` call with no tools, no MCP servers and no user settings. Returns the parsed JSON result."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(HOST_NOTE + "\n\n" + (system or "You are a helpful assistant."))
        system_file = f.name
    cmd = [
        _find_claude_cli(), "-p",
        "--model", model,
        "--tools", "",                 # no built-in tools: this is a plain model call
        "--setting-sources", "",       # ignore the user's CLAUDE.md, hooks and settings
        "--strict-mcp-config",         # and their MCP servers
        "--no-session-persistence",
        "--output-format", "json",
        "--system-prompt-file", system_file,
    ]
    if schema is not None:
        cmd += ["--json-schema", json.dumps(schema)]
    # Claude Code always tells the model its working folder and today's date (that can't be switched off with a
    # custom system prompt), so run it from an empty folder: the model learns nothing about this project.
    neutral_cwd = os.path.join(tempfile.gettempdir(), "claude-lab-cwd")
    os.makedirs(neutral_cwd, exist_ok=True)
    try:
        last_error = ""
        for attempt in range(4):
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=600,
                                  cwd=neutral_cwd)
            if proc.returncode == 0:
                data = json.loads(proc.stdout)
                if not data.get("is_error"):
                    return data
                last_error = str(data.get("result"))[:300]
            else:
                last_error = (proc.stderr or proc.stdout)[:300]
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"claude -p failed: {last_error}")
    finally:
        os.unlink(system_file)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _render(messages: Sequence[BaseMessage]) -> tuple[str, str]:
    """Split a conversation into (system prompt, user prompt) for one CLI call."""
    system = "\n\n".join(_text(m.content) for m in messages if isinstance(m, SystemMessage))
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    if len(rest) == 1 and isinstance(rest[0], HumanMessage):
        return system, _text(rest[0].content)
    lines = ["Here is the conversation so far. Write the assistant's next message.", ""]
    for m in rest:
        if isinstance(m, HumanMessage):
            lines += ["[user]", _text(m.content), ""]
        elif isinstance(m, AIMessage):
            if _text(m.content).strip():
                lines += ["[assistant]", _text(m.content), ""]
            for tc in m.tool_calls or []:
                lines += [f"[assistant called tool `{tc['name']}` with {json.dumps(tc['args'])}]", ""]
        elif isinstance(m, ToolMessage):
            lines += [f"[result of tool `{getattr(m, 'name', None) or 'tool'}`]", _text(m.content), ""]
        else:
            lines += [f"[{m.type}]", _text(m.content), ""]
    return system, "\n".join(lines)


class ChatClaudeCode(BaseChatModel):
    """A LangChain chat model backed by `claude -p`. See the module docstring for what is emulated."""

    model: str = DEFAULT_MODELS["claude"]
    tools: list = []            # OpenAI-format tool dicts, set by bind_tools
    tool_choice: Optional[str] = None

    @property
    def _llm_type(self) -> str:
        return "claude-code-cli"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        converted = [convert_to_openai_tool(t) for t in tools]
        if isinstance(tool_choice, dict):  # {"type": "function", "function": {"name": ...}}
            tool_choice = tool_choice.get("function", {}).get("name") or tool_choice.get("name")
        if tool_choice in ("any", "required", True) and len(converted) == 1:
            tool_choice = converted[0]["function"]["name"]
        return self.model_copy(update={"tools": converted, "tool_choice": tool_choice})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        # Every structured reply (tool calls, forced arguments, JSON) is asked for as plain-text JSON and parsed
        # here. The CLI's --json-schema option is not used: it works through a structured-output tool, and with a
        # tool in play the model sometimes tries to call the lab's functions by name, which fail ("No such tool").
        system, prompt = _render(messages)
        mode, shape, forced = "text", None, None
        response_format = kwargs.get("response_format")
        if self.tools and self.tool_choice not in (None, "auto", "none"):
            forced = next((t for t in self.tools if t["function"]["name"] == self.tool_choice), None)
        if forced is not None:
            mode = "forced"
            shape = forced["function"].get("parameters") or {"type": "object"}
            system += (f"\n\nYour whole reply is one JSON object: the arguments for `{forced['function']['name']}` "
                       f"({forced['function'].get('description', '')}), matching this JSON Schema:\n{json.dumps(shape)}")
        elif self.tools and self.tool_choice != "none":
            mode = "tools"
            catalogue = json.dumps([t["function"] for t in self.tools], indent=1)
            system += (
                "\n\nYou are the decision step of an application that can run the functions listed below; it runs them, "
                "not you. Your whole reply is one JSON object with two keys:\n"
                '  "tool_calls": a list of {"name": <function name>, "arguments": {...}} for the functions you need run '
                "(empty list if none),\n"
                '  "answer": your reply to the user (empty string while you still need function results).\n'
                "The application runs the listed functions and adds their results to the conversation, then asks you "
                "again. Ask for a function whenever its result would make your answer exact or grounded in data.\n\n"
                "Functions:\n" + catalogue
            )
        elif response_format:
            mode = "json"
            if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
                shape = response_format["json_schema"]["schema"]
                system += f"\n\nYour whole reply is one JSON object matching this JSON Schema:\n{json.dumps(shape)}"
            else:
                system += "\n\nYour whole reply is one valid JSON object."
        if mode != "text":
            system += "\nNo other text and no code fences."

        retries, parsed = 0, None
        while True:
            data = _claude_once(system, prompt, None, self.model)
            content = data.get("result") or ""
            if mode == "text":
                break
            parsed = _parse_json_object(content)
            ok = isinstance(parsed, dict) and (mode != "tools" or "tool_calls" in parsed or "answer" in parsed)
            if ok or retries >= 2:
                break
            retries += 1
            prompt += "\n\n(Your last reply was not one valid JSON object. Reply with the JSON object only.)"

        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0) + usage.get("cache_creation_input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        meta = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
        tool_calls = []
        parsed = parsed if isinstance(parsed, dict) else {}
        if mode == "forced":
            tool_calls = [{"name": forced["function"]["name"], "args": parsed, "id": f"call_{uuid.uuid4().hex[:12]}"}]
            content = ""
        elif mode == "tools":
            names = {t["function"]["name"] for t in self.tools}
            for tc in parsed.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("name") in names:
                    tool_calls.append({"name": tc["name"], "args": tc.get("arguments") or {}, "id": f"call_{uuid.uuid4().hex[:12]}"})
            content = "" if tool_calls else str(parsed.get("answer", content))
        elif mode == "json":
            content = json.dumps(parsed)
        message = AIMessage(
            content=content,
            tool_calls=tool_calls,
            usage_metadata=meta,
            response_metadata={"model_name": self.model, "duration_ms": data.get("duration_ms"),
                               "cli_turns": data.get("num_turns"), "json_retries": retries},
        )
        return ChatResult(generations=[ChatGeneration(message=message)], llm_output={"model_name": self.model, "token_usage": meta})


def _parse_json_object(text: str):
    """The first {...} in a reply, parsed; None if there isn't a valid one."""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# What the labs call
# ---------------------------------------------------------------------------

def make_llm(**kwargs) -> BaseChatModel:
    """The chat model for this run. kwargs go to ChatOpenAI on Groq and are ignored for Claude."""
    if PROVIDER == "groq":
        from langchain_openai import ChatOpenAI

        kwargs.setdefault("reasoning_effort", "low")  # gpt-oss has no "none"; "low" is its fastest setting
        kwargs.setdefault("max_retries", 6)            # Groq's free tier rate-limits bursts
        return ChatOpenAI(model=MODEL, base_url=GROQ_BASE_URL, api_key=os.environ["GROQ_API_KEY"], **kwargs)
    return ChatClaudeCode(model=MODEL)


class _NS(SimpleNamespace):
    """A SimpleNamespace that, like the OpenAI SDK's objects, can turn itself back into a dict."""

    def model_dump(self, **_):
        def plain(v):
            if isinstance(v, _NS):
                return v.model_dump()
            if isinstance(v, list):
                return [plain(x) for x in v]
            return v
        return {k: plain(v) for k, v in vars(self).items() if k != "parsed"}


class _ClaudeCompletions:
    """The part of `client.chat.completions` the labs use: create() and parse()."""

    def __init__(self, llm: ChatClaudeCode):
        self._llm = llm

    @staticmethod
    def _wrap(msg: AIMessage, parsed=None):
        calls = [
            _NS(id=tc["id"], type="function", function=_NS(name=tc["name"], arguments=json.dumps(tc["args"])))
            for tc in msg.tool_calls
        ]
        u = msg.usage_metadata or {}
        message = _NS(role="assistant", content=msg.content or None, tool_calls=calls or None, parsed=parsed)
        return _NS(
            model=msg.response_metadata.get("model_name"),
            choices=[_NS(index=0, message=message, finish_reason="tool_calls" if calls else "stop")],
            usage=_NS(prompt_tokens=u.get("input_tokens", 0), completion_tokens=u.get("output_tokens", 0), total_tokens=u.get("total_tokens", 0)),
        )

    @staticmethod
    def _plain(messages):
        # The labs append the SDK's own message objects to the history; turn them back into dicts.
        out = []
        for m in messages:
            d = m.model_dump() if hasattr(m, "model_dump") else dict(m)
            d = {k: v for k, v in d.items() if v is not None}
            d.setdefault("content", "")
            out.append(d)
        return out

    def create(self, *, messages, tools=None, tool_choice=None, response_format=None, model=None, **_ignored):
        messages = self._plain(messages)
        llm = self._llm
        if tools:
            llm = llm.bind_tools(tools, tool_choice=tool_choice)
        msg = llm.invoke(convert_to_messages(messages), **({"response_format": response_format} if response_format else {}))
        return self._wrap(msg)

    def parse(self, *, messages, response_format, model=None, **_ignored):
        schema = response_format.model_json_schema()
        msg = self._llm.invoke(
            convert_to_messages(self._plain(messages)),
            response_format={"type": "json_schema", "json_schema": {"name": response_format.__name__, "schema": schema}},
        )
        parsed = response_format.model_validate_json(msg.content)
        return self._wrap(msg, parsed=parsed)


def make_client():
    """An OpenAI-style client for the labs that call the raw SDK (client.chat.completions.create / .parse)."""
    if PROVIDER == "groq":
        from openai import OpenAI

        return OpenAI(base_url=GROQ_BASE_URL, api_key=os.environ["GROQ_API_KEY"], max_retries=6)
    completions = _ClaudeCompletions(ChatClaudeCode(model=MODEL))
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


@lru_cache(maxsize=1)
def _sentence_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts locally (all-MiniLM-L6-v2, normalised so dot product = cosine similarity)."""
    return _sentence_model().encode(list(texts), normalize_embeddings=True).tolist()


def make_embeddings():
    """A LangChain Embeddings object for the same local model."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
