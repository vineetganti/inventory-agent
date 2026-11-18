# Inventory Management Agent: built step by step

*Code and design by **Shivank Agarwal** ([Future with Shivank](https://www.linkedin.com/in/shivankagrawal/)) for the AI Bootcamp at **Scaler Academy**. This copy is ported to run on Groq or on a Claude subscription; the write-up of every version, re-run on both, is at [vineetganti.com/blog/inventory-agent](https://vineetganti.com/blog/inventory-agent).*

An AI assistant that answers questions like **"Should I reorder toothpaste?"** by reading a live
inventory database. It's built in **8 small versions**, and each version adds **one** idea.
Run them in order to watch a single prompt grow into a production-style agent.

| Version | What it adds | Idea it teaches |
|---|---|---|
| `inventoryplanner_v0.py` | One prompt, numbers typed in by hand | The simplest LLM call |
| `inventoryplanner_v1.py` | A Gradio web form | Putting a UI on a model |
| `inventoryplanner_v2.py` | SQLite database + tools + agent (chat UI) | RAG and tool calling |
| `inventoryplanner_v3.py` | Rule-based input guardrails | Blocking prompt injection & jailbreaks |
| `inventoryplanner_v4.py` | Answer cache with 5-minute expiry + stats | Speed and cost savings |
| `inventoryplanner_v5.py` | Send an email alert | Write actions (the agent can act, not just read) |
| `inventoryplanner_v6.py` | LangSmith tracing | Observability |
| `inventoryplanner_v7.py` | `calculate_reorder_quantity` tool | Let code do the maths, not the model |

`database.py` creates `inventory.db` (19 sample products) automatically the first time v2+ runs.

## How one question flows (v7)

```
user question
   -> guardrail check        (blocked? reply with a warning)
   -> cache check            (asked in the last 5 minutes? reply instantly with 🔄)
   -> agent (see llm.py)     picks a tool:
        get_inventory_data / get_all_inventory / calculate_reorder_quantity
   -> tool reads inventory.db, returns facts
   -> agent writes the answer
   -> answer saved in the cache, shown in the chat, trace sent to LangSmith
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env        # then put your keys in .env
python inventoryplanner_v0.py
python inventoryplanner_v7.py   # opens a web page at http://127.0.0.1:7860
```

- **Every version** needs a model: `GROQ_API_KEY` (default), or `LLM_PROVIDER=claude` with Claude Code signed in. See below.
- **v5–v7** email button needs the `EMAIL_*` settings (for Gmail: an App Password, not your normal password).
- **v6–v7** tracing needs the `LANGSMITH_*` settings (free account at smith.langchain.com).
  Without them the app still works; it just doesn't send traces.

## The reorder rule used everywhere

```
days remaining = current stock / average daily demand
if days remaining <= lead time  ->  reorder at least (average daily demand x lead time) units
```
Example: toothpaste, 20 in stock, 5 sold per day, 4-day delivery -> 4 days left = lead time -> reorder 20 units now.

## Which model: `llm.py`

The course version calls OpenAI's `gpt-6-luna`. This copy changes only the lines that create the model, so every version runs on either provider:

| `LLM_PROVIDER` | Model | Needs |
|---|---|---|
| `groq` (default) | `openai/gpt-oss-120b` on Groq | `GROQ_API_KEY` |
| `claude` | Claude Haiku 5.5 through Claude Code's headless mode (`claude -p`) | Claude Code installed and signed in with a Claude subscription |

`LLM_MODEL` picks a different model id for the chosen provider (for example `openai/gpt-oss-20b`, which has its own daily free-tier quota on Groq).

Things to know about the Claude path:
- A subscription has no API key, so `llm.py` calls the Claude Code CLI once per model call. Tool calls are **emulated**: the tools are listed in the system prompt and the model replies with JSON naming the calls, which `llm.py` turns into the usual `tool_calls`.
- Claude Code adds the signed-in account's email, the OS, folder paths and today's date to every prompt; `llm.py` runs it from an empty folder and tells the model to ignore that context. Claude still knows today's date.
- Each call carries about 1,000 tokens of Claude Code's own wrapper and takes a few seconds.

## Changes from the course code

- The model and client lines (`llm.py`), and `reasoning_effort="none"` -> the lowest setting gpt-oss supports (`"low"`).
- Nothing else in `inventoryplanner_v*.py` or `database.py`. Known limits listed in the course explainer (first `LIKE` match only, no chat history, keyword guardrails that err both ways, a cache that can go stale) are kept on purpose; the write-up measures each.

