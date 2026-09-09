"""
Central PLANNER: turns a natural-language goal into a task-JSON file that
`agentic-or run` can execute - it does NOT submit or run anything itself.

Flow: goal (text) -> LLM -> validated task list -> written to disk -> YOU
review the printed summary and the file, then run it yourself:

    export GROQ_API_KEY="..."   # or GEMINI_API_KEY
    uv run python my_agents/planner_agent.py "Summarize the latest AgentOR docs"
    # review my_agents/generated_pipeline.json, then:
    uv run agentic-or
    AgentOR › agents add my_agents/web_fetch_agent.py --count 2
    AgentOR › agents add my_agents/pipeline_agents.py --class CrawlerAgent
    AgentOR › agents add my_agents/pipeline_agents.py --class ProcessorAgent
    AgentOR › run my_agents/generated_pipeline.json

Nothing here ever calls orchestrator.submit_tasks()/run() - by design (see
docs/custom-agents.md#multi-agent-pipelines and the planner-mode decision
recorded there): an LLM-generated plan is reviewed by a human before any
task in it - a real web fetch, a paid API call, anything - actually runs.

AGENT_CATALOG below describes the example agents already in this folder, so
the LLM knows what's actually available to target and what `metadata` shape
each one expects. Edit it to describe YOUR agents when you add your own.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

AGENT_CATALOG = [
    {
        "class": "WebFetchSummarizerAgent",
        "workload_type": "API",
        "file": "my_agents/web_fetch_agent.py",
        "does": "Fetches one URL over real HTTP and summarizes it via LLM.",
        "metadata_shape": {"url": "<https url to fetch>"},
    },
    {
        "class": "CrawlerAgent",
        "workload_type": "BROWSER",
        "file": "my_agents/pipeline_agents.py",
        "does": "Produces raw text data about a topic (stage 1 of a 2-stage pipeline).",
        "metadata_shape": {"topic": "<topic string>"},
    },
    {
        "class": "ProcessorAgent",
        "workload_type": "LOCAL",
        "file": "my_agents/pipeline_agents.py",
        "does": (
            "Processes a CrawlerAgent task's output. MUST list that CrawlerAgent "
            "task's task_id in both 'predecessors' and metadata.read_from."
        ),
        "metadata_shape": {"read_from": "<task_id of a CrawlerAgent task>"},
    },
]

REQUIRED_TASK_FIELDS = {"task_id", "name", "workload_type", "predecessors", "metadata"}
VALID_WORKLOAD_TYPES = {"LOCAL", "API", "BROWSER"}


def _build_prompt(goal: str) -> str:
    catalog_desc = "\n".join(
        f"- {a['class']} (workload_type={a['workload_type']}): {a['does']} "
        f"metadata must look like {json.dumps(a['metadata_shape'])}."
        for a in AGENT_CATALOG
    )
    return f"""You are a task-planning engine for a multi-agent orchestrator called AgentOR.
Break the GOAL below into a minimal list of tasks the AVAILABLE AGENTS can execute.

AVAILABLE AGENTS:
{catalog_desc}

Output ONLY a raw JSON array (no markdown fences, no prose before/after), where
each element matches exactly this shape:
{{
  "task_id": "unique_snake_case_id",
  "name": "short human-readable name",
  "workload_type": "LOCAL" | "API" | "BROWSER",
  "predecessors": ["task_id", ...],
  "estimated_duration_ms": 500,
  "ram_mb": 100,
  "metadata": {{ ... exactly the fields the chosen agent's metadata_shape requires ... }}
}}

RULES:
- Every id in "predecessors" must be some other task's "task_id" in this same array.
- No cycles.
- Use ONLY the workload_type + metadata shape of one of the AVAILABLE AGENTS above
  for every task - do not invent capabilities that aren't listed.
- A ProcessorAgent task MUST have its CrawlerAgent's task_id in BOTH
  "predecessors" and metadata.read_from.
- Keep it minimal: only the tasks actually needed for the goal.

GOAL: {goal}
"""


async def _call_llm(prompt: str) -> str:
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    provider = os.environ.get("AGENT_LLM_PROVIDER", "").lower() or ("groq" if groq_key else "gemini")
    api_key = groq_key if provider == "groq" else gemini_key
    if not api_key:
        raise RuntimeError("No GROQ_API_KEY/GEMINI_API_KEY set in this process's environment.")

    async with httpx.AsyncClient(timeout=30.0) as client:
        if provider == "gemini":
            model = os.environ.get("AGENT_LLM_MODEL", "gemini-3.5-flash-lite")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            resp = await client.post(url, json={"contents": [{"parts": [{"text": prompt}]}]})
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

        model = os.environ.get("AGENT_LLM_MODEL", "qwen/qwen3.8-27b")
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def validate_plan(tasks: Any) -> list[str]:
    """Returns a list of problems found (empty = valid). Never raises."""
    problems: list[str] = []
    if not isinstance(tasks, list) or not tasks:
        return ["Plan must be a non-empty JSON array of task objects."]

    ids_seen: set[str] = set()
    for i, t in enumerate(tasks):
        if not isinstance(t, dict):
            problems.append(f"Task #{i} is not an object.")
            continue
        missing = REQUIRED_TASK_FIELDS - t.keys()
        if missing:
            problems.append(f"Task #{i} ({t.get('task_id', '?')}) missing fields: {sorted(missing)}")
            continue
        tid = t["task_id"]
        if tid in ids_seen:
            problems.append(f"Duplicate task_id: '{tid}'")
        ids_seen.add(tid)
        if t["workload_type"] not in VALID_WORKLOAD_TYPES:
            problems.append(f"Task '{tid}': invalid workload_type '{t['workload_type']}'")
        if not isinstance(t["predecessors"], list):
            problems.append(f"Task '{tid}': predecessors must be a list")

    for t in tasks:
        if not isinstance(t, dict) or "predecessors" not in t:
            continue
        for p in t["predecessors"]:
            if p not in ids_seen:
                problems.append(f"Task '{t.get('task_id')}': predecessor '{p}' does not exist")

    if not problems:
        cycle = _find_cycle(tasks)
        if cycle:
            problems.append(f"Cycle detected: {' -> '.join(cycle)}")

    return problems


def _find_cycle(tasks: list[dict]) -> list[str] | None:
    graph = {t["task_id"]: t.get("predecessors", []) for t in tasks}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {tid: WHITE for tid in graph}
    path: list[str] = []

    def visit(node: str) -> list[str] | None:
        color[node] = GRAY
        path.append(node)
        for pred in graph.get(node, []):
            if pred not in graph:
                continue
            if color[pred] == GRAY:
                return path[path.index(pred):] + [pred]
            if color[pred] == WHITE:
                found = visit(pred)
                if found:
                    return found
        path.pop()
        color[node] = BLACK
        return None

    for tid in graph:
        if color[tid] == WHITE:
            found = visit(tid)
            if found:
                return found
    return None


def print_summary(tasks: list[dict]) -> None:
    sep = "-" * 78
    print(sep)
    print(f"  {'TASK_ID':<20}{'TYPE':<10}{'DEPENDS ON':<25}{'METADATA'}")
    print(sep)
    for t in tasks:
        deps = ",".join(t.get("predecessors", [])) or "-"
        meta = json.dumps(t.get("metadata", {}))
        print(f"  {t['task_id']:<20}{t['workload_type']:<10}{deps:<25}{meta[:40]}")
    print(sep)


async def plan(goal: str, output_path: str = "my_agents/generated_pipeline.json") -> list[dict]:
    prompt = _build_prompt(goal)
    raw = await _call_llm(prompt)
    cleaned = _strip_markdown_fences(raw)

    try:
        tasks = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"LLM did not return valid JSON: {e}\n--- raw output ---\n{raw}") from e

    problems = validate_plan(tasks)
    if problems:
        raise RuntimeError("Generated plan failed validation:\n" + "\n".join(f"  - {p}" for p in problems))

    Path(output_path).write_text(json.dumps(tasks, indent=2))

    print(f"\n✅ Plan validated OK - {len(tasks)} task(s) written to {output_path}\n")
    print_summary(tasks)
    print(f"\nNothing has been run. Review {output_path}, then:")
    print(f"  uv run agentic-or")
    print(f"  AgentOR › agents add <the .py file(s) for the agents this plan uses>")
    print(f"  AgentOR › run {output_path}")

    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan a task pipeline from a natural-language goal.")
    parser.add_argument("goal", help="What you want the pipeline to accomplish")
    parser.add_argument("--out", default="my_agents/generated_pipeline.json", help="Where to write the plan")
    args = parser.parse_args()

    import asyncio
    try:
        asyncio.run(plan(args.goal, args.out))
    except RuntimeError as e:
        print(f"❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
