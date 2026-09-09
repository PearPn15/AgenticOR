"""
Test script for executing real LLM tasks via Agentic-OR Orchestrator.
Supports Gemini, OpenAI, Groq, DeepSeek, OpenRouter, and Ollama.
"""

import os
import sys
import argparse
import asyncio
import time
from agentic_or.models import TaskNode, WorkloadType
from agentic_or.orchestrator import Orchestrator


async def run_llm_pipeline(provider: str, api_key: str, model: str):
    print("=" * 70)
    print(f"      🧠 AGENTIC-OR: REAL LLM PIPELINE ({provider.upper()})")
    print("=" * 70)

    orchestrator = Orchestrator(
        num_local_workers=2,
        num_api_workers=3,
        num_browser_workers=1,
        alns_time_budget_ms=50
    )

    # Multi-Agent LLM DAG Workflow:
    # Task 1 (Local): Setup input query
    # Task 2 (LLM API): Brainstorm 3 innovative features for desktop agent orchestration
    # Task 3 (LLM API - Parallel): Analyze hardware safety challenges for local agents
    # Task 4 (LLM API): Synthesize findings from Task 2 & Task 3 into an executive summary
    # Task 5 (Local): Save final response to markdown file
    tasks = [
        TaskNode(
            task_id="llm_task_01_init",
            name="Initialize Prompt Context",
            workload_type=WorkloadType.LOCAL,
            estimated_duration_ms=20,
        ),
        TaskNode(
            task_id="llm_task_02_brainstorm",
            name="LLM Agent 1: Brainstorm Features",
            workload_type=WorkloadType.API,
            predecessors=["llm_task_01_init"],
            estimated_duration_ms=1500,
            token_cost=500,
            affinity_key=f"provider:{provider}",
            metadata={
                "llm_prompt": (
                    "Briefly list the 3 most groundbreaking features for an AI Desktop Orchestrator "
                    "that combines C++ Operations Research with LLM Agents. Answer concisely in English."
                ),
                "llm_provider": provider,
                "api_key": api_key,
                "model": model,
            }
        ),
        TaskNode(
            task_id="llm_task_03_safety",
            name="LLM Agent 2: Analyze Hardware Safety",
            workload_type=WorkloadType.API,
            predecessors=["llm_task_01_init"],
            estimated_duration_ms=1500,
            token_cost=500,
            affinity_key=f"provider:{provider}",
            metadata={
                "llm_prompt": (
                    "Briefly analyze the 2 biggest hardware resource risks (RAM, CPU, Temperature) "
                    "when running many Multi-Agents on a Laptop, and how to prevent them. Answer concisely in English."
                ),
                "llm_provider": provider,
                "api_key": api_key,
                "model": model,
            }
        ),
        TaskNode(
            task_id="llm_task_04_synthesis",
            name="LLM Agent 3: Synthesize Executive Summary",
            workload_type=WorkloadType.API,
            predecessors=["llm_task_02_brainstorm", "llm_task_03_safety"],
            estimated_duration_ms=2000,
            token_cost=800,
            affinity_key=f"provider:{provider}",
            metadata={
                "llm_prompt": (
                    "Synthesize the groundbreaking ideas and hardware-protection solutions into a "
                    "concise 3-sentence manifesto for the Desktop-Agent-OR product. Answer in English."
                ),
                "llm_provider": provider,
                "api_key": api_key,
                "model": model,
            }
        ),
        TaskNode(
            task_id="llm_task_05_export",
            name="Export Report",
            workload_type=WorkloadType.LOCAL,
            predecessors=["llm_task_04_synthesis"],
            estimated_duration_ms=30,
        )
    ]

    print(f"\n🚀 Submitting 5-node Multi-Agent LLM DAG to Orchestrator...")
    orchestrator.submit_tasks(tasks)

    t0 = time.time()
    summary = await orchestrator.run()
    elapsed = time.time() - t0

    print("\n" + "=" * 70)
    print(f"  🎉 LLM PIPELINE COMPLETED IN {elapsed:.2f}s!")
    print("=" * 70)

    # Print out LLM responses from checkpoints
    for tid in ["llm_task_02_brainstorm", "llm_task_03_safety", "llm_task_04_synthesis"]:
        cp = orchestrator.broker._checkpoints.get(tid)
        task = orchestrator.broker.get_task(tid)
        print(f"\n📌 [{task.name}]")
        if cp and cp.status == "COMPLETED":
            print(f"{cp.result}")
        else:
            print(f"❌ Error: {cp.error if cp else 'No result'}")
    print("\n" + "=" * 70)

    return orchestrator


def main():
    parser = argparse.ArgumentParser(description="Run real LLM pipeline with Agentic-OR")
    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "groq", "deepseek", "openrouter", "ollama", "mock"],
        default="mock",
        help="LLM Provider (default: mock - runs immediately without requiring API key)"
    )
    parser.add_argument("--key", help="API Key (or set GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY)")
    parser.add_argument("--model", help="Model name (e.g. gemini-2.0-flash, gpt-4o-mini, qwen/qwen3.8-27b)")

    args = parser.parse_args()

    # Determine default model per provider
    default_models = {
        "gemini": "gemini-3.5-flash-lite",
        "openai": "gpt-4o-mini",
        "groq": "qwen/qwen3.8-27b",
        "deepseek": "deepseek-chat",
        "openrouter": "google/gemini-2.0-flash-exp:free",
        "ollama": "llama3",
        "mock": "mock-model",
    }
    model = args.model or default_models.get(args.provider, "gemini-3.5-flash-lite")

    # Determine API key
    api_key = args.key
    if not api_key:
        if args.provider == "gemini":
            api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        elif args.provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY")
        elif args.provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY")
        elif args.provider == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY")
        elif args.provider in ("ollama", "mock"):
            api_key = "none"

    if not api_key and args.provider not in ("ollama", "mock"):
        print(f"❌ Error: Please provide an API key for provider '{args.provider}'.", file=sys.stderr)
        print(f"Usage:", file=sys.stderr)
        print(f"  uv run python test_llm.py --provider {args.provider} --key <YOUR_API_KEY>", file=sys.stderr)
        print(f"Or try it instantly, no key needed:", file=sys.stderr)
        print(f"  uv run python test_llm.py --provider mock", file=sys.stderr)
        sys.exit(1)

    asyncio.run(run_llm_pipeline(args.provider, api_key or "", model))


if __name__ == "__main__":
    main()

