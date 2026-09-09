"""
Interactive AgentOR shell - a lightweight REPL in the spirit of the
Claude Code CLI: run `agentic-or` with no subcommand and get a persistent
prompt where you type command names directly (with or without a leading
'/'), see a menu, and keep a command history - instead of re-invoking
`agentic-or <cmd>` from the OS shell every time.

Stdlib only: `cmd`-free hand-rolled loop over `input()` + `shlex` + the very
same argparse parser/dispatcher the one-shot CLI uses, so behavior never
drifts between `agentic-or demo` and typing `demo` inside the shell.
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys

try:
    import readline  # noqa: F401  (POSIX only; enables history + line-editing for input())
except ImportError:
    pass

_CYAN = "\x1b[38;5;51m"
_DIM = "\x1b[2m"
_BOLD = "\x1b[1m"
_RESET = "\x1b[0m"

_EXIT_WORDS = {"exit", "quit", "q", ":q"}
_HELP_WORDS = {"help", "menu", "?"}
_CLEAR_WORDS = {"clear", "cls"}

# (command line to show, description) - kept in sync with build_parser() in cli.py.
_MENU_ROWS = [
    ("status [--watch]", "View / watch system telemetry (RAM, CPU, battery, temperature)"),
    ("run <file.json> [--monitor] [--fresh]", "Run a DAG pipeline from a task JSON file"),
    ("demo [--monitor]", "Run the built-in 11-task DAG demo (GitHub + HuggingFace)"),
    ("llm --provider <name>", "Run the real multi-agent LLM pipeline (mock/gemini/openai/groq/...)"),
    ("agents", "Show Agents running RIGHT NOW (real-time)"),
    ("agents add <file.py> [--persist]", "Plug in your own BaseWorker subclass (session-only, or permanent)"),
    ("agents persisted", "List agents saved with 'agents add --persist'"),
    ("agents reset [--persist]", "Drop custom agents (add --persist to also forget saved ones)"),
    ("history", "Show summaries of runs already completed in this session"),
    ("watch", "Watch ANOTHER agentic-or process's live run, from this terminal"),
    ("ui [--port N] [--no-browser]", "Open a lightweight local web dashboard (browser-based watch)"),
    ("clear", "Clear the screen"),
    ("help", "Show this menu"),
    ("exit / quit", "Exit the AgentOR shell"),
]


def _use_color(stream) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _print_menu() -> None:
    color = _use_color(sys.stdout)
    width = max(len(cmd) for cmd, _ in _MENU_ROWS) + 2
    sep = "─" * 62
    lines = [sep, "  Available commands (type directly, with or without a leading '/'):", sep]
    for cmd, desc in _MENU_ROWS:
        lines.append(f"  {cmd:<{width}} {desc}")
    lines.append(sep)
    text = "\n".join(lines)
    if color:
        print(f"{_DIM}{text}{_RESET}")
    else:
        print(text)


def _prompt(color: bool) -> str:
    if color:
        return f"{_BOLD}{_CYAN}AgentOR{_RESET} › "
    return "AgentOR > "


def run_repl(parser: argparse.ArgumentParser) -> None:
    """Persistent interactive shell. Reuses `parser` for every line typed."""
    color = _use_color(sys.stdout)
    print(f"{_DIM}Type help to see the menu, exit to quit (no quotes needed).{_RESET}\n" if color
          else "Type help to see the menu, exit to quit (no quotes needed).\n")

    prompt = _prompt(color)
    while True:
        try:
            line = input(prompt).strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            # Ctrl+C on an empty/typed line clears it, like bash - doesn't exit the shell.
            print()
            continue

        if not line:
            continue

        # Claude-Code-style '/command' aliases are accepted too.
        stripped = line[1:].strip() if line.startswith("/") else line

        try:
            tokens = shlex.split(stripped)
        except ValueError as e:
            print(f"❌ Syntax error: {e}")
            continue

        if not tokens:
            continue

        # Compare the shlex-parsed token (quotes/whitespace already stripped)
        # rather than the raw line, so 'help', "help", and help all work.
        head = tokens[0].lower()
        if len(tokens) == 1 and head in _EXIT_WORDS:
            break
        if len(tokens) == 1 and head in _HELP_WORDS:
            _print_menu()
            continue
        if len(tokens) == 1 and head in _CLEAR_WORDS:
            sys.stdout.write("\x1b[H\x1b[2J")
            sys.stdout.flush()
            continue

        try:
            args = parser.parse_args(tokens)
        except SystemExit:
            # argparse already printed its own usage/error/help - stay in the shell.
            continue

        if args.command is None:
            _print_menu()
            continue

        from agentic_or.cli import dispatch  # local import: avoids a circular import at module load
        try:
            dispatch(args, parser)
        except SystemExit:
            continue
        except KeyboardInterrupt:
            print("\n⏹  Command cancelled.")
        except Exception as e:
            print(f"❌ Error running '{stripped}': {e}")

    print("👋 Goodbye!")
