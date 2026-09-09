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
import shlex
import sys

from agentic_or import theme

try:
    import readline  # noqa: F401  (POSIX only; enables history + line-editing for input())
except ImportError:
    pass

_EXIT_WORDS = {"exit", "quit", "q", ":q"}
_HELP_WORDS = {"help", "menu", "?"}
_CLEAR_WORDS = {"clear", "cls"}

# (section title, [(command line to show, description), ...]) - kept in sync
# with build_parser() in cli.py.
_MENU_SECTIONS = [
    ("Run", [
        ("run <file.json> [--monitor] [--fresh]", "Run a DAG pipeline from a task JSON file"),
        ("demo [--monitor]", "Run the built-in 11-task DAG demo (GitHub + HuggingFace)"),
        ("llm --provider <name>", "Run the real multi-agent LLM pipeline (mock/gemini/openai/groq/...)"),
    ]),
    ("Agents", [
        ("agents", "Show Agents running RIGHT NOW (real-time)"),
        ("agents add <file.py> [--persist]", "Plug in your own BaseWorker subclass (session-only, or permanent)"),
        ("agents persisted", "List agents saved with 'agents add --persist'"),
        ("agents reset [--persist]", "Drop custom agents (add --persist to also forget saved ones)"),
    ]),
    ("Observe", [
        ("status [--watch]", "View / watch system telemetry (RAM, CPU, battery, temperature)"),
        ("history", "Show summaries of runs already completed in this session"),
        ("watch", "Watch ANOTHER agentic-or process's live run, from this terminal"),
        ("ui [--port N] [--no-browser]", "Open a lightweight local web dashboard (browser-based watch)"),
    ]),
    ("Shell", [
        ("clear", "Clear the screen"),
        ("help", "Show this menu"),
        ("exit / quit", "Exit the AgentOR shell"),
    ]),
]


def _print_menu() -> None:
    en = theme.color_enabled(sys.stdout)
    cmd_w = max(len(cmd) for _, rows in _MENU_SECTIONS for cmd, _ in rows) + 1
    width = cmd_w + 44
    print(theme.rule(width, enabled=en))
    print(theme.paint("  Commands  (type directly, with or without a leading '/')", theme.BOLD, enabled=en))
    for section, rows in _MENU_SECTIONS:
        print(theme.rule(width, enabled=en))
        print(theme.paint(f"  {section}", theme.DIM, enabled=en))
        for cmd, desc in rows:
            print(f"  {theme.paint(theme.pad(cmd, cmd_w), theme.CYAN, enabled=en)} {desc}")
    print(theme.rule(width, enabled=en))


def _prompt(color: bool) -> str:
    if color:
        return f"{theme.BOLD}{theme.CYAN}AgentOR{theme.RESET} › "
    return "AgentOR > "


def run_repl(parser: argparse.ArgumentParser) -> None:
    """Persistent interactive shell. Reuses `parser` for every line typed."""
    color = theme.color_enabled(sys.stdout)
    print(f"{theme.DIM}Type help to see the menu, exit to quit (no quotes needed).{theme.RESET}\n" if color
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
            # \x1b[H\x1b[2J alone only erases the visible viewport - most
            # modern terminals (iTerm2, GNOME Terminal, Windows Terminal,
            # VS Code, kitty, ...) respond to that by shoving the old
            # content into scrollback rather than actually discarding it,
            # so it visually looks like everything just scrolled up. \x1b[3J
            # (xterm's "erase scrollback" extension, widely supported) is
            # what makes this an actual clear, matching what `clear`/Ctrl+L
            # does in a modern shell.
            sys.stdout.write("\x1b[H\x1b[2J\x1b[3J")
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
