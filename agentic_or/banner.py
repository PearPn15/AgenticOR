"""
Startup banner for the `agentic-or` CLI - purely cosmetic, stdlib-only
(no new dependency), respects NO_COLOR / non-tty output / AGENTOR_NO_BANNER.
"""
from __future__ import annotations

import os
import sys

_LOGO = r"""
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗ ██████╗ ██████╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗██╔══██╗
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║   ██║██████╔╝
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║   ██║██╔══██╗
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ╚██████╔╝██║  ██║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
""".strip("\n")

_TAGLINE = "OS-Native Multi-Agent Orchestrator  ·  C++20 ALNS Engine + Python Async Workers"

_CYAN = "\x1b[38;5;51m"
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


def _color_enabled(stream) -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("AGENTOR_FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def print_banner(stream=None) -> None:
    """Print the AgentOR ASCII banner + tagline once. Safe/quiet by default."""
    if os.environ.get("AGENTOR_NO_BANNER"):
        return

    stream = stream or sys.stdout
    if _color_enabled(stream):
        stream.write(f"{_CYAN}{_LOGO}{_RESET}\n")
        stream.write(f"{_DIM}  {_TAGLINE}{_RESET}\n\n")
    else:
        stream.write(f"{_LOGO}\n")
        stream.write(f"  {_TAGLINE}\n\n")
    stream.flush()
