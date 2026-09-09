"""
Startup banner for the `agentic-or` CLI - purely cosmetic, stdlib-only
(no new dependency), respects NO_COLOR / non-tty output / AGENTOR_NO_BANNER.
"""
from __future__ import annotations

import os
import sys

from agentic_or.theme import BOLD, CYAN, DIM, RESET, color_enabled

_LOGO = r"""
 █████╗  ██████╗ ███████╗███╗   ██╗████████╗ ██████╗ ██████╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██╔═══██╗██╔══██╗
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║   ██║██████╔╝
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║   ██║██╔══██╗
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ╚██████╔╝██║  ██║
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝
""".strip("\n")

_TAGLINE = "OS-Native Multi-Agent Orchestrator"
_SUBLINE = "C++20 ALNS Engine + Python Async Workers"
_VERSION = os.environ.get("AGENTOR_VERSION", "0.1.0")


def print_banner(stream=None) -> None:
    """
    Print the AgentOR banner once. Safe/quiet by default:
    - `AGENTOR_NO_BANNER` skips it entirely (scripts, CI, piping).
    - `AGENTOR_COMPACT_BANNER` prints a single line instead of the ASCII
      logo - useful once you've seen the logo a hundred times and just want
      the prompt.
    """
    if os.environ.get("AGENTOR_NO_BANNER"):
        return

    stream = stream or sys.stdout
    color = color_enabled(stream)

    if os.environ.get("AGENTOR_COMPACT_BANNER"):
        line = f"AgentOR v{_VERSION} — {_TAGLINE}"
        stream.write(f"{BOLD}{CYAN}{line}{RESET}\n" if color else f"{line}\n")
        stream.flush()
        return

    if color:
        stream.write(f"{CYAN}{_LOGO}{RESET}\n")
        stream.write(f"{DIM}  {_TAGLINE}  ·  {_SUBLINE}  ·  v{_VERSION}{RESET}\n\n")
    else:
        stream.write(f"{_LOGO}\n")
        stream.write(f"  {_TAGLINE}  ·  {_SUBLINE}  ·  v{_VERSION}\n\n")
    stream.flush()
