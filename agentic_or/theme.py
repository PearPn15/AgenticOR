"""
Shared presentation helpers for every `agentic-or` screen (banner, REPL,
`status`/`watch`, agent tables, history) - stdlib-only, zero new dependency,
by design: this project already tracks the machine's own RAM/CPU budget for
its agents, so the CLI's own footprint stays a `print()` + ANSI escape codes,
never a rendering framework.

Centralizing color choices and the box-drawing table/panel helpers here means
every screen renders with the *same* palette and the *same* alignment logic,
instead of each command hand-rolling its own `f"{x:<10}"` padding and picking
its own colors (which is how the CLI drifted inconsistent in the first
place). Every function degrades to plain text automatically when NO_COLOR is
set or stdout isn't a tty - piping `agentic-or status` never emits stray
escape codes.
"""
from __future__ import annotations

import os
import re
import sys
from typing import List, Sequence, TextIO

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"

CYAN = "\x1b[38;5;51m"
GREEN = "\x1b[38;5;42m"
YELLOW = "\x1b[38;5;220m"
RED = "\x1b[38;5;203m"
GRAY = "\x1b[38;5;244m"
MAGENTA = "\x1b[38;5;177m"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def color_enabled(stream: TextIO = None) -> bool:
    """Same rule everywhere: NO_COLOR wins, AGENTOR_FORCE_COLOR forces on,
    otherwise color only when actually writing to a real terminal."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("AGENTOR_FORCE_COLOR"):
        return True
    return bool(getattr(stream, "isatty", lambda: False)())


def paint(text: str, *codes: str, enabled: bool = True) -> str:
    """Wrap `text` in ANSI `codes`; a no-op when `enabled` is False so plain
    output (piped, NO_COLOR) never carries stray escape sequences."""
    if not enabled or not codes:
        return text
    return f"{''.join(codes)}{text}{RESET}"


def visible_len(text: str) -> int:
    """len(), ignoring ANSI escapes - needed so column widths line up even
    when a cell is already colored (e.g. a red RAM bar next to plain text)."""
    return len(_ANSI_RE.sub("", text))


def pad(text: str, width: int) -> str:
    """Left-pad-to-width that accounts for embedded ANSI codes - plain
    f"{text:<{width}}" over-pads a colored string by the length of its
    escape codes. Truncates (with an ellipsis) plain text that overflows
    `width`, so dynamic content (a long list of blocked domains, say) can
    never push a panel's right border out of alignment - colored text is
    left alone since slicing mid-escape-code would corrupt it, so callers
    that colorize dynamic content should keep it short themselves."""
    vis = visible_len(text)
    if vis > width and width > 1 and not _ANSI_RE.search(text):
        text = text[: width - 1] + "…"
        vis = width
    return text + " " * max(0, width - vis)


def level_color(ratio: float, invert: bool = False) -> str:
    """Green/yellow/red by how healthy `ratio` (0..1) is. `invert=True` for
    metrics where LOW is good (CPU load, temperature-as-ratio) instead of
    HIGH (RAM free, battery)."""
    r = (1.0 - ratio) if invert else ratio
    if r >= 0.5:
        return GREEN
    if r >= 0.2:
        return YELLOW
    return RED


def bar(ratio: float, width: int = 18, enabled: bool = True, invert: bool = False) -> str:
    """A filled/empty block bar, colored by how healthy the ratio is."""
    ratio = max(0.0, min(1.0, ratio))
    filled = int(round(ratio * width))
    body = "█" * filled + "░" * (width - filled)
    return paint(body, level_color(ratio, invert=invert), enabled=enabled)


def status_dot(active: bool, enabled: bool = True) -> str:
    return paint("●", GREEN, enabled=enabled) + " running" if active else paint("○", GRAY, enabled=enabled) + " idle"


def rule(width: int = 66, char: str = "─", enabled: bool = True) -> str:
    return paint(char * width, GRAY, enabled=enabled)


def header(title: str, right: str = "", width: int = 66, enabled: bool = True) -> List[str]:
    """A titled panel top border: `┌─ TITLE ──...──┐` with an optional
    right-aligned tag (timestamp, profile, ...) before the border closes."""
    left = f"─ {title} "
    fill_len = max(1, width - visible_len(left) - visible_len(right) - 3)
    top = "┌" + left + ("─" * fill_len) + (f" {right} " if right else "") + "┐"
    return [paint(top, GRAY, enabled=enabled)]


def footer(width: int = 66, enabled: bool = True) -> str:
    return paint("└" + "─" * (width - 2) + "┘", GRAY, enabled=enabled)


def divider(width: int = 66, enabled: bool = True) -> str:
    """An internal `├─...─┤` row separating sections within one panel -
    unlike `rule()`, this keeps the panel's left/right border unbroken."""
    return paint("├" + "─" * (width - 2) + "┤", GRAY, enabled=enabled)


def panel_line(text: str, width: int = 66, enabled: bool = True) -> str:
    """One `│ ... │` row inside a panel, padded to `width` (ANSI-aware)."""
    border = paint("│", GRAY, enabled=enabled)
    inner_width = width - 4
    return f"{border} {pad(text, inner_width)} {border}"


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], enabled: bool = True) -> List[str]:
    """Render a box-drawing table. Cells may already contain ANSI codes -
    column widths are computed on visible length, not raw string length."""
    cols = len(headers)
    widths = [visible_len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], visible_len(cell))
    widths = [w + 2 for w in widths]  # 1 space padding each side

    def rule_line(left: str, mid: str, right: str) -> str:
        return paint(left + mid.join("─" * w for w in widths) + right, GRAY, enabled=enabled)

    def row_line(cells: Sequence[str]) -> str:
        border = paint("│", GRAY, enabled=enabled)
        parts = [f" {pad(cells[i], widths[i] - 2)} " for i in range(cols)]
        return border + border.join(parts) + border

    lines = [rule_line("┌", "┬", "┐")]
    lines.append(row_line([paint(h, BOLD, enabled=enabled) for h in headers]))
    lines.append(rule_line("├", "┼", "┤"))
    for row in rows:
        lines.append(row_line(row))
    lines.append(rule_line("└", "┴", "┘"))
    return lines
