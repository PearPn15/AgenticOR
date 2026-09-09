# `agentic-or guard` — AgenticOR as a resource governor for Claude Code

Everywhere else in AgenticOR (`Orchestrator`, `LiveMonitor`, `watch`, `ui`)
is deliberately **read-only** — it observes, it never gates (see
[`agentic_or/webui.py`](../agentic_or/webui.py)'s docstring). `guard` is the
one explicit exception: a small daemon that lets Claude Code ask AgenticOR
*"is it safe to run this tool right now?"* before it actually runs it, using
the exact same `OOMGuard`/`ThermalBatteryGuard` real-machine thresholds that
already govern AgenticOR's own workers, **plus** a real concurrency cap (how
many heavy tool calls / subagents are already running, machine-wide) — and
reports Claude Code's own tool/subagent activity into the same dashboard
`watch`/`ui` already show.

## The one rule this whole feature is built around

**A hook that can't reach the daemon must always fail open (allow).** Not
running `agentic-or guard`, the daemon crashing, or a slow/malformed reply
must all be indistinguishable, from Claude Code's point of view, from this
integration not being installed at all — never a hang, never a wrongly
blocked tool call. See [`agentic_or/guard/hook_client.py`](../agentic_or/guard/hook_client.py)'s
docstring for exactly how that's enforced (short timeouts, broad
exception handling, default-allow on any doubt).

A small, fixed whitelist of tools (`Read`, `Grep`, `Glob`, `TodoWrite`,
`AskUserQuestion`, `ExitPlanMode`, `EnterPlanMode` — see
[`agentic_or/guard/decision.py`](../agentic_or/guard/decision.py)) is
**never** denied, no matter how bad the machine's resource state is, so
Claude Code always keeps enough capability to diagnose a problem and talk
to you about it, even under real memory/thermal pressure. Only `Bash`,
`WebFetch`, `WebSearch` (gated together, "heavy") and `Task` (subagent
spawns, gated **separately** with its own cap) are ever gated.

## Three-way decision, not just allow/deny

- **`allow`** — runs normally.
- **`ask`** — Claude Code prompts you for a permission decision, same as
  any other permission prompt. Used for *policy* throttles — concurrency
  caps, thermal/battery caution — not immediate danger, just something
  worth your call. You can say "go ahead anyway."
- **`deny`** — hard block, no prompt. Reserved for actual crash risk (RAM
  genuinely low/critical), where waiting on an answer isn't safe.

| Condition | Decision |
|---|---|
| RAM below AgenticOR's critical/warning threshold | `deny` (hard) |
| Thermal/battery guard forced `ECO_SILENT` | `ask` |
| Already at the concurrency cap for the current profile | `ask` |
| Everything else | `allow` |

Concurrency caps (heavy tools / subagents, machine-wide across every
session this daemon knows about) scale with the same `ExecutionProfile`
AgenticOR's own scheduler already uses:

| Profile | Max concurrent heavy tools | Max concurrent subagents |
|---|---|---|
| `ECO_SILENT` | 1 | 1 |
| `BALANCED` | 3 | 2 |
| `TURBO_SPEED` | 6 | 4 |

## 1. The daemon auto-starts — you shouldn't need to think about it

A `SessionStart` hook checks `/health` and, if the daemon isn't reachable,
launches `agentic-or guard` detached in the background automatically. You
can still manage it by hand:

```bash
agentic-or guard              # foreground, its own terminal - Ctrl+C to stop
agentic-or guard status       # pid, uptime, known session count, current profile - or "not running"
agentic-or guard stop         # graceful shutdown from another terminal (POST /shutdown)
agentic-or guard log [-n 20]  # last N logged ask/deny decisions (never "allow" - that'd be pure noise)
```

It binds to `127.0.0.1` only — never reachable from the network.

## 2. Hooks are wired globally

`~/.claude/settings.json` has the full hook set (`SessionStart`,
`SessionEnd`, `PreToolUse`, `PostToolUse`, `PostToolUseFailure`,
`SubagentStart`, `SubagentStop`) pointing at
`/home/vc/AgenticOR/.venv/bin/python3 -m agentic_or.guard.hook_client <event>`
— an absolute interpreter path, so it works regardless of which directory a
Claude Code session is started in, not just inside this repo. This means
**every** Claude Code session on this machine now reports to and is gated
by this one daemon. (It was validated project-scoped first, in this repo's
own now-removed `.claude/settings.json`, before being widened here.)

If you ever want to scope it back down to one repo, move the same `hooks`
block into that repo's own `.claude/settings.json` and remove it from the
global one — don't run both at once, or every tool call fires the hook
twice for no benefit.

## 3. Session lifecycle - the dashboard doesn't accumulate ghosts

- `SessionEnd` tells the daemon to drop that session entirely.
- As a fallback (a process killed before it could fire `SessionEnd`), any
  session with no activity at all for **1 hour** is dropped automatically.
- Within a session, an individual tool call that finished (or got
  denied/asked and never actually ran) is dropped after **5 minutes** of
  sitting idle - only genuinely `RUNNING` entries are exempt from this.

## 4. Verify it

```bash
curl http://127.0.0.1:8422/health          # {"ok": true}
curl http://127.0.0.1:8422/status          # pid/uptime/session_count/profile
agentic-or watch                           # a "CLAUDE CODE (via agentic-or guard)" section
                                            # appears once a hook has fired at least once
agentic-or guard log                       # anything it's actually throttled so far
```

## 5. Kill switch

`agentic-or guard stop` (or Ctrl+C in its terminal) at any time — every
hook call then fails open within ~1 second (the client's timeout) and
Claude Code behaves exactly as if `guard` were never configured. Since
`SessionStart` auto-restarts it on the next new session, remove the
`hooks` block from `~/.claude/settings.json` for a durable stop.

## What it does NOT do

- It does not see or gate Claude **Desktop** — Desktop has no hooks system.
  Its MCP server activity is only visible via its own log files
  (`~/Library/Logs/Claude/mcp*.log` on macOS, `%APPDATA%\Claude\logs\` on
  Windows), which `guard` does not read.
- It does not inspect tool *content* (command text, file paths) — only the
  tool *name*, live machine state, and how many are already running. It
  cannot block "a specific dangerous command", only "any `Bash` call, full
  stop, while the machine/concurrency is in a bad state."
- It never touches AgenticOR's own `Orchestrator`/task scheduling — a
  completely separate code path that keeps behaving exactly as before.
- An `ask` that a human then approves is **not** reported back to `guard`
  by Claude Code (the hook contract doesn't call back after an `ask`
  resolves) - if it goes on to run, the eventual `PostToolUse` event still
  finds and updates that same dashboard entry, but the daemon never learns
  which way the human actually decided.
