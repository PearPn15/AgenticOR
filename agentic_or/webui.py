"""
`agentic-or ui` - lightweight LOCAL web dashboard, a browser-based view of
`agentic-or watch`. NOT a new monitoring mechanism: reads the exact same
shared status file (~/.agentic_or/status.json, see
agentic_or/telemetry/live_monitor.py's STATUS_FILE_PATH) that `watch`/
`LiveMonitor` already write to. Pure Python stdlib (`http.server`) - zero
new dependencies, nothing to `pip install`, nothing bundled/compiled (no
Tauri/Electron/Node toolchain).

Read-only, same as `watch`: the page only polls and renders, it never sends
a command back to any Orchestrator - consistent with "AgenticOR observes,
it doesn't gate" used throughout this project. Binds to 127.0.0.1 only
(never 0.0.0.0) - this is a local dev dashboard, not a service meant to be
reachable from the network.
"""

from __future__ import annotations

import http.server
import json
import socketserver
import threading
import webbrowser
from typing import Optional

from agentic_or.telemetry.live_monitor import STATUS_FILE_PATH

_HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AgentOR</title>
<style>
  :root {
    --bg: #0b0d12; --panel: #12151c; --border: #232733; --text: #e6e8ee;
    --dim: #8b93a7; --accent: #5fd0ff; --good: #34d399; --warn: #fbbf24; --bad: #f87171;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--text);
    font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    padding: 20px;
  }
  h1 { font-size: 16px; margin: 0 0 4px; letter-spacing: .04em; }
  h1 .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 8px; }
  .sub { color: var(--dim); font-size: 12px; margin-bottom: 16px; }
  .banner { padding: 10px 14px; border-radius: 8px; margin-bottom: 16px; font-size: 13px; }
  .banner.crashed { background: #3a1414; color: var(--bad); border: 1px solid #5a1f1f; }
  .banner.ended { background: #1a2233; color: var(--dim); border: 1px solid var(--border); }
  .banner.active { background: #0f2a22; color: var(--good); border: 1px solid #1c4a3a; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 16px; }
  .card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
  .card .label { color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 6px; }
  .card .value { font-size: 20px; font-weight: 600; }
  .bar { height: 6px; border-radius: 3px; background: #1c2029; overflow: hidden; margin-top: 8px; }
  .bar > div { height: 100%; background: var(--accent); border-radius: 3px; transition: width .3s; }
  .section { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; margin-bottom: 16px; }
  .section .title { color: var(--dim); font-size: 11px; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 10px; }
  .action { color: var(--accent); font-size: 13px; }
  .tree { font-size: 13px; }
  .node { padding: 6px 8px; border-radius: 6px; margin: 2px 0; display: flex; align-items: center; gap: 8px; }
  .node .status-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
  .node .agent-id { font-weight: 600; }
  .node .type { color: var(--dim); font-size: 11px; padding: 1px 6px; border: 1px solid var(--border); border-radius: 4px; }
  .node .task { color: var(--dim); font-size: 12px; }
  .node .measured { color: var(--warn); font-size: 11px; }
  .empty { color: var(--dim); text-align: center; padding: 24px; }
  code { color: var(--accent); }
</style>
</head>
<body>
  <h1><span class="dot" id="conn-dot" style="background:var(--dim)"></span>AgentOR Dashboard</h1>
  <div class="sub" id="sub">connecting...</div>
  <div id="banner"></div>
  <div class="grid" id="metrics"></div>
  <div class="section"><div class="title">Self-Healing</div><div id="healing" class="sub"></div></div>
  <div class="section"><div class="title">Agent Tree</div><div class="tree" id="tree"><div class="empty">Waiting for data...</div></div></div>
  <div class="section" id="claude-section" hidden><div class="title">Claude Code (via agentic-or guard)</div><div class="tree" id="claude-tree"></div></div>

<script>
function bar(ratio) {
  const pct = Math.max(0, Math.min(100, ratio * 100));
  return `<div class="bar"><div style="width:${pct}%"></div></div>`;
}

function statusColor(s) { return s === "RUNNING" ? "var(--good)" : "var(--dim)"; }

function renderNode(node, depth) {
  const indent = depth * 18;
  let html = `<div class="node" style="margin-left:${indent}px">
    <span class="status-dot" style="background:${statusColor(node.status)}"></span>
    <span class="agent-id">${node.agent_id}</span>
    <span class="type">${node.type}</span>
    <span class="task">task=${node.current_task || "-"}</span>`;
  if (node.measured_resources && node.measured_resources.samples > 0) {
    const m = node.measured_resources;
    html += `<span class="measured">measured: ${m.peak_ram_mb.toFixed(1)}MB, ${m.avg_cpu_percent.toFixed(1)}% CPU</span>`;
  }
  html += `</div>`;
  for (const child of (node.children || [])) html += renderNode(child, depth + 1);
  return html;
}

async function poll() {
  const dot = document.getElementById("conn-dot");
  const sub = document.getElementById("sub");
  try {
    const res = await fetch("/api/status", {cache: "no-store"});
    const d = await res.json();

    if (!d.time) {
      dot.style.background = "var(--dim)";
      sub.textContent = "No AgentOR process has written a status file yet - run something with --monitor, or main_agent.py.";
      document.getElementById("banner").innerHTML = "";
      document.getElementById("metrics").innerHTML = "";
      document.getElementById("tree").innerHTML = '<div class="empty">Nothing to show yet</div>';
      return;
    }

    dot.style.background = "var(--good)";
    sub.textContent = `Last update: ${d.time}  ·  Profile: ${d.profile || "?"}`;

    const banner = document.getElementById("banner");
    if (d.session_active === false && d.session_error) {
      banner.innerHTML = `<div class="banner crashed">❌ SESSION CRASHED: ${d.session_error}</div>`;
    } else if (d.session_active === false) {
      banner.innerHTML = `<div class="banner ended">Session ended (exited normally)</div>`;
    } else if (d.monitor_active) {
      banner.innerHTML = `<div class="banner active">● Live${d.current_action ? " — " + d.current_action : ""}</div>`;
    } else {
      banner.innerHTML = `<div class="banner ended">Idle between actions${d.current_action ? " — " + d.current_action : ""}</div>`;
    }

    const t = d.tasks || {};
    document.getElementById("metrics").innerHTML = `
      <div class="card"><div class="label">RAM free</div><div class="value">${(d.ram_free_mb||0).toFixed(0)} MB</div>${bar(d.ram_free_ratio||0)}</div>
      <div class="card"><div class="label">CPU</div><div class="value">${(d.cpu_percent||0).toFixed(1)}%</div>${bar((d.cpu_percent||0)/100)}</div>
      <div class="card"><div class="label">Battery</div><div class="value">${(d.battery_percent||0).toFixed(0)}%</div><div class="sub">${d.is_charging ? "charging" : "on battery"}</div></div>
      <div class="card"><div class="label">Temp</div><div class="value">${(d.cpu_temperature_c||0).toFixed(1)}°C</div></div>
      <div class="card"><div class="label">Tasks</div><div class="value">${t.completed||0}/${t.total||0}</div><div class="sub">running=${t.running||0} pending=${t.pending||0} failed=${t.failed||0}</div></div>
    `;

    const h = d.self_healing || {};
    document.getElementById("healing").textContent =
      `circuit-broken domains: ${(h.circuit_broken_domains||[]).length}   captcha pending: ${h.captcha_pending||0}   locked sessions: ${h.locked_sessions||0}`;

    const tree = d.agent_tree || [];
    document.getElementById("tree").innerHTML = tree.length
      ? tree.map(n => renderNode(n, 0)).join("")
      : '<div class="empty">No agents registered</div>';

    // Written by a SEPARATE process (`agentic-or guard`) reporting Claude
    // Code's own tool/subagent activity via its hooks - purely displayed
    // here, this page never sends anything back to it (read-only, same as
    // the rest of this dashboard).
    const claudeSection = document.getElementById("claude-section");
    const claudeTree = d.claude_code_agents || [];
    claudeSection.hidden = claudeTree.length === 0;
    if (claudeTree.length) {
      document.getElementById("claude-tree").innerHTML = claudeTree.map(n => renderNode(n, 0)).join("");
    }

  } catch (e) {
    dot.style.background = "var(--bad)";
    sub.textContent = "Lost connection to local AgentOR UI server.";
  }
}

poll();
setInterval(poll, 1000);
</script>
</body>
</html>
"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet: don't spam the terminal with HTTP access logs
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, _HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send_status()
        else:
            self._send(404, b"not found", "text/plain")

    def _send_status(self):
        body = b"{}"
        if STATUS_FILE_PATH.exists():
            try:
                body = STATUS_FILE_PATH.read_bytes()
            except OSError:
                pass
        self._send(200, body, "application/json")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


class _ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True


def run_ui(port: int = 8420, open_browser: bool = True) -> None:
    """Blocking - runs until Ctrl+C. See `agentic-or ui`."""
    server: Optional[_ReusableTCPServer] = None
    try:
        server = _ReusableTCPServer(("127.0.0.1", port), _Handler)
    except OSError as e:
        print(f"❌ Could not start the UI server on 127.0.0.1:{port}: {e}")
        print(f"   Try a different port: agentic-or ui --port {port + 1}")
        return

    url = f"http://127.0.0.1:{port}"
    print(f"🖥️  AgentOR UI: {url}  (reads ~/.agentic_or/status.json - same data as `watch`)")
    print("   Ctrl+C to stop.")
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
