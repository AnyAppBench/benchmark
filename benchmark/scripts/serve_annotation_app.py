#!/usr/bin/env python3
"""Serve a local web UI for CATBench human annotation.

The app reads the sample_manifest.jsonl produced by
sample_episodes_for_annotation.py and writes annotations.jsonl in the same
schema as annotate_judge_cases.py, so validate_judge_vs_human.py can be used
unchanged.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import datetime as dt
import fcntl
import hmac
import json
import mimetypes
import os
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs
from urllib.parse import quote
from urllib.parse import unquote
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from annotate_judge_cases import _save_episode_artifacts  # noqa: E402
from classify_catbench_failures import DEFAULT_FAILURE_MODES  # noqa: E402
from classify_catbench_failures import _read_pkl_gz  # noqa: E402


CONFIDENCES = ("low", "medium", "high")
AUTH_REALM = "CATBench Annotation"

# API responses use explicit allowlists.  A manifest often contains source
# JSONL paths, judge configuration hashes, run conditions, and both judges'
# rationales; none of those should become public merely because a new field was
# added upstream.
_BLIND_PICK_FIELDS = (
    "category",
    "app_id",
    "app_name",
    "task_template",
    "goal",
    "is_successful",
)
_UNBLINDED_PICK_FIELDS = _BLIND_PICK_FIELDS + (
    "episode_id",
    "model_name",
    "condition",
    "pool",
    "pkl_path",
    "judge_label",
    "judge_confidence",
    "judge_rationale",
    "judge_verdict",
)
_PUBLIC_HUMAN_LABEL_FIELDS = (
    "primary_failure_mode",
    "planning_score",
    "grounding_score",
    "confidence",
    "rationale",
)


class BasicAuth:
  """In-memory Basic-auth credentials with a deliberately redacted repr."""

  __slots__ = ("_username", "_password")

  def __init__(self, username: str, password: str) -> None:
    self._username = username.encode("utf-8")
    self._password = password.encode("utf-8")

  def __repr__(self) -> str:
    return "BasicAuth(<redacted>)"

  def accepts(self, authorization: str | None) -> bool:
    """Validate an Authorization header without short-circuiting secrets."""
    supplied_username = b""
    supplied_password = b""
    if authorization:
      try:
        scheme, token = authorization.strip().split(None, 1)
        if scheme.lower() == "basic":
          decoded = base64.b64decode(token, validate=True)
          supplied_username, separator, supplied_password = decoded.partition(b":")
          if not separator:
            supplied_username = b""
            supplied_password = b""
      except (ValueError, binascii.Error):
        pass

    # Bitwise-and is intentional: compare both values even if the username is
    # wrong, and use the standard constant-time primitive for both secrets.
    username_matches = hmac.compare_digest(supplied_username, self._username)
    password_matches = hmac.compare_digest(supplied_password, self._password)
    return bool(username_matches & password_matches)


def _load_basic_auth(
    username: str | None,
    password_file: str | None,
    password_env: str | None,
) -> BasicAuth | None:
  """Load optional credentials without accepting a plaintext password flag."""
  sources = int(bool(password_file)) + int(bool(password_env))
  if username is None and sources == 0:
    return None
  if not username:
    raise ValueError("Basic authentication requires --auth_username.")
  if sources != 1:
    raise ValueError(
        "Basic authentication requires exactly one of --auth_password_file "
        "or --auth_password_env."
    )

  if password_file:
    # Remove conventional line endings written by secret-file tooling, while
    # preserving any other leading or trailing password characters.
    password = Path(password_file).expanduser().read_text(encoding="utf-8")
    password = password.rstrip("\r\n")
  else:
    assert password_env is not None
    password = os.environ.get(password_env)
    if password is None:
      raise ValueError("The configured authentication password environment variable is unset.")
  if not password:
    raise ValueError("The authentication password must not be empty.")
  return BasicAuth(username, password)


def _is_successful(value: Any) -> bool:
  try:
    return float(value or 0.0) >= 0.5
  except (TypeError, ValueError):
    return False


def _public_pick(pick: dict[str, Any], blind: bool) -> dict[str, Any]:
  """Return only fields that the annotation client is allowed to receive."""
  fields = _BLIND_PICK_FIELDS if blind else _UNBLINDED_PICK_FIELDS
  return {key: pick.get(key) for key in fields if key in pick}


def _public_annotation(
    annotation: dict[str, Any] | None,
) -> dict[str, Any] | None:
  """Return enough of a saved annotation to prefill, without audit metadata."""
  if not annotation:
    return None
  label = annotation.get("human_label")
  if not isinstance(label, dict):
    label = {}
  return {
      "human_label": {
          key: label.get(key)
          for key in _PUBLIC_HUMAN_LABEL_FIELDS
          if key in label
      },
      "suspect_validator": bool(annotation.get("suspect_validator")),
  }


def _validated_human_label(payload: dict[str, Any]) -> dict[str, Any]:
  """Validate explicit, non-defaulted form values from the browser."""
  mode = payload.get("primary_failure_mode")
  if not isinstance(mode, str) or mode not in DEFAULT_FAILURE_MODES:
    raise ValueError(f"Invalid failure mode: {mode!r}")

  confidence = payload.get("confidence")
  if not isinstance(confidence, str) or confidence not in CONFIDENCES:
    raise ValueError(f"Invalid confidence: {confidence!r}")

  def required_score(name: str) -> int:
    if name not in payload:
      raise ValueError(f"Missing required score: {name}")
    value = payload[name]
    if isinstance(value, bool):
      raise ValueError(f"Invalid {name}: {value!r}; expected an integer 0-3")
    if isinstance(value, int):
      parsed = value
    elif isinstance(value, str) and value in {"0", "1", "2", "3"}:
      parsed = int(value)
    else:
      raise ValueError(f"Invalid {name}: {value!r}; expected an integer 0-3")
    if parsed < 0 or parsed > 3:
      raise ValueError(f"Invalid {name}: {value!r}; expected an integer 0-3")
    return parsed

  rationale = payload.get("rationale")
  if not isinstance(rationale, str) or not rationale.strip():
    raise ValueError("Rationale is required.")

  return {
      "primary_failure_mode": mode,
      "planning_score": required_score("planning_score"),
      "grounding_score": required_score("grounding_score"),
      "confidence": confidence,
      "rationale": rationale.strip(),
  }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CATBench Annotation</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1b1f24;
      --muted: #69717d;
      --line: #d9dee7;
      --accent: #1264a3;
      --accent-soft: #e8f2fb;
      --danger: #9a3412;
      --ok: #166534;
      --shadow: 0 1px 2px rgba(18, 25, 38, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app {
      display: grid;
      grid-template-columns: 280px minmax(420px, 1fr) 340px;
      height: 100vh;
      min-height: 680px;
    }
    aside, main, .form-panel {
      min-height: 0;
      overflow: hidden;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #fbfcfd;
      display: grid;
      grid-template-rows: auto auto 1fr;
    }
    header {
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }
    h1 {
      margin: 0;
      font-size: 17px;
      letter-spacing: 0;
    }
    .meta {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
    }
    .queue-controls {
      display: grid;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    .search {
      width: 100%;
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 0 10px;
      background: white;
    }
    .queue {
      overflow: auto;
      padding: 8px;
    }
    .case-row {
      width: 100%;
      border: 1px solid transparent;
      border-radius: 7px;
      background: transparent;
      color: var(--text);
      display: grid;
      gap: 3px;
      text-align: left;
      padding: 9px;
      cursor: pointer;
    }
    .case-row:hover { background: #eef2f7; }
    .case-row.active {
      background: var(--accent-soft);
      border-color: #b9d7ef;
    }
    .case-title {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      font-weight: 650;
      font-size: 13px;
    }
    .case-sub {
      color: var(--muted);
      font-size: 12px;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
    }
    .dot {
      flex: 0 0 auto;
      width: 8px;
      height: 8px;
      border-radius: 999px;
      margin-top: 5px;
      background: #c0c7d1;
    }
    .dot.done { background: var(--ok); }
    main {
      position: relative;
      display: grid;
      grid-template-rows: auto minmax(260px, 1fr) minmax(220px, 0.8fr);
      gap: 12px;
      padding: 12px;
    }
    .case-header, .viewer, .trace, .form-panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .case-header {
      padding: 14px 16px;
      display: grid;
      gap: 6px;
    }
    .header-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: start;
      gap: 12px;
    }
    .goal {
      font-size: 16px;
      font-weight: 650;
      min-width: 0;
    }
    .chips {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      color: #394150;
      background: #fafbfc;
      font-size: 12px;
    }
    .viewer {
      min-height: 0;
      height: 100%;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 150px;
      overflow: hidden;
    }
    .image-stage {
      min-height: 0;
      min-width: 0;
      display: grid;
      place-items: start center;
      padding: 10px;
      background: #111827;
      overflow: auto;
    }
    .image-stage img {
      width: auto;
      height: auto;
      max-width: 100%;
      max-height: none;
      object-fit: contain;
      border-radius: 4px;
      background: #0b0f17;
    }
    .thumbs {
      border-left: 1px solid var(--line);
      overflow: auto;
      padding: 8px;
      display: grid;
      align-content: start;
      gap: 8px;
    }
    .thumb {
      border: 2px solid transparent;
      border-radius: 7px;
      padding: 3px;
      background: #f8fafc;
      cursor: pointer;
    }
    .thumb.active { border-color: var(--accent); }
    .thumb img {
      display: block;
      width: 100%;
      aspect-ratio: 9 / 16;
      object-fit: cover;
      border-radius: 4px;
      background: #e5e7eb;
    }
    .thumb-label {
      margin-top: 3px;
      color: var(--muted);
      font-size: 11px;
    }
    .trace {
      min-height: 0;
      overflow: auto;
      padding: 14px 16px;
    }
    .trace pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      color: #2f3744;
    }
    .form-panel {
      border-left: 1px solid var(--line);
      background: var(--panel);
      display: grid;
      grid-template-rows: auto 1fr auto;
    }
    .form-head {
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }
    .form-body {
      overflow: auto;
      padding: 16px;
      display: grid;
      gap: 16px;
      align-content: start;
    }
    .field {
      display: grid;
      gap: 7px;
    }
    label, .label {
      color: #303846;
      font-weight: 650;
      font-size: 13px;
    }
    .segmented {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow: hidden;
    }
    .segmented button {
      height: 34px;
      border: 0;
      border-right: 1px solid var(--line);
      background: white;
      cursor: pointer;
      font-weight: 600;
      color: #394150;
    }
    .segmented button:last-child { border-right: 0; }
    .segmented button.selected {
      background: var(--accent);
      color: white;
    }
    select, textarea, input[type="number"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: white;
      color: var(--text);
      padding: 8px 10px;
      font: inherit;
    }
    textarea {
      min-height: 110px;
      resize: vertical;
    }
    .scores {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .actions {
      padding: 12px;
      border-top: 1px solid var(--line);
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    button.primary, button.secondary {
      height: 38px;
      border-radius: 7px;
      border: 1px solid var(--line);
      cursor: pointer;
      font-weight: 700;
      background: white;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }
    button.secondary:hover, button.primary:hover { filter: brightness(0.98); }
    button.iconish {
      height: 32px;
      min-width: 76px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: white;
      color: #394150;
      cursor: pointer;
      font-weight: 700;
    }
    button.iconish.active {
      background: var(--accent-soft);
      border-color: #b9d7ef;
      color: #16476f;
    }
    .status {
      min-height: 18px;
      color: var(--muted);
      font-size: 12px;
    }
    .error { color: var(--danger); }
    .loading {
      height: 100%;
      display: grid;
      place-items: center;
      color: var(--muted);
    }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 230px 1fr; }
      .form-panel {
        position: fixed;
        right: 0;
        top: 0;
        bottom: 0;
        width: min(380px, 100vw);
        z-index: 5;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <header>
        <h1>CATBench Annotation</h1>
        <div id="runMeta" class="meta">Loading...</div>
      </header>
      <div class="queue-controls">
        <input id="search" class="search" placeholder="Filter cases">
        <button id="nextUnlabeled" class="secondary">Next unlabeled</button>
      </div>
      <div id="queue" class="queue"></div>
    </aside>
    <main>
      <section class="case-header">
        <div id="goal" class="goal">Select a case</div>
        <div id="chips" class="chips"></div>
      </section>
      <section class="viewer">
        <div id="imageStage" class="image-stage"><div class="loading">No frame selected</div></div>
        <div id="thumbs" class="thumbs"></div>
      </section>
      <section class="trace"><pre id="summary"></pre></section>
    </main>
    <section class="form-panel">
      <div class="form-head">
        <h1>Human Label</h1>
        <div id="saveStatus" class="status"></div>
      </div>
      <div class="form-body">
        <div class="field">
          <label for="failureMode">Primary failure mode</label>
          <select id="failureMode"></select>
        </div>
        <div class="scores">
          <div class="field">
            <label for="planningScore">Planning score</label>
            <input id="planningScore" type="number" min="0" max="3" step="1" required>
          </div>
          <div class="field">
            <label for="groundingScore">Grounding score</label>
            <input id="groundingScore" type="number" min="0" max="3" step="1" required>
          </div>
        </div>
        <div class="field">
          <div class="label">Confidence</div>
          <div id="confidenceButtons" class="segmented"></div>
        </div>
        <div class="field">
          <label for="rationale">Short rationale</label>
          <textarea id="rationale"></textarea>
        </div>
        <label class="field">
          <span class="label">Validator audit flag</span>
          <span><input id="suspectValidator" type="checkbox"> Screenshots contradict the validator verdict</span>
        </label>
      </div>
      <div class="actions">
        <button id="prevCase" class="secondary">Previous</button>
        <button id="nextCase" class="secondary">Next</button>
        <button id="save" class="primary">Save</button>
        <button id="saveNext" class="primary">Save next</button>
      </div>
    </section>
  </div>
  <script>
    const state = {
      cases: [],
      activeIndex: 0,
      activeCase: null,
      confidence: null,
      failureModes: [],
      filter: "",
    };
    const $ = (id) => document.getElementById(id);

    function setStatus(text, isError = false) {
      const node = $("saveStatus");
      node.textContent = text || "";
      node.className = isError ? "status error" : "status";
    }

    function makeSegment(containerId, values, key) {
      const el = $(containerId);
      el.innerHTML = "";
      for (const value of values) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = value.replaceAll("_", " ");
        button.dataset.value = value;
        button.onclick = () => {
          state[key] = value;
          updateSegments();
        };
        el.appendChild(button);
      }
    }

    function updateSegments() {
      for (const [containerId, key] of [["confidenceButtons", "confidence"]]) {
        for (const button of $(containerId).querySelectorAll("button")) {
          button.classList.toggle("selected", button.dataset.value === state[key]);
        }
      }
    }

    function renderQueue() {
      const queue = $("queue");
      queue.innerHTML = "";
      const term = state.filter.toLowerCase();
      state.cases.forEach((item, index) => {
        const haystack = [item.category, item.app_id, item.app_name, item.task_template, item.goal]
          .filter(Boolean).join(" ").toLowerCase();
        if (term && !haystack.includes(term)) return;
        const button = document.createElement("button");
        button.className = "case-row" + (index === state.activeIndex ? " active" : "");
        button.onclick = () => loadCase(index);
        const title = item.model_name ? `${index + 1}. ${item.model_name}` : `Case ${index + 1}`;
        button.innerHTML = `
          <div class="case-title"><span>${title}</span><span class="dot ${item.done ? "done" : ""}"></span></div>
          <div class="case-sub">${item.category} / ${item.app_id}</div>
          <div class="case-sub">${item.task_template}</div>
        `;
        queue.appendChild(button);
      });
    }

    function renderMeta(item) {
      $("goal").textContent = item.goal || "";
      const chips = $("chips");
      chips.innerHTML = "";
      const values = [
        item.model_name,
        item.category,
        item.app_id,
        item.task_template,
        `validator verdict (read-only): ${Number(item.is_successful) >= 0.5 ? "success" : "failure"}`,
        item.pool ? `pool: ${item.pool}` : null,
      ].filter(Boolean);
      for (const value of values) {
        const chip = document.createElement("span");
        chip.className = "chip";
        chip.textContent = value;
        chips.appendChild(chip);
      }
    }

    function renderFrames(frames) {
      const stage = $("imageStage");
      const thumbs = $("thumbs");
      stage.innerHTML = "";
      thumbs.innerHTML = "";
      if (!frames.length) {
        stage.innerHTML = '<div class="loading">No frames saved for this case</div>';
        return;
      }
      const show = (frame, thumbNode) => {
        stage.innerHTML = "";
        const img = document.createElement("img");
        img.src = frame.url;
        img.alt = `step ${frame.step}`;
        stage.appendChild(img);
        for (const node of thumbs.querySelectorAll(".thumb")) node.classList.remove("active");
        if (thumbNode) thumbNode.classList.add("active");
      };
      frames.forEach((frame, i) => {
        const thumb = document.createElement("button");
        thumb.className = "thumb" + (i === 0 ? " active" : "");
        thumb.innerHTML = `<img src="${frame.url}" alt="step ${frame.step}"><div class="thumb-label">step ${frame.step}</div>`;
        thumb.onclick = () => show(frame, thumb);
        thumbs.appendChild(thumb);
      });
      show(frames[0], thumbs.querySelector(".thumb"));
    }

    function fillForm(annotation) {
      state.confidence = annotation?.human_label?.confidence || null;
      $("failureMode").value = annotation?.human_label?.primary_failure_mode || "";
      $("planningScore").value = annotation?.human_label?.planning_score ?? "";
      $("groundingScore").value = annotation?.human_label?.grounding_score ?? "";
      $("rationale").value = annotation?.human_label?.rationale || "";
      $("suspectValidator").checked = Boolean(annotation?.suspect_validator);
      updateSegments();
    }

    async function loadCase(index) {
      if (index < 0 || index >= state.cases.length) return;
      state.activeIndex = index;
      setStatus("");
      renderQueue();
      $("summary").textContent = "Loading case...";
      $("imageStage").innerHTML = '<div class="loading">Loading frames...</div>';
      $("thumbs").innerHTML = "";
      const res = await fetch(`/api/case?index=${index}`);
      if (!res.ok) {
        setStatus(await res.text(), true);
        return;
      }
      const data = await res.json();
      state.activeCase = data;
      renderMeta(data.pick);
      renderFrames(data.frames || []);
      $("summary").textContent = data.summary_markdown || "";
      fillForm(data.annotation);
    }

    function nextUnlabeledIndex(start = state.activeIndex + 1) {
      for (let offset = 0; offset < state.cases.length; offset++) {
        const index = (start + offset) % state.cases.length;
        if (!state.cases[index].done) return index;
      }
      return state.activeIndex;
    }

    async function saveAnnotation(moveNext) {
      const failureMode = $("failureMode").value;
      const planningRaw = $("planningScore").value;
      const groundingRaw = $("groundingScore").value;
      if (!state.failureModes.includes(failureMode)) {
        setStatus("Select a primary failure mode.", true);
        return;
      }
      if (!/^[0-3]$/.test(planningRaw)) {
        setStatus("Select an explicit planning score from 0 to 3.", true);
        return;
      }
      if (!/^[0-3]$/.test(groundingRaw)) {
        setStatus("Select an explicit grounding score from 0 to 3.", true);
        return;
      }
      if (!["low", "medium", "high"].includes(state.confidence)) {
        setStatus("Select confidence.", true);
        return;
      }
      const rationale = $("rationale").value.trim();
      if (!rationale) {
        setStatus("Rationale is required.", true);
        return;
      }
      const payload = {
        index: state.activeIndex,
        primary_failure_mode: failureMode,
        planning_score: Number(planningRaw),
        grounding_score: Number(groundingRaw),
        confidence: state.confidence,
        rationale,
        suspect_validator: $("suspectValidator").checked,
      };
      const res = await fetch("/api/annotation", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        setStatus(await res.text(), true);
        return;
      }
      const data = await res.json();
      state.cases[state.activeIndex].done = true;
      setStatus(`Saved ${data.saved_at}`);
      renderQueue();
      if (moveNext) loadCase(nextUnlabeledIndex(state.activeIndex + 1));
    }

    async function init() {
      makeSegment("confidenceButtons", ["low", "medium", "high"], "confidence");
      updateSegments();
      const res = await fetch("/api/state");
      const data = await res.json();
      state.cases = data.cases;
      state.failureModes = data.failure_modes;
      $("runMeta").textContent = data.pool
        ? `${data.annotator} | ${data.done_count}/${data.total} done | pool ${data.pool}`
        : `${data.annotator} | ${data.done_count}/${data.total} done`;
      const select = $("failureMode");
      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "Select failure mode";
      placeholder.disabled = true;
      placeholder.selected = true;
      select.appendChild(placeholder);
      for (const mode of data.failure_modes) {
        const option = document.createElement("option");
        option.value = mode;
        option.textContent = mode.replaceAll("_", " ");
        select.appendChild(option);
      }
      $("search").oninput = (event) => {
        state.filter = event.target.value;
        renderQueue();
      };
      $("nextUnlabeled").onclick = () => loadCase(nextUnlabeledIndex(state.activeIndex + 1));
      $("prevCase").onclick = () => loadCase(state.activeIndex - 1);
      $("nextCase").onclick = () => loadCase(state.activeIndex + 1);
      $("save").onclick = () => saveAnnotation(false);
      $("saveNext").onclick = () => saveAnnotation(true);
      renderQueue();
      if (state.cases.length) {
        loadCase(nextUnlabeledIndex(0));
      } else {
        $("summary").textContent = "No cases in this pool.";
      }
    }
    init().catch((err) => setStatus(String(err), true));
  </script>
</body>
</html>
"""


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
  body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
  handler.send_response(status)
  handler.send_header("Content-Type", "application/json; charset=utf-8")
  handler.send_header("Content-Length", str(len(body)))
  handler.end_headers()
  handler.wfile.write(body)


def _text_response(
    handler: BaseHTTPRequestHandler,
    text: str,
    status: int = 200,
    content_type: str = "text/plain; charset=utf-8",
) -> None:
  body = text.encode("utf-8")
  handler.send_response(status)
  handler.send_header("Content-Type", content_type)
  handler.send_header("Content-Length", str(len(body)))
  handler.end_headers()
  handler.wfile.write(body)


def _load_sample_manifest(
    path: Path,
    pool: str,
    allow_successes: bool,
) -> list[dict[str, Any]]:
  picks: list[dict[str, Any]] = []
  success_examples: list[str] = []
  with path.open("r", encoding="utf-8") as handle:
    for line in handle:
      line = line.strip()
      if not line:
        continue
      pick = json.loads(line)
      if pool != "all" and pick.get("pool", "primary") != pool:
        continue
      if _is_successful(pick.get("is_successful")):
        if not allow_successes:
          success_examples.append(
              str(pick.get("episode_id") or pick.get("pkl_path") or "<unknown>")
          )
          continue
      picks.append(pick)
  if success_examples and not allow_successes:
    preview = ", ".join(success_examples[:3])
    raise ValueError(
        "Sample manifest contains validator-success episodes. "
        "This website is in failure-mode mode and refuses successes by default. "
        f"Examples: {preview}. Use --allow_successes only for validator-audit."
    )
  return picks


def _done_key(row: dict[str, Any]) -> str:
  return f"{row.get('pool', 'primary')}::{row.get('episode_id')}"


class AnnotationState:
  def __init__(
      self,
      sample_manifest: Path,
      out_dir: Path,
      annotator: str,
      blind: bool,
      pool: str,
      max_frames: int,
      max_dim: int,
      allow_successes: bool,
  ) -> None:
    self.sample_manifest = sample_manifest
    self.out_dir = out_dir
    self.annotator = annotator
    self.blind = blind
    self.pool = pool
    self.max_frames = max_frames
    self.max_dim = max_dim
    self.allow_successes = allow_successes
    self.frames_root = out_dir / "frames"
    self.annotations_path = out_dir / "annotations.jsonl"
    self.audit_path = out_dir / "audit_flags.jsonl"
    self.out_dir.mkdir(parents=True, exist_ok=True)
    self.frames_root.mkdir(parents=True, exist_ok=True)
    self.picks = _load_sample_manifest(sample_manifest, pool, allow_successes)

  def annotations(self) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not self.annotations_path.exists():
      return latest
    with self.annotations_path.open("r", encoding="utf-8") as handle:
      for line in handle:
        line = line.strip()
        if not line:
          continue
        try:
          row = json.loads(line)
        except json.JSONDecodeError:
          continue
        if row.get("annotator") != self.annotator:
          continue
        latest[_done_key(row)] = row
    return latest

  def case_list(self) -> list[dict[str, Any]]:
    annotations = self.annotations()
    out = []
    for index, pick in enumerate(self.picks):
      key = f"{pick.get('pool', 'primary')}::{pick.get('episode_id')}"
      public = _public_pick(pick, self.blind)
      public.update({"index": index, "done": key in annotations})
      out.append(public)
    return out

  def load_case(self, index: int) -> dict[str, Any]:
    pick = self.picks[index]
    pkl_path = Path(pick["pkl_path"])
    payload = _read_pkl_gz(pkl_path)
    episodes = payload if isinstance(payload, list) else [payload]
    episode = episodes[pick.get("episode_index", 0)]
    if not isinstance(episode, dict):
      raise ValueError(f"Episode is not a dict: {pkl_path}")
    episode_id = str(pick["episode_id"])
    episode_dir = self.frames_root / episode_id
    artifacts = _save_episode_artifacts(
        pick=pick,
        episode=episode,
        out_dir=episode_dir,
        max_frames=self.max_frames,
        max_dim=self.max_dim,
        blind=self.blind,
    )
    summary_path = Path(artifacts["summary"])
    frames = []
    for frame in artifacts["frames"]:
      file_name = Path(frame["path"]).name
      frames.append(
          {
              "step": frame["step"],
              "field": frame["field"],
              # Use the public queue index, not the private episode identifier,
              # in browser-visible artifact URLs.
              "url": f"/artifact/{index}/{quote(file_name)}",
          }
      )
    annotation = self.annotations().get(
        f"{pick.get('pool', 'primary')}::{pick.get('episode_id')}"
    )
    return {
        "pick": _public_pick(pick, self.blind),
        "frames": frames,
        "summary_markdown": summary_path.read_text(encoding="utf-8"),
        "annotation": _public_annotation(annotation),
    }

  def save_annotation(self, payload: dict[str, Any]) -> dict[str, Any]:
    raw_index = payload.get("index")
    if isinstance(raw_index, bool):
      raise ValueError(f"Invalid case index: {raw_index!r}")
    try:
      index = int(raw_index)
    except (TypeError, ValueError) as exc:
      raise ValueError(f"Invalid case index: {raw_index!r}") from exc
    if index < 0 or index >= len(self.picks):
      raise ValueError(f"Case index out of range: {index}")
    pick = self.picks[index]
    if _is_successful(pick.get("is_successful")) and not self.allow_successes:
      raise ValueError(
          "Refusing to annotate a validator-success episode in failure-mode mode. "
          "Use --allow_successes only for the separate validator-audit pass."
      )
    human_label = _validated_human_label(payload)

    episode_id = str(pick["episode_id"])
    saved_at = dt.datetime.now(dt.timezone.utc).isoformat()
    record = {
        "annotator": self.annotator,
        "pool": pick.get("pool", "primary"),
        "episode_id": episode_id,
        "model_name": pick["model_name"],
        "category": pick["category"],
        "app_id": pick["app_id"],
        "task_template": pick["task_template"],
        "goal": pick["goal"],
        "is_successful": pick["is_successful"],
        "pkl_path": pick["pkl_path"],
        "frames_dir": str(self.frames_root / episode_id),
        "annotated_at": saved_at,
        "created_at": saved_at,
        "human_label": human_label,
        "judge_prior": {
            "primary_failure_mode": pick.get("judge_label"),
            "confidence": pick.get("judge_confidence"),
            "verdict": pick.get("judge_verdict"),
            "rationale": pick.get("judge_rationale"),
        },
        "suspect_validator": bool(payload.get("suspect_validator")),
    }
    with self.annotations_path.open("a", encoding="utf-8") as handle:
      fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
      handle.write(json.dumps(record, ensure_ascii=False) + "\n")
      handle.flush()
      fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    if record["suspect_validator"]:
      with self.audit_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {"ok": True, "saved_at": saved_at}


class AnnotationHandler(BaseHTTPRequestHandler):
  server: "AnnotationHTTPServer"

  def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
    sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))

  def _is_authenticated(self) -> bool:
    auth = self.server.basic_auth
    return auth is None or auth.accepts(self.headers.get("Authorization"))

  def _send_authentication_required(self) -> None:
    body = b"Authentication required."
    self.send_response(HTTPStatus.UNAUTHORIZED)
    self.send_header(
        "WWW-Authenticate", f'Basic realm="{AUTH_REALM}", charset="UTF-8"'
    )
    self.send_header("Cache-Control", "no-store")
    self.send_header("Content-Type", "text/plain; charset=utf-8")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)

  def do_GET(self) -> None:  # noqa: N802
    if not self._is_authenticated():
      self._send_authentication_required()
      return
    parsed = urlparse(self.path)
    try:
      if parsed.path == "/":
        _text_response(self, INDEX_HTML, content_type="text/html; charset=utf-8")
      elif parsed.path == "/api/state":
        self._handle_state()
      elif parsed.path == "/api/case":
        self._handle_case(parsed.query)
      elif parsed.path.startswith("/artifact/"):
        self._handle_artifact(parsed.path)
      else:
        _text_response(self, "not found", HTTPStatus.NOT_FOUND)
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self.log_error("case request failed: %s", exc)
      message = (
          "Unable to load the case; see the server log."
          if self.server.state.blind
          else str(exc)
      )
      _text_response(self, message, HTTPStatus.INTERNAL_SERVER_ERROR)

  def do_POST(self) -> None:  # noqa: N802
    if not self._is_authenticated():
      self._send_authentication_required()
      return
    parsed = urlparse(self.path)
    try:
      if parsed.path != "/api/annotation":
        _text_response(self, "not found", HTTPStatus.NOT_FOUND)
        return
      length = int(self.headers.get("Content-Length", "0"))
      payload = json.loads(self.rfile.read(length).decode("utf-8"))
      _json_response(self, self.server.state.save_annotation(payload))
    except Exception as exc:  # pylint: disable=broad-exception-caught
      _text_response(self, str(exc), HTTPStatus.BAD_REQUEST)

  def _handle_state(self) -> None:
    cases = self.server.state.case_list()
    done_count = sum(1 for item in cases if item["done"])
    payload = {
        "annotator": self.server.state.annotator,
        "blind": self.server.state.blind,
        "total": len(cases),
        "done_count": done_count,
        "failure_modes": list(DEFAULT_FAILURE_MODES),
        "cases": cases,
    }
    if not self.server.state.blind:
      payload["pool"] = self.server.state.pool
      payload["allow_successes"] = self.server.state.allow_successes
    _json_response(self, payload)

  def _handle_case(self, query: str) -> None:
    params = parse_qs(query)
    index = int(params.get("index", ["0"])[0])
    if index < 0 or index >= len(self.server.state.picks):
      _text_response(self, "case index out of range", HTTPStatus.BAD_REQUEST)
      return
    _json_response(self, self.server.state.load_case(index))

  def _handle_artifact(self, path: str) -> None:
    parts = path.split("/", 3)
    if len(parts) != 4:
      _text_response(self, "bad artifact path", HTTPStatus.BAD_REQUEST)
      return
    try:
      index = int(unquote(parts[2]))
    except ValueError:
      _text_response(self, "bad artifact case", HTTPStatus.BAD_REQUEST)
      return
    if index < 0 or index >= len(self.server.state.picks):
      _text_response(self, "artifact case out of range", HTTPStatus.BAD_REQUEST)
      return
    episode_id = str(self.server.state.picks[index]["episode_id"])
    file_name = unquote(parts[3])
    root = (self.server.state.frames_root / episode_id).resolve()
    file_path = (root / file_name).resolve()
    if root not in file_path.parents or not file_path.exists():
      _text_response(self, "artifact not found", HTTPStatus.NOT_FOUND)
      return
    content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
    body = file_path.read_bytes()
    self.send_response(HTTPStatus.OK)
    self.send_header("Content-Type", content_type)
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)


class AnnotationHTTPServer(ThreadingHTTPServer):
  def __init__(
      self,
      server_address: tuple[str, int],
      state: AnnotationState,
      basic_auth: BasicAuth | None = None,
  ) -> None:
    super().__init__(server_address, AnnotationHandler)
    self.state = state
    self.basic_auth = basic_auth


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--sample_manifest", required=True)
  parser.add_argument("--out_dir", required=True)
  parser.add_argument("--annotator", required=True)
  parser.add_argument("--blind", dest="blind", action="store_true", default=True)
  parser.add_argument(
      "--show_prior_judge",
      dest="blind",
      action="store_false",
      help="Show prior VLM labels. Do not use for validation.",
  )
  parser.add_argument(
      "--pool",
      choices=("all", "primary", "double_annotation"),
      default="primary",
  )
  parser.add_argument(
      "--max_frames",
      type=int,
      default=0,
      help="Number of screenshots to show; 0 means all available step screenshots.",
  )
  parser.add_argument("--max_dim", type=int, default=896)
  parser.add_argument(
      "--allow_successes",
      action="store_true",
      help=(
          "Allow validator-success episodes in the queue. Use only for the "
          "separate validator-audit pass; failure-mode validation should leave "
          "this off."
      ),
  )
  parser.add_argument("--host", default="127.0.0.1")
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument(
      "--auth_username",
      help=(
          "Enable HTTP Basic authentication with this username. A password "
          "source must also be configured; the password is never accepted on "
          "the command line."
      ),
  )
  password_source = parser.add_mutually_exclusive_group()
  password_source.add_argument(
      "--auth_password_file",
      help="Read the HTTP Basic authentication password from this file.",
  )
  password_source.add_argument(
      "--auth_password_env",
      help="Read the HTTP Basic authentication password from this environment variable.",
  )
  args = parser.parse_args()

  try:
    basic_auth = _load_basic_auth(
        args.auth_username, args.auth_password_file, args.auth_password_env
    )
  except (OSError, UnicodeError, ValueError) as exc:
    parser.error(str(exc))

  state = AnnotationState(
      sample_manifest=Path(args.sample_manifest).expanduser().resolve(),
      out_dir=Path(args.out_dir).expanduser().resolve(),
      annotator=args.annotator,
      blind=args.blind,
      pool=args.pool,
      max_frames=args.max_frames,
      max_dim=args.max_dim,
      allow_successes=args.allow_successes,
  )
  server = AnnotationHTTPServer((args.host, args.port), state, basic_auth=basic_auth)
  print(
      f"Serving {len(state.picks)} cases for annotator {args.annotator} at "
      f"http://{args.host}:{args.port} "
      f"(authentication {'enabled' if basic_auth is not None else 'disabled'})",
      flush=True,
  )
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("\nStopping annotation server.", flush=True)
  finally:
    server.server_close()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
