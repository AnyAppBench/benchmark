#!/usr/bin/env python3
"""Focused tests for the browser annotation protocol (no browser required)."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import annotate_judge_cases as annotate  # noqa: E402
import serve_annotation_app as serve  # noqa: E402


def _pick() -> dict[str, object]:
  return {
      "episode_id": "private-episode-id",
      "model_name": "GPT-5.1-private",
      "condition": "C1-private",
      "pool": "primary",
      "source": "private-source-roster",
      "category": "clock",
      "app_id": "clock_chrono",
      "app_name": "Chrono",
      "task_template": "ClockCreateAlarmForChrono",
      "goal": "In Chrono, create an alarm at 10:00.",
      "is_successful": 0.0,
      "pkl_path": "$HOME/private/run/GPT-5.1-private/case.pkl.gz",
      "jsonl_path": "/home/researcher/private/judgments.jsonl",
      "judge_label": "planning",
      "judge_confidence": "high",
      "judge_rationale": "PRIVATE-JUDGE-RATIONALE",
      "judge_model": "Gemini-private-judge",
      "endpoint": "https://private.example/v1/chat/completions",
      "extra_private_source_field": "must-not-cross-the-api",
  }


class BlindReportTest(unittest.TestCase):

  def test_blind_report_redacts_audit_values_and_keeps_full_trace(self) -> None:
    pick = _pick()
    episode = {
        "episode_data": {
            "thought": [
                "Use GPT-5.1-private to inspect the alarm list.",
                "gemini-3.1-pro-preview endpoint=https://private.example/v1/chat/completions then continue",
                "Read /tmp/custom/private/run/debug.log, then inspect "
                "/storage/emulated/0/Documents/task.txt and finish the UI task.",
            ],
            "action": ["open Chrono", "tap the add-alarm button", "type 10:00"],
            "summary": [
                "source=private-source-roster",
                "api_key=0123456789abcdef0123456789abcdef",
                "The alarm was not saved.",
            ],
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
      artifacts = annotate._render_html_report(  # pylint: disable=protected-access
          pick=pick,
          episode=episode,
          out_dir=Path(tmp),
          max_frames=1,
          max_dim=896,
          blind=True,
      )
      summary = Path(artifacts["summary_md"]).read_text(encoding="utf-8")
      report = Path(artifacts["html_path"]).read_text(encoding="utf-8")

    visible = summary + report
    for private in (
        "GPT-5.1-private",
        "gemini-3.1-pro-preview",
        "private.example",
        "$HOME/private",
        "/tmp/custom/private",
        "/home/researcher",
        "private-source-roster",
        "PRIVATE-JUDGE-RATIONALE",
        "0123456789abcdef0123456789abcdef",
    ):
      self.assertNotIn(private, visible)
    self.assertIn("In Chrono, create an alarm at 10:00.", visible)
    self.assertIn("open Chrono", summary)
    self.assertIn("tap the add-alarm button", summary)
    self.assertIn("type 10:00", summary)
    self.assertIn("/storage/emulated/0/Documents/task.txt", visible)
    self.assertIn("### Step 3", summary)
    self.assertIn("Validator verdict (read-only)", visible)


class PublicApiShapeTest(unittest.TestCase):

  def test_blind_pick_is_an_allowlist(self) -> None:
    public = serve._public_pick(_pick(), blind=True)  # pylint: disable=protected-access
    self.assertEqual(
        set(public),
        {"category", "app_id", "app_name", "task_template", "goal", "is_successful"},
    )
    serialized = json.dumps(public)
    for private in (
        "model_name",
        "condition",
        "pool",
        "episode_id",
        "pkl_path",
        "judge_label",
        "judge_rationale",
        "extra_private_source_field",
    ):
      self.assertNotIn(private, serialized)

  def test_saved_annotation_prefill_omits_private_audit_fields(self) -> None:
    annotation = {
        "model_name": "private-model",
        "pkl_path": "/private/path.pkl.gz",
        "pool": "primary",
        "judge_prior": {"primary_failure_mode": "planning"},
        "human_label": {
            "primary_failure_mode": "grounding",
            "planning_score": 1,
            "grounding_score": 3,
            "confidence": "medium",
            "rationale": "Missed the visible button.",
            "private_note": "do not expose",
        },
        "suspect_validator": False,
    }
    public = serve._public_annotation(annotation)  # pylint: disable=protected-access
    self.assertEqual(public["human_label"]["primary_failure_mode"], "grounding")
    serialized = json.dumps(public)
    self.assertNotIn("private-model", serialized)
    self.assertNotIn("/private/path", serialized)
    self.assertNotIn("judge_prior", serialized)
    self.assertNotIn("private_note", serialized)

  def test_blind_case_list_and_case_payload_do_not_expose_source_identity(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      manifest = root / "sample.jsonl"
      manifest.write_text(json.dumps(_pick()) + "\n", encoding="utf-8")
      state = serve.AnnotationState(
          sample_manifest=manifest,
          out_dir=root / "out",
          annotator="human-a",
          blind=True,
          pool="primary",
          max_frames=6,
          max_dim=896,
          allow_successes=False,
      )
      annotation = {
          "annotator": "human-a",
          "pool": "primary",
          "episode_id": "private-episode-id",
          "model_name": "private-model",
          "pkl_path": "/private/path.pkl.gz",
          "judge_prior": {"primary_failure_mode": "planning"},
          "human_label": {
              "primary_failure_mode": "grounding",
              "planning_score": 1,
              "grounding_score": 3,
              "confidence": "medium",
              "rationale": "Missed the visible button.",
          },
          "suspect_validator": False,
      }
      state.annotations_path.write_text(
          json.dumps(annotation) + "\n", encoding="utf-8"
      )
      summary_path = root / "safe-summary.md"
      summary_path.write_text("Full action trace", encoding="utf-8")
      fake_artifacts = {
          "summary": str(summary_path),
          "frames": [
              {"step": 4, "field": "before_screenshot", "path": "/tmp/frame.jpg"}
          ],
      }
      with (
          mock.patch.object(serve, "_read_pkl_gz", return_value={"episode_data": {}}),
          mock.patch.object(serve, "_save_episode_artifacts", return_value=fake_artifacts),
      ):
        case = state.load_case(0)

      queue_serialized = json.dumps(state.case_list())
      case_serialized = json.dumps(case)
      for private in (
          "GPT-5.1-private",
          "C1-private",
          "private-source-roster",
          "$HOME",
          "PRIVATE-JUDGE-RATIONALE",
          "must-not-cross-the-api",
      ):
        self.assertNotIn(private, queue_serialized)
        self.assertNotIn(private, case_serialized)
      self.assertEqual(case["frames"][0]["url"], "/artifact/0/frame.jpg")
      self.assertEqual(
          case["annotation"]["human_label"]["primary_failure_mode"], "grounding"
      )


class AnnotationValidationTest(unittest.TestCase):

  def test_form_has_blank_defaults(self) -> None:
    self.assertNotIn('value="0"', serve.INDEX_HTML)
    self.assertIn("confidence: null", serve.INDEX_HTML)
    self.assertNotIn('|| "planning"', serve.INDEX_HTML)
    self.assertNotIn('|| "high"', serve.INDEX_HTML)
    self.assertIn("Select failure mode", serve.INDEX_HTML)

  def test_backend_requires_every_human_label_field(self) -> None:
    valid = {
        "primary_failure_mode": "planning",
        "planning_score": 3,
        "grounding_score": 1,
        "confidence": "medium",
        "rationale": "The agent stopped before opening the add-alarm form.",
    }
    label = serve._validated_human_label(valid)  # pylint: disable=protected-access
    self.assertEqual(label, valid)

    for missing in valid:
      bad = dict(valid)
      del bad[missing]
      with self.subTest(missing=missing):
        with self.assertRaises(ValueError):
          serve._validated_human_label(bad)  # pylint: disable=protected-access

  def test_backend_rejects_clamped_or_implicit_scores(self) -> None:
    base = {
        "primary_failure_mode": "planning",
        "planning_score": 3,
        "grounding_score": 1,
        "confidence": "medium",
        "rationale": "Stopped early.",
    }
    for value in (-1, 4, 1.5, True, "", None):
      bad = {**base, "planning_score": value}
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          serve._validated_human_label(bad)  # pylint: disable=protected-access


class _MinimalHttpState:
  annotator = "human-a"
  blind = True
  pool = "primary"
  allow_successes = False
  picks: list[dict[str, object]] = []

  @staticmethod
  def case_list() -> list[dict[str, object]]:
    return []


def _authorization(username: str, password: str) -> str:
  encoded = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode(
      "ascii"
  )
  return f"Basic {encoded}"


class BasicAuthenticationTest(unittest.TestCase):

  def setUp(self) -> None:
    self.username = "private-user"
    self.password = "private-password"

  def _handler(
      self,
      authorization: str | None,
      path: str,
  ) -> tuple[serve.AnnotationHandler, list[int], dict[str, str]]:
    handler = serve.AnnotationHandler.__new__(serve.AnnotationHandler)
    handler.server = SimpleNamespace(
        basic_auth=serve.BasicAuth(self.username, self.password),
        state=_MinimalHttpState(),
    )
    handler.path = path
    handler.headers = {}
    if authorization is not None:
      handler.headers["Authorization"] = authorization
    handler.rfile = io.BytesIO(b"{}")
    handler.wfile = io.BytesIO()
    statuses: list[int] = []
    response_headers: dict[str, str] = {}
    handler.send_response = lambda status: statuses.append(status)
    handler.send_header = (
        lambda key, value: response_headers.__setitem__(key.lower(), value)
    )
    handler.end_headers = lambda: None
    return handler, statuses, response_headers

  def test_unauthenticated_get_post_api_and_artifacts_all_challenge(self) -> None:
    requests = (
        ("GET", "/", None),
        ("GET", "/api/state", None),
        ("GET", "/api/case?index=0", None),
        ("GET", "/artifact/0/frame.jpg", None),
        ("POST", "/api/annotation", "{}"),
    )
    for method, path, body in requests:
      with self.subTest(method=method, path=path):
        handler, statuses, headers = self._handler(None, path)
        if body is not None:
          handler.headers["Content-Length"] = str(len(body))
          handler.rfile = io.BytesIO(body.encode("utf-8"))
        getattr(handler, f"do_{method}")()
        response_body = handler.wfile.getvalue().decode("utf-8")
        self.assertEqual(statuses, [401])
        self.assertEqual(
            headers.get("www-authenticate"),
            'Basic realm="CATBench Annotation", charset="UTF-8"',
        )
        self.assertEqual(headers.get("cache-control"), "no-store")
        self.assertNotIn(self.username, response_body)
        self.assertNotIn(self.password, response_body)

  def test_wrong_credentials_are_rejected_and_valid_credentials_work(self) -> None:
    for header in (
        _authorization(self.username, "wrong"),
        _authorization("wrong", self.password),
        "Basic malformed%%%",
        "Bearer token",
    ):
      with self.subTest(header=header):
        handler, statuses, _ = self._handler(header, "/api/state")
        handler.do_GET()
        self.assertEqual(statuses, [401])

    handler, _, _ = self._handler(
        _authorization(self.username, self.password), "/api/state"
    )
    self.assertTrue(handler._is_authenticated())  # pylint: disable=protected-access

  def test_both_credential_components_use_constant_time_comparison(self) -> None:
    auth = serve.BasicAuth(self.username, self.password)
    with mock.patch.object(
        serve.hmac, "compare_digest", wraps=serve.hmac.compare_digest
    ) as compare:
      self.assertFalse(auth.accepts(_authorization("wrong", self.password)))
    self.assertEqual(compare.call_count, 2)

  def test_auth_can_be_loaded_from_file_or_environment_but_not_partially(self) -> None:
    with tempfile.TemporaryDirectory() as tmp:
      secret_path = Path(tmp) / "password.txt"
      secret_path.write_text(self.password + "\n", encoding="utf-8")
      from_file = serve._load_basic_auth(  # pylint: disable=protected-access
          self.username, str(secret_path), None
      )
    self.assertTrue(
        from_file.accepts(_authorization(self.username, self.password))
    )
    self.assertNotIn(self.password, repr(from_file))

    with mock.patch.dict(os.environ, {"CATBENCH_TEST_PASSWORD": self.password}):
      from_env = serve._load_basic_auth(  # pylint: disable=protected-access
          self.username, None, "CATBENCH_TEST_PASSWORD"
      )
    self.assertTrue(from_env.accepts(_authorization(self.username, self.password)))

    self.assertIsNone(
        serve._load_basic_auth(None, None, None)  # pylint: disable=protected-access
    )
    with self.assertRaises(ValueError):
      serve._load_basic_auth(  # pylint: disable=protected-access
          self.username, None, None
      )
    with self.assertRaises(ValueError):
      serve._load_basic_auth(  # pylint: disable=protected-access
          None, str(secret_path), None
      )

  def test_omitting_auth_keeps_legacy_unauthenticated_behavior(self) -> None:
    handler, _, _ = self._handler(None, "/api/state")
    handler.server.basic_auth = None
    self.assertTrue(handler._is_authenticated())  # pylint: disable=protected-access

if __name__ == "__main__":
  unittest.main()
