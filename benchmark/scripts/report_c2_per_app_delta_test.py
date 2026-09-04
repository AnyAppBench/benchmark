"""Tests for the C2 per-app delta reporter using a tiny synthetic batch."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
import sys
import tempfile
import unittest

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
for _p in (SCRIPT_DIR, BENCHMARK_ROOT):
  if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))

import report_c2_per_app_delta as subject  # pylint: disable=wrong-import-position

MODELS = ("UI-Venus-7B", "GUI-Owl-7B")
COHORT = {
    "models": list(MODELS),
    "categories": {
        "sms": {
            "aw_app_id": "sms_simple_sms_messenger",
            "app_ids": ["sms_simple_sms_messenger", "sms_fossify_messages"],
            "semantic_task_ids": ["SmsSend", "SmsSendToContact"],
        },
    },
}
MODEL_CONFIG = {"models": [{"name": m} for m in MODELS]}
APP_SUFFIX = {
    "sms_simple_sms_messenger": "SimpleSMSMessenger",
    "sms_fossify_messages": "FossifyMessages",
}


def _episode(template: str, instance: int, success: float, **overrides) -> dict:
  ep = {
      "task_template": template,
      "instance_id": instance,
      "is_successful": success,
      "exception_info": None,
      "catbench_condition": "c2_g",
      "catbench_condition_config_valid": True,
      "catbench_episode_status": "valid_success" if success >= 0.5 else "valid_failure",
  }
  ep.update(overrides)
  return ep


def _write(path: Path, episodes: list[dict]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with gzip.open(path, "wb") as gz:
    pickle.dump(episodes, gz)


class ReportC2PerAppDeltaTest(unittest.TestCase):

  def setUp(self) -> None:
    self._tmp = tempfile.TemporaryDirectory()
    self.tmp = Path(self._tmp.name)
    self.cohort_path = self.tmp / "cohort.json"
    self.cohort_path.write_text(json.dumps(COHORT), encoding="utf-8")
    self.config_path = self.tmp / "models.json"
    self.config_path.write_text(json.dumps(MODEL_CONFIG), encoding="utf-8")
    self.baseline_path = self.tmp / "k1.json"
    per_app = []
    for model in MODELS:
      for app_id in APP_SUFFIX:
        per_app.append({
            "model": model, "category": "sms", "app_id": app_id,
            "K1": {"AW": {"successes": 1, "total": 1}, "New": {"successes": 0, "total": 1}},
        })
    self.baseline_path.write_text(json.dumps({"per_app_rows": per_app}), encoding="utf-8")
    self.root = self.tmp / "c2"

  def tearDown(self) -> None:
    self._tmp.cleanup()

  def _ckpt(self, replicate: str, model: str, app_id: str, template: str, instance: int) -> Path:
    return (
        self.root / replicate / "laneA" / "matrix" / subject._slug(model) / "sms" / app_id
        / "run_1" / f"{template}For{APP_SUFFIX[app_id]}_{instance}.pkl.gz"
    )

  def _build_fixture(self) -> None:
    """Full schedule for Venus; Owl has gaps, invalid and mis-conditioned episodes."""
    for rep in ("r1", "r2"):
      for app_id in APP_SUFFIX:
        for sid in ("SmsSend", "SmsSendToContact"):
          for inst in range(3):
            # Venus: instance 0 succeeds, others fail; r2 mirrors r1 except one flip.
            success = 1.0 if inst == 0 else 0.0
            if rep == "r2" and app_id == "sms_fossify_messages" and sid == "SmsSend" and inst == 1:
              success = 1.0
            _write(self._ckpt(rep, "UI-Venus-7B", app_id, sid, inst),
                   [_episode(f"{sid}For{APP_SUFFIX[app_id]}", inst, success)])
    # Owl, r1 only: simple messenger complete, all successes except:
    #   SmsSend#1 wrong condition, SmsSend#2 infrastructure-invalid.
    app = "sms_simple_sms_messenger"
    for sid in ("SmsSend", "SmsSendToContact"):
      for inst in range(3):
        overrides = {}
        if sid == "SmsSend" and inst == 1:
          overrides = {"catbench_condition": "c1"}
        if sid == "SmsSend" and inst == 2:
          overrides = {"catbench_episode_status": "invalid_infrastructure", "is_successful": 0.0}
        _write(self._ckpt("r1", "GUI-Owl-7B", app, sid, inst),
               [_episode(f"{sid}For{APP_SUFFIX[app]}", inst, 1.0, **overrides)])
    # Owl fossify: only instance 0 of SmsSend present (5 missing of 6).
    _write(self._ckpt("r1", "GUI-Owl-7B", "sms_fossify_messages", "SmsSend", 0),
           [_episode("SmsSendForFossifyMessages", 0, 0.0)])

  def _run(self, *extra: str) -> Path:
    out = self.tmp / "out"
    subject.main([
        "--c2_root", str(self.root), "--c1_baseline", str(self.baseline_path),
        "--model_config", str(self.config_path), "--cohort_manifest", str(self.cohort_path),
        "--out_dir", str(out), "--workers", "1", *extra,
    ])
    return out

  def test_counts_exclusions_and_missing(self) -> None:
    self._build_fixture()
    out = self._run()
    summary = json.loads((out / "c2_summary_k3.json").read_text(encoding="utf-8"))
    pooled = {(r["model"], r["app_id"]): r for r in summary["c2_pooled"]}

    venus = pooled[("UI-Venus-7B", "sms_simple_sms_messenger")]
    self.assertEqual((venus["scheduled"], venus["valid"], venus["success"], venus["missing"]), (12, 12, 4, 0))

    r1 = {(r["model"], r["app_id"]): r for r in summary["c2_per_replicate"]["r1"]}
    owl_simple = r1[("GUI-Owl-7B", "sms_simple_sms_messenger")]
    self.assertEqual(owl_simple["excluded_condition"], 1)
    self.assertEqual(owl_simple["invalid"], 1)
    self.assertEqual(owl_simple["valid"], 4)
    self.assertEqual(owl_simple["success"], 4)
    # Wrong-condition episode leaves its slot empty and is reported as missing.
    self.assertEqual(owl_simple["missing"], 1)
    self.assertIn("SmsSend#1", owl_simple["missing_slots"])

    owl_fossify = r1[("GUI-Owl-7B", "sms_fossify_messages")]
    self.assertEqual(owl_fossify["missing"], 5)
    self.assertEqual(owl_fossify["valid"], 1)
    # Pooled counts add r2, where Owl has no checkpoints at all (6 more missing).
    self.assertEqual(pooled[("GUI-Owl-7B", "sms_simple_sms_messenger")]["missing"], 7)
    self.assertEqual(pooled[("GUI-Owl-7B", "sms_fossify_messages")]["missing"], 11)

    # Only Venus has two replicates; Owl has r1 only, so r2 is all-missing.
    self.assertEqual(summary["replicates"], ["r1", "r2"])
    r2_owl = {(r["model"], r["app_id"]): r for r in summary["c2_per_replicate"]["r2"]}
    self.assertEqual(r2_owl[("GUI-Owl-7B", "sms_simple_sms_messenger")]["missing"], 6)

    agreement = summary["replicate_agreement"]["UI-Venus-7B"]
    self.assertAlmostEqual(agreement["rate_per_replicate"]["r1"], 100.0 * 4 / 12)
    self.assertAlmostEqual(agreement["rate_per_replicate"]["r2"], 100.0 * 5 / 12)
    # Per-app |diff|: simple 0, fossify |3/6 - 2/6| = 16.67 -> mean 8.33.
    self.assertAlmostEqual(agreement["mean_abs_per_app_diff"], 100.0 / 12)

    counts_md = (out / "c2_per_app_counts_k3.md").read_text(encoding="utf-8")
    self.assertIn("cond=1", counts_md)
    self.assertIn("miss=5", counts_md)
    self.assertTrue((out / "c2_per_app_counts_k3.csv").exists())

  def test_delta_table_and_formatting(self) -> None:
    self._build_fixture()
    out = self._run()
    summary = json.loads((out / "c2_summary_k3.json").read_text(encoding="utf-8"))
    rows = {r["label"]: r for r in summary["delta_table"]}
    # Venus on Simple SMS Messenger: C2 4/12 = 33.3%, C1 1/2 = 50% -> -16.7.
    venus_cell = rows["Simple SMS Messenger"]["cells"][0]
    self.assertAlmostEqual(venus_cell["delta"], 100.0 * 4 / 12 - 50.0)
    # Owl on Simple SMS Messenger: 4/4 = 100% vs 50% -> +50, flagged incomplete.
    owl_cell = rows["Simple SMS Messenger"]["cells"][1]
    self.assertAlmostEqual(owl_cell["delta"], 50.0)
    self.assertTrue(owl_cell["c2_incomplete"])
    self.assertIn("Overall", rows)
    self.assertIn("SMS (all apps)", rows)

    tex = (out / "c2_per_app_delta_k3.tex").read_text(encoding="utf-8")
    self.assertIn(r"\label{tab:c2_per_app_delta_appendix}", tex)
    self.assertIn(r"+50$^{*}$", tex)
    self.assertIn("-16.7", tex)
    md = (out / "c2_per_app_delta_k3.md").read_text(encoding="utf-8")
    self.assertIn("| SMS | Simple SMS Messenger | -16.7 | +50* |", md)

  def test_fmt_delta(self) -> None:
    self.assertEqual(subject.fmt_delta(20.0), "+20")
    self.assertEqual(subject.fmt_delta(-30.0), "-30")
    self.assertEqual(subject.fmt_delta(0.0), "0")
    self.assertEqual(subject.fmt_delta(-0.04), "0")
    self.assertEqual(subject.fmt_delta(12.34), "+12.3")
    self.assertEqual(subject.fmt_delta(-3.35), "-3.4")
    self.assertEqual(subject.fmt_delta(None), "--")

  def test_instance_filter(self) -> None:
    self._build_fixture()
    out = self._run()
    inst0 = json.loads((out / "c2_summary_inst0.json").read_text(encoding="utf-8"))
    pooled = {(r["model"], r["app_id"]): r for r in inst0["c2_pooled"]}
    venus = pooled[("UI-Venus-7B", "sms_simple_sms_messenger")]
    # 2 templates x 1 instance x 2 replicates, all instance-0 episodes succeed.
    self.assertEqual((venus["scheduled"], venus["valid"], venus["success"]), (4, 4, 4))
    owl = pooled[("GUI-Owl-7B", "sms_simple_sms_messenger")]
    self.assertEqual(owl["excluded_condition"], 0)

    out2 = self.tmp / "out2"
    subject.main([
        "--c2_root", str(self.root), "--c1_baseline", str(self.baseline_path),
        "--model_config", str(self.config_path), "--cohort_manifest", str(self.cohort_path),
        "--out_dir", str(out2), "--workers", "1", "--instance_ids", "1", "2",
    ])
    self.assertTrue((out2 / "c2_summary_inst12.json").exists())
    self.assertFalse((out2 / "c2_summary_k3.json").exists())
    s12 = json.loads((out2 / "c2_summary_inst12.json").read_text(encoding="utf-8"))
    venus12 = {(r["model"], r["app_id"]): r for r in s12["c2_pooled"]}[
        ("UI-Venus-7B", "sms_simple_sms_messenger")]
    self.assertEqual((venus12["scheduled"], venus12["success"]), (8, 0))

  def test_flat_layout_and_c1_root(self) -> None:
    self._build_fixture()
    flat_root = self.root / "r1"
    c1_root = self.tmp / "c1"
    for app_id in APP_SUFFIX:
      for sid in ("SmsSend", "SmsSendToContact"):
        for inst in range(3):
          path = (c1_root / "r1" / "laneA" / "matrix" / "UI-Venus-7B" / "sms" / app_id
                  / "run_1" / f"{sid}For{APP_SUFFIX[app_id]}_{inst}.pkl.gz")
          _write(path, [_episode(f"{sid}For{APP_SUFFIX[app_id]}", inst, 1.0,
                                 catbench_condition="c1")])
    out = self.tmp / "out_flat"
    subject.main([
        "--c2_root", str(flat_root), "--c2_root_layout", "flat", "--c1_root", str(c1_root),
        "--model_config", str(self.config_path), "--cohort_manifest", str(self.cohort_path),
        "--out_dir", str(out), "--workers", "1", "--instance_ids", "0", "1", "2",
    ])
    summary = json.loads((out / "c2_summary_inst012.json").read_text(encoding="utf-8"))
    self.assertEqual(summary["replicates"], ["r1"])
    self.assertEqual(summary["c1_source"]["kind"], "c1_root")
    c1 = {(r["model"], r["app_id"]): r for r in summary["c1_counts"]}
    self.assertEqual(c1[("UI-Venus-7B", "sms_simple_sms_messenger")]["success"], 6)
    rows = {r["label"]: r for r in summary["delta_table"]}
    # Venus r1 only: 2/6 = 33.3% vs C1 100% -> -66.7.
    self.assertAlmostEqual(rows["Simple SMS Messenger"]["cells"][0]["delta"], 100.0 * 2 / 6 - 100.0)


if __name__ == "__main__":
  unittest.main()
