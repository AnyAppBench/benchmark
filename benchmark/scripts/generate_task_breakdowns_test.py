"""Tests for semantic C2 breakdown generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

import generate_task_breakdowns as generator


class SemanticPlanGenerationTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls) -> None:
    cls.cohort = (
        SCRIPT_DIR.parent / "configs" / "catbench_5cat_primary_cohort.json"
    )
    cohort_args = argparse.Namespace(
        tasks="",
        categories="sms,files,maps,contacts,clock",
        suite_family="android_world",
        n_task_combinations=3,
        task_random_seed=30,
        fixed_task_seed=False,
        limit=0,
        cohort_manifest=str(cls.cohort),
    )
    cls.real_entries = generator._iter_task_instances(  # pylint: disable=protected-access
        cohort_args
    )
    grouped: dict[str, list[dict]] = {}
    for entry in cls.real_entries:
      grouped.setdefault(entry["plan_key"], []).append(entry)
    cls.real_shared_pair = next(
        entries[:2]
        for entries in grouped.values()
        if len({entry["app_id"] for entry in entries}) >= 2
    )

  def _args(self, root: Path, **overrides) -> argparse.Namespace:
    values = {
        "output": str(root / "c2_g_plans.json"),
        "audit_log": str(root / "c2_g_attempts.jsonl"),
        "provider": "openai",
        "model": "gpt-5.1",
        "model_identity": "",
        "temperature": 0.0,
        "max_retry": 2,
        "timeout_sec": 120.0,
        "sleep_seconds": 0.0,
        "openai_base_url": "https://api.openai.com/v1/chat/completions",
        "openai_api_key": "unit-test-secret",
        "suite_family": "android_world",
        "categories": "sms,files,maps,contacts,clock",
        "tasks": "",
        "n_task_combinations": 3,
        "task_random_seed": 30,
        "fixed_task_seed": False,
        "limit": 0,
        "cohort_manifest": str(self.cohort),
        "resume": False,
        "overwrite": False,
        "dry_run": False,
        "dedupe_by_goal": True,
        "strict_forbidden_check": True,
        "validation_retry": 1,
        "allow_prompt_mismatch": False,
        "allow_provider_mismatch": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)

  def _audit(
      self,
      root: Path,
      args: argparse.Namespace,
      entries: list[dict] | None = None,
  ) -> generator.AttemptAuditLog:
    entries = entries or [self.real_entries[0]]
    binding = generator._attempt_audit_binding(  # pylint: disable=protected-access
        args,
        args.model,
        entries,
        args.openai_base_url,
    )
    return generator.AttemptAuditLog.create(
        Path(args.audit_log), binding, (args.openai_api_key,)
    )

  def test_dynamic_app_name_check_rejects_real_brand(self) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {"steps": ["Open Chrono and create the alarm."]},
        ("Chrono", "Google Clock"),
    )

    self.assertIn("app_name_mention", warnings)

  def test_generic_domain_noun_is_not_mistaken_for_app_identity(self) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {"steps": ["Create an alarm in the clock application."]},
        ("Clock",),
    )

    self.assertNotIn("app_name_mention", warnings)

  def test_goal_geographic_coordinates_are_preserved_task_values(self) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {
            "steps": [
                "Enter the geographic coordinates 47.23976, 9.5262837."
            ]
        },
        semantic_goal=(
            "Add a favorite marker for 47.23976, 9.5262837 in "
            "[TARGET_APP]."
        ),
    )

    self.assertNotIn("coordinate_word", warnings)

  def test_coordinate_word_without_goal_value_remains_forbidden(self) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {"steps": ["Use the screen coordinates to select the target."]},
        semantic_goal="Add a favorite marker for Madrid in [TARGET_APP].",
    )

    self.assertIn("coordinate_word", warnings)

  def test_goal_coordinate_allows_whitespace_and_terminal_punctuation(
      self,
  ) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {
            "steps": [
                "Enter the geographic coordinates:\n"
                "(47.23976 ,\t9.5262837)."
            ]
        },
        semantic_goal="Mark 47.23976, 9.5262837. in [TARGET_APP].",
    )

    self.assertNotIn("coordinate_pair", warnings)
    self.assertNotIn("coordinate_word", warnings)

  def test_goal_coordinate_does_not_hide_other_coordinate_leakage(self) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {
            "steps": [
                "Enter the geographic coordinates 47.23976, 9.5262837.",
                "Use screen coordinates 120.5, 300.25 to select it.",
            ]
        },
        semantic_goal="Mark 47.23976, 9.5262837 in [TARGET_APP].",
    )

    self.assertIn("coordinate_pair", warnings)
    self.assertIn("coordinate_word", warnings)

  def test_goal_coordinate_repurposed_as_pixel_location_is_forbidden(
      self,
  ) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {
            "steps": [
                "Tap position 47.23976, 9.5262837 as a pixel location."
            ]
        },
        semantic_goal="Mark 47.23976, 9.5262837 in [TARGET_APP].",
    )

    self.assertIn("coordinate_pair", warnings)

  def test_invented_decimal_pair_is_forbidden(self) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {"steps": ["Use the location value 47.20001, 9.50002."]},
        semantic_goal="Mark 47.23976, 9.5262837 in [TARGET_APP].",
    )

    self.assertIn("coordinate_pair", warnings)

  def test_unattached_coordinate_word_remains_forbidden(self) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {
            "steps": [
                "Enter the coordinates 47.23976, 9.5262837.",
                "Ensure the coordinates use the required format.",
            ]
        },
        semantic_goal="Mark 47.23976, 9.5262837 in [TARGET_APP].",
    )

    self.assertIn("coordinate_word", warnings)

  def test_geographic_references_are_checked_per_occurrence(self) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {
            "steps": [
                "Enter geographic coordinates 47.23976, 9.5262837."
            ],
            "notes": [
                "The location marker uses coordinates.",
                "The coordinates must be entered precisely as provided.",
            ],
        },
        semantic_goal="Mark 47.23976, 9.5262837 in [TARGET_APP].",
    )

    self.assertNotIn("coordinate_pair", warnings)
    self.assertNotIn("coordinate_word", warnings)

  def test_geographic_reference_requires_preserved_goal_pair(self) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {"steps": ["Create the location marker using coordinates."]},
        semantic_goal="Mark 47.23976, 9.5262837 in [TARGET_APP].",
    )

    self.assertIn("coordinate_word", warnings)

  def test_invalid_goal_ranges_do_not_create_coordinate_exemption(self) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {"steps": ["Enter the coordinates 91.5, 181.5."]},
        semantic_goal="Mark 91.5, 181.5 in [TARGET_APP].",
    )

    self.assertIn("coordinate_pair", warnings)
    self.assertIn("coordinate_word", warnings)

  def test_multiple_goal_coordinates_are_exempted_independently(self) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {
            "steps": [
                "Use coordinates 47.23976, 9.5262837.",
                "Then use coordinates -33.8688, 151.2093.",
            ]
        },
        semantic_goal=(
            "Compare 47.23976, 9.5262837 with -33.8688, 151.2093."
        ),
    )

    self.assertNotIn("coordinate_pair", warnings)
    self.assertNotIn("coordinate_word", warnings)

  def test_decimal_x_y_coordinate_is_forbidden(self) -> None:
    warnings = generator._forbidden_warnings(  # pylint: disable=protected-access
        {"steps": ["Use x = 120.5 and y:\t300.25."]},
    )

    self.assertIn("x_y_coordinate", warnings)

  def test_entry_persists_shared_plan_identity(self) -> None:
    task_item = {
        "key": "ClockCreateAlarmForChrono|goal-hash",
        "task_template": "ClockCreateAlarmForChrono",
        "instance_id": 0,
        "goal": "In Chrono, create an alarm for 10:00.",
        "goal_sha256": "goal-hash",
        "semantic_task_id": "ClockCreateAlarm",
        "app_display_name": "Chrono",
        "semantic_goal": "In [TARGET_APP], create an alarm for 10:00.",
        "semantic_goal_sha256": "semantic-goal-hash",
        "semantic_parameter_sha256": "parameter-hash",
        "plan_key": "ClockCreateAlarm|instance=0|semantic-goal-hash",
    }
    entry = generator._build_entry(  # pylint: disable=protected-access
        task_item,
        {"steps": ["Establish an alarm with the requested time."], "notes": []},
        "planner-revision",
        [],
    )

    self.assertEqual(task_item["plan_key"], entry["plan_key"])
    self.assertEqual("ClockCreateAlarm", entry["semantic_task_id"])
    self.assertEqual(64, len(entry["plan_sha256"]))

  def test_qwen_routes_through_openai_compatible_transport(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      args = self._args(
          Path(temp_dir),
          provider="qwen",
          model="catbench-judge",
          model_identity="Qwen/Qwen3-VL-30B-A3B-Instruct",
      )
      with mock.patch.object(
          generator, "_call_openai_once", return_value='{"steps": []}'
      ) as call:
        result = generator._call_provider_once(  # pylint: disable=protected-access
            args=args,
            prompt="semantic prompt",
            resolved_model=args.model,
            gemini_wrapper=None,
            openai_base_url="https://qwen.example/v1",
            openai_api_key="qwen-test-secret",
        )

    self.assertEqual('{"steps": []}', result)
    call.assert_called_once_with(
        prompt="semantic prompt",
        model="catbench-judge",
        base_url="https://qwen.example/v1",
        api_key="qwen-test-secret",
        timeout_sec=args.timeout_sec,
        temperature=args.temperature,
    )

  def test_qwen_default_model_is_endpoint_alias(self) -> None:
    args = argparse.Namespace(provider="qwen", model="")

    self.assertEqual(
        generator.DEFAULT_QWEN_MODEL,
        generator._resolve_model(args),  # pylint: disable=protected-access
    )

  def test_frozen_real_cohort_has_690_entries_and_150_plans(self) -> None:
    entries = self.real_entries

    self.assertEqual(690, len(entries))
    self.assertEqual(690, len({entry["key"] for entry in entries}))
    self.assertEqual(150, len({entry["plan_key"] for entry in entries}))
    self.assertEqual(
        30,
        sum(entry["app_id"] == "clock_clockyou" for entry in entries),
    )

  def test_g6_generation_enumerates_only_five_frozen_blocks(self) -> None:
    g6_cohort = (
        SCRIPT_DIR.parent / "configs" / "catbench_5cat_g6_dryrun_cohort.json"
    )
    args = argparse.Namespace(
        tasks="",
        categories="sms,files,maps,contacts,clock",
        suite_family="android_world",
        n_task_combinations=3,
        task_random_seed=30,
        fixed_task_seed=False,
        limit=0,
        cohort_manifest=str(g6_cohort),
    )
    entries = generator._iter_task_instances(  # pylint: disable=protected-access
        args
    )
    cohort = json.loads(g6_cohort.read_text(encoding="utf-8"))
    expected = {
        (
            str(block["category"]),
            str(block["app_id"]),
            str(block["semantic_task_id"]),
            int(block["instance_id"]),
        )
        for block in cohort["paired_blocks"]
    }

    self.assertEqual(5, len(entries))
    self.assertEqual(5, len({entry["plan_key"] for entry in entries}))
    self.assertEqual(
        expected,
        {
            (
                str(entry["category"]),
                str(entry["app_id"]),
                str(entry["semantic_task_id"]),
                int(entry["instance_id"]),
            )
            for entry in entries
        },
    )

  def test_mocked_retries_preserve_raw_candidates_rejections_and_acceptance(
      self,
  ) -> None:
    item = self.real_entries[0]
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      args = self._args(root)
      audit = self._audit(root, args, [item])
      responses = [
          RuntimeError("Authorization: Bearer unit-test-secret"),
          json.dumps(
              {"steps": ["Tap the Add button to begin."], "notes": []}
          ),
          json.dumps(
              {
                  "steps": ["Establish the requested message outcome."],
                  "notes": [],
              }
          ),
      ]
      with (
          mock.patch.object(
              generator, "_call_provider_once", side_effect=responses
          ) as provider_call,
          mock.patch.object(generator.time, "sleep"),
      ):
        # pylint: disable=protected-access
        breakdown, warnings, repairs = generator._generate_one_plan(
            args=args,
            item=item,
            app_display_names=(str(item["app_display_name"]),),
            resolved_model=args.model,
            gemini_wrapper=None,
            openai_base_url=args.openai_base_url,
            openai_api_key=args.openai_api_key,
            audit=audit,
        )

      self.assertEqual(3, provider_call.call_count)
      self.assertEqual([], warnings)
      self.assertEqual(1, repairs)
      self.assertEqual(
          ["Establish the requested message outcome."], breakdown["steps"]
      )
      records = generator._read_audit_records(  # pylint: disable=protected-access
          Path(args.audit_log)
      )
      self.assertEqual(
          ["request_error", "validation_rejection", "accepted"],
          [
              record["outcome"]["status"]
              for record in records[1:]
              if record["attempt_phase"] == "completed"
          ],
      )
      self.assertEqual(7, len(records))
      rejected = records[4]["outcome"]
      self.assertIn("Tap the Add button", rejected["raw_response"])
      self.assertEqual(
          ["Tap the Add button to begin."],
          rejected["parsed_candidate"]["steps"],
      )
      self.assertEqual(
          "forbidden_detail", rejected["validation_rejection"]["kind"]
      )
      accepted = records[6]["outcome"]
      self.assertEqual(breakdown, accepted["final_accepted_plan"])
      self.assertEqual(64, len(accepted["final_accepted_plan_sha256"]))
      audit_text = Path(args.audit_log).read_text(encoding="utf-8")
      self.assertNotIn("unit-test-secret", audit_text)
      self.assertIn("[REDACTED]", audit_text)

  def test_full_generator_calls_once_for_shared_real_semantic_instance(
      self,
  ) -> None:
    first, second = self.real_shared_pair
    self.assertNotEqual(first["app_id"], second["app_id"])
    self.assertEqual(first["plan_key"], second["plan_key"])
    accepted_text = json.dumps(
        {"steps": ["Complete the requested semantic outcome."], "notes": []}
    )
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      args = self._args(root, max_retry=1)
      with (
          mock.patch.object(
              generator,
              "_iter_task_instances",
              return_value=[first, second],
          ),
          mock.patch.object(
              generator, "_call_provider_once", return_value=accepted_text
          ) as provider_call,
      ):
        generator.generate(args)

      self.assertEqual(1, provider_call.call_count)
      payload = json.loads(Path(args.output).read_text(encoding="utf-8"))
      self.assertEqual(2, len(payload["breakdowns"]))
      self.assertEqual(
          1, len({entry["plan_sha256"] for entry in payload["breakdowns"]})
      )
      records = generator._read_audit_records(  # pylint: disable=protected-access
          Path(args.audit_log)
      )
      self.assertEqual(3, len(records))
      self.assertEqual("accepted", records[2]["outcome"]["status"])

      resume_args = self._args(root, max_retry=1, resume=True)
      with (
          mock.patch.object(
              generator,
              "_iter_task_instances",
              return_value=[first, second],
          ),
          mock.patch.object(generator, "_call_provider_once") as no_call,
      ):
        generator.generate(resume_args)
      no_call.assert_not_called()

  def test_primary_generation_requires_new_audit_before_provider_call(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      args = self._args(root, audit_log="")
      with (
          mock.patch.object(
              generator,
              "_iter_task_instances",
              return_value=self.real_shared_pair,
          ),
          mock.patch.object(generator, "_call_provider_once") as provider_call,
      ):
        with self.assertRaisesRegex(ValueError, "requires --audit_log"):
          generator.generate(args)
      provider_call.assert_not_called()

    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      args = self._args(root, overwrite=True)
      with mock.patch.object(
          generator, "_call_provider_once"
      ) as provider_call:
        with self.assertRaisesRegex(ValueError, "forbids --overwrite"):
          generator.generate(args)
      provider_call.assert_not_called()

  def test_frozen_qwen_requires_model_identity_and_endpoint_before_call(
      self,
  ) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      args = self._args(
          root,
          provider="qwen",
          model="catbench-judge",
          model_identity="  ",
          openai_base_url="https://qwen.example/v1",
      )
      with (
          mock.patch.object(
              generator,
              "_iter_task_instances",
              return_value=self.real_shared_pair,
          ),
          mock.patch.object(generator, "_call_provider_once") as provider_call,
      ):
        with self.assertRaisesRegex(ValueError, "--model_identity"):
          generator.generate(args)
      provider_call.assert_not_called()
      self.assertFalse(Path(args.output).exists())
      self.assertFalse(Path(args.audit_log).exists())

    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      args = self._args(
          root,
          provider="qwen",
          model="catbench-judge",
          model_identity="Qwen/Qwen3-VL-30B-A3B-Instruct",
          openai_base_url="",
      )
      with (
          mock.patch.dict(
              generator.os.environ, {"QWEN_C2_BASE_URL": ""}
          ),
          mock.patch.object(
              generator,
              "_iter_task_instances",
              return_value=self.real_shared_pair,
          ),
          mock.patch.object(generator, "_call_provider_once") as provider_call,
      ):
        with self.assertRaisesRegex(ValueError, "QWEN_C2_BASE_URL"):
          generator.generate(args)
      provider_call.assert_not_called()
      self.assertFalse(Path(args.output).exists())
      self.assertFalse(Path(args.audit_log).exists())

  def test_qwen_model_identity_is_bound_to_audit_output_and_entries(self) -> None:
    first, second = self.real_shared_pair
    accepted_text = json.dumps(
        {"steps": ["Complete the requested semantic outcome."], "notes": []}
    )
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      args = self._args(
          root,
          provider="qwen",
          model="catbench-judge",
          model_identity="Qwen/Qwen3-VL-30B-A3B-Instruct",
          openai_base_url="https://qwen.example/v1",
          openai_api_key="qwen-private-test-secret",
          max_retry=1,
      )
      with (
          mock.patch.object(
              generator,
              "_iter_task_instances",
              return_value=[first, second],
          ),
          mock.patch.object(
              generator, "_call_provider_once", return_value=accepted_text
          ),
      ):
        generator.generate(args)

      payload = json.loads(Path(args.output).read_text(encoding="utf-8"))
      identity = "Qwen/Qwen3-VL-30B-A3B-Instruct"
      self.assertEqual(
          identity, payload["metadata"]["generator_model_identity"]
      )
      self.assertEqual(
          {identity},
          {
              entry["generator_model_identity"]
              for entry in payload["breakdowns"]
          },
      )
      records = generator._read_audit_records(  # pylint: disable=protected-access
          Path(args.audit_log)
      )
      binding = records[0]["binding"]
      self.assertEqual(identity, binding["model_identity"])
      self.assertEqual(
          identity, binding["generator_config"]["model_identity"]
      )
      self.assertNotIn(
          args.openai_api_key,
          Path(args.audit_log).read_text(encoding="utf-8"),
      )

  def test_audit_is_exclusive_hash_linked_and_binding_locked(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      args = self._args(root)
      audit = self._audit(root, args)
      with self.assertRaisesRegex(FileExistsError, "already exists"):
        generator.AttemptAuditLog.create(
            Path(args.audit_log), audit.binding, (args.openai_api_key,)
        )

      drifted = dict(audit.binding)
      drifted["model"] = "gemini-3.1-pro-preview"
      with self.assertRaisesRegex(ValueError, "binding does not match"):
        generator.AttemptAuditLog.resume(Path(args.audit_log), drifted)

      lines = Path(args.audit_log).read_text(encoding="utf-8").splitlines()
      header = json.loads(lines[0])
      header["security_policy"] = "tampered"
      Path(args.audit_log).write_text(
          json.dumps(header, sort_keys=True) + "\n", encoding="utf-8"
      )
      with self.assertRaisesRegex(ValueError, "hash mismatch"):
        generator.AttemptAuditLog.resume(
            Path(args.audit_log), audit.binding
        )

  def test_resume_rejects_unresolved_attempt_and_legacy_output(self) -> None:
    item = self.real_entries[0]
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      args = self._args(root)
      audit = self._audit(root, args, [item])
      request = generator._safe_request_payload(  # pylint: disable=protected-access
          args,
          args.model,
          generator._prompt_for_goal(item["semantic_goal"]),  # pylint: disable=protected-access
          "initial",
          0,
          1,
          1,
      )
      generator._append_attempt_started(  # pylint: disable=protected-access
          audit, item=item, request=request
      )
      audit.close()
      with self.assertRaisesRegex(RuntimeError, "unresolved semantic instance"):
        generator.AttemptAuditLog.resume(
            Path(args.audit_log), audit.binding, (args.openai_api_key,)
        )

    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      args = self._args(root)
      audit = self._audit(root, args, [item])
      with self.assertRaisesRegex(RuntimeError, "Refusing to retrofit"):
        generator._verify_output_audit_continuity(  # pylint: disable=protected-access
            payload={"metadata": {}, "breakdowns": []},
            output_preexisted=True,
            task_items=[item],
            audit=audit,
        )


if __name__ == "__main__":
  unittest.main()
