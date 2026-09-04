"""Endpoint contract tests for the UI-TARS/MAI runner."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
from unittest import mock

try:
  from benchmark import endpoint_contract
  from benchmark import run_maiui
except ModuleNotFoundError:
  import endpoint_contract
  import run_maiui


class EndpointContractTest(unittest.TestCase):

  def test_context_requirement_accepts_sufficient_endpoint(self):
    self.assertTrue(
        endpoint_contract.context_compatible(
            {"max_model_len": 16384}, 16384
        )
    )

  def test_context_requirement_rejects_small_or_missing_endpoint(self):
    for record in ({"max_model_len": 8192}, {}, {"max_model_len": "16384"}):
      with self.subTest(record=record):
        self.assertFalse(
            endpoint_contract.context_compatible(
                record, 16384
            )
        )

  def test_zero_requirement_allows_legacy_endpoint(self):
    self.assertTrue(
        endpoint_contract.context_compatible(
            {}, 0
        )
    )

  def test_loopback_contract_accepts_only_explicit_loopback_hosts(self):
    for url in (
        "http://127.0.0.1:8000",
        "http://127.9.8.7:8000/v1",
        "http://localhost:8000",
        "http://[::1]:8000",
    ):
      with self.subTest(url=url):
        endpoint_contract.require_loopback(url)

    for url in (
        "http://0.0.0.0:8000",
        "http://198.51.100.4:8000",
        "https://models.example:8000",
        "ftp://127.0.0.1:8000",
        "http://user@127.0.0.1:8000",
    ):
      with self.subTest(url=url):
        with self.assertRaises(endpoint_contract.EndpointContractError):
          endpoint_contract.require_loopback(url)

  def test_exact_model_identity_and_context_are_both_required(self):
    records = endpoint_contract.parse_model_records({
        "data": [
            {"id": "mPLUG/GUI-Owl-7B", "max_model_len": 16384},
        ]
    })
    record = endpoint_contract.validate_model_record(
        records, "mPLUG/GUI-Owl-7B", 16384
    )
    self.assertEqual(record["max_model_len"], 16384)

    with self.assertRaisesRegex(
        endpoint_contract.EndpointContractError, "exact model id"
    ):
      endpoint_contract.validate_model_record(records, "gui-owl-7b", 16384)
    with self.assertRaisesRegex(
        endpoint_contract.EndpointContractError, "max_model_len"
    ):
      endpoint_contract.validate_model_record(
          {"mPLUG/GUI-Owl-7B": {"max_model_len": 8192}},
          "mPLUG/GUI-Owl-7B",
          16384,
      )

  def test_malformed_or_duplicate_model_records_fail_closed(self):
    invalid_payloads = (
        [],
        {},
        {"data": [{}]},
        {"data": ["model"]},
        {"data": [{"id": "m"}, {"id": "m"}]},
    )
    for payload in invalid_payloads:
      with self.subTest(payload=payload):
        with self.assertRaises(endpoint_contract.EndpointContractError):
          endpoint_contract.parse_model_records(payload)

  @mock.patch.object(endpoint_contract, "fetch_model_records")
  def test_wait_fails_immediately_on_reachable_wrong_model(self, fetch):
    fetch.return_value = {"wrong": {"id": "wrong", "max_model_len": 32768}}
    with self.assertRaisesRegex(
        endpoint_contract.EndpointContractError, "exact model id"
    ):
      endpoint_contract.wait_for_model(
          "http://127.0.0.1:8000",
          "expected",
          16384,
          timeout_sec=1800,
          poll_sec=5,
          loopback_only=True,
      )
    fetch.assert_called_once()


class ProspectiveModelConfigTest(unittest.TestCase):

  def test_all_five_models_pin_exact_local_endpoint_contract(self):
    config_path = (
        Path(__file__).resolve().parent / "configs" / "catbench_5cat_models.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    by_name = {record["name"]: record for record in payload["models"]}
    expected = {
        "UI-Venus-7B": "inclusionAI/UI-Venus-Navi-7B",
        "GUI-Owl-7B": "mPLUG/GUI-Owl-7B",
        "MAI-UI-8B": "Tongyi-MAI/MAI-UI-8B",
        "UI Voyager-4B": "UI-Voyager",
        "Qwen3-VL-8B": "Qwen/Qwen3-VL-8B-Instruct",
    }
    for model_name, served_id in expected.items():
      with self.subTest(model=model_name):
        args = by_name[model_name]["args"]
        self.assertIn("--endpoint_require_loopback=true", args)
        self.assertIn(f"--model_name={served_id}", args)
        self.assertIn("--endpoint_min_context_len=16384", args)

  @mock.patch.object(run_maiui.subprocess, "call", return_value=0)
  @mock.patch.object(
      run_maiui, "_check_agent_dependency", return_value=(True, "test backend")
  )
  def test_mai_wrapper_defaults_to_frozen_endpoint_contract(
      self, _dependency, subprocess_call
  ):
    with mock.patch.object(run_maiui.sys, "argv", ["run_maiui.py"]):
      self.assertEqual(run_maiui.main(), 0)
    command = subprocess_call.call_args.args[0]
    self.assertIn("--endpoint_min_context_len=16384", command)
    self.assertIn("--endpoint_require_loopback=true", command)


if __name__ == "__main__":
  unittest.main()
