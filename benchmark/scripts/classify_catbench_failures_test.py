"""Tests for failure-judge evidence selection."""

from __future__ import annotations

import unittest

import classify_catbench_failures as classifier


class EvidenceSelectionTest(unittest.TestCase):

  def test_zero_step_budget_keeps_full_ordered_trace(self) -> None:
    self.assertEqual(list(range(9)), classifier._step_indices(9, 0))

  def test_zero_smart_step_budget_keeps_full_ordered_trace(self) -> None:
    episode = {
        "episode_data": {
            "action": [f"action {index}" for index in range(7)],
            "thought": [f"thought {index}" for index in range(7)],
        }
    }

    self.assertEqual(
        list(range(7)), classifier._pick_key_step_indices(episode, 0)
    )

  def test_six_step_budget_remains_available_for_ablation(self) -> None:
    self.assertEqual(6, len(classifier._step_indices(20, 6)))


if __name__ == "__main__":
  unittest.main()
