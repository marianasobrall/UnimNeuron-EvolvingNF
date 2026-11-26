"""Check that rule-memory replay respects sample boundaries and live rule IDs."""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe
from src.instrumented import InstrumentedSafe
from src.memory_metrics import (PhaseActivationAccumulator, RuleMemory, compare_snapshots,
                                dominant_activation_by_phase, rule_snapshots)

EVENT_FIELDS = ("sample", "event", "rule_id", "related_rule_id", "previous_class",
                "rule_class", "support", "center", "sigma", "class_counts")


class MemoryMetricsTests(unittest.TestCase):
    def test_replay_matches_live_model_after_each_sample(self):
        rng = np.random.default_rng(31)
        x = rng.uniform(size=(80, 2))
        y = (x[:, 0] > x[:, 1]).astype(int)
        events = []
        original = EvolvingNeuroFuzzyUnimSafe(2, 2, max_rules=3)
        observed = InstrumentedSafe(2, 2, max_rules=3, on_event=events.append)
        live = {}
        for t, (row, label) in enumerate(zip(x, y), start=1):
            xi, yi = row[None, :], np.array([label])
            np.testing.assert_array_equal(original.predict(xi), observed.predict(xi))
            original.partial_fit(xi, yi)
            observed.partial_fit(xi, yi)
            self.assertEqual(original.n_rules_, observed.n_rules_)
            live[t] = {observed._ids[id(rule)]: RuleMemory(
                tuple(rule.center), rule.sigma, rule.support) for rule in observed.rules}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=EVENT_FIELDS)
                writer.writeheader()
                writer.writerows(events)
            replayed = rule_snapshots(path, [0, *range(1, 81)])
        self.assertEqual(replayed[0], {})
        for t in live:
            self.assertEqual(replayed[t], live[t], f"Rule state differs after sample {t}")

    def test_comparison_keeps_births_and_removals_separate(self):
        old = {4: RuleMemory((0.0, 1.0), 2.0, 3),
               7: RuleMemory((1.0, 1.0), 1.0, 5)}
        new = {4: RuleMemory((3.0, 5.0), 1.0, 6),
               9: RuleMemory((0.0, 0.0), 1.0, 1)}
        rows = {row["rule_id"]: row for row in compare_snapshots(old, new)}
        self.assertEqual(rows[4]["center_l2"], 5.0)
        self.assertEqual(rows[4]["sigma_ratio"], 0.5)
        self.assertEqual(rows[4]["support_delta"], 3)
        self.assertEqual(rows[7]["status"], "removed")
        self.assertEqual(rows[7]["center_l2"], "")
        self.assertEqual(rows[9]["status"], "born")

    def test_dominance_uses_phase_size_and_ignores_missing_first_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=("sample", "dominant_rule_id",
                                                         "dominant_activation"))
                writer.writeheader()
                writer.writerows(({"sample": 1, "dominant_rule_id": "", "dominant_activation": ""},
                                  {"sample": 2, "dominant_rule_id": 4, "dominant_activation": 0.4},
                                  {"sample": 3, "dominant_rule_id": 4, "dominant_activation": 0.8}))
            rows, sizes = dominant_activation_by_phase(path, {"A": (1, 2), "B": (3, 3)})
        self.assertEqual(sizes, {"A": 2, "B": 1})
        self.assertEqual(rows[0]["dominant_fraction"], 0.5)
        self.assertEqual(rows[1]["mean_dominant_activation"], 0.8)

    def test_all_rule_activations_are_normalized_per_prediction(self):
        trace = PhaseActivationAccumulator({"A": (1, 2), "B": (3, 3)})
        for sample, rule_id, activation, dominant in (
                (1, 4, 0.6, 1), (1, 7, 0.4, 0),
                (2, 4, 0.2, 0), (2, 7, 0.8, 1),
                (3, 4, 0.5, 1)):
            trace.on_activation({"sample": sample, "rule_id": rule_id,
                                 "activation": activation, "dominant": dominant})
        rows = {(row["phase"], row["rule_id"]): row for row in trace.rows()}
        self.assertEqual(rows[("A", 4)]["samples_present"], 2)
        self.assertAlmostEqual(rows[("A", 4)]["mean_relative_activation_when_present"], 0.4)
        self.assertEqual(rows[("A", 4)]["dominant_count"], 1)
        self.assertEqual(rows[("B", 4)]["phase_mean_relative_activation"], 1.0)
        self.assertEqual(rows[("B", 4)]["last_dominant_sample"], 3)
        selectivity = trace.phase_selectivity()
        self.assertEqual(selectivity["A"]["samples_with_multiple_rules"], 2)
        self.assertAlmostEqual(selectivity["A"]["mean_winner_to_mean_ratio"], 1.4)
        self.assertNotIn("B", selectivity)  # one active rule provides no selectivity test


if __name__ == "__main__":
    unittest.main()
