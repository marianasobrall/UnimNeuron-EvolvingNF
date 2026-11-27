"""Regression checks for the three budget policies."""
import numpy as np
import pytest

from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe
from src.eviction import RetentionSafe


@pytest.mark.parametrize(("policy", "expected_ids"), [
    ("support", [1, 2]),
    ("fifo", [2, 3]),
    ("lru", [1, 3]),
])
def test_budget_selects_expected_victim(policy, expected_ids):
    rows = []
    model = RetentionSafe(1, 2, max_rules=2, policy=policy, on_event=rows.append)
    model._create_rule(np.array([0.0]), 0, 0.1)
    model._create_rule(np.array([1.0]), 1, 0.1)
    for rule in model.rules:
        rule.support = 3
    model.predict(np.array([[0.0]]))
    model._create_rule(np.array([0.5]), 1, 0.1)
    assert [model._ids[id(rule)] for rule in model.rules] == expected_ids
    assert [row["rule_id"] for row in rows if row["event"] == "evict"] == [
        ({1, 2, 3} - set(expected_ids)).pop()
    ]


def test_lru_without_use_evicts_oldest_resident():
    model = RetentionSafe(1, 2, max_rules=2, policy="lru")
    for value in (0.0, 0.5, 1.0):
        model._create_rule(np.array([value]), 0, 0.1)
    assert [model._ids[id(rule)] for rule in model.rules] == [2, 3]


@pytest.mark.parametrize("policy", ["support", "fifo", "lru"])
def test_uncapped_matches_original(policy):
    rng = np.random.default_rng(42)
    models = (RetentionSafe(2, 2, policy=policy), EvolvingNeuroFuzzyUnimSafe(2, 2))
    for row in rng.random((100, 2)):
        sample = row[None, :]
        label = np.array([int(row[0] > row[1])])
        np.testing.assert_array_equal(*(model.predict(sample) for model in models))
        for model in models:
            model.partial_fit(sample, label)
        assert models[0].n_rules_ == models[1].n_rules_


@pytest.mark.parametrize("policy", ["support", "fifo", "lru"])
def test_zero_budget_retains_no_rule(policy):
    model = RetentionSafe(1, 2, max_rules=0, policy=policy)
    model.partial_fit(np.array([[0.0], [1.0]]), np.array([0, 1]))
    assert model.n_rules_ == 0
