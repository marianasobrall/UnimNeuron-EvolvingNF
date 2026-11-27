"""Change which rule is removed when a candidate exceeds the budget."""

import numpy as np

from src.instrumented import InstrumentedSafe

POLICIES = ("support", "lru", "fifo")


class RetentionSafe(InstrumentedSafe):
    def __init__(self, *args, policy="support", **kwargs):
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy: {policy}")
        super().__init__(*args, **kwargs)
        self.policy = policy
        self._last_dominant = {}

    def _compute_rule_activations_and_dists(self, x):
        activations, distances = super()._compute_rule_activations_and_dists(x)
        if self.policy == "lru" and self._predicting and activations.size:
            winner = int(np.argmax(activations))
            if activations[winner] > 0:
                self._last_dominant[self._ids[id(self.rules[winner])]] = self.K + 1
        return activations, distances

    def _update_one(self, x, y):
        super()._update_one(x, y)
        if self.policy == "lru":
            self._last_dominant = {
                rid: self._last_dominant[rid]
                for rid in self._ids.values() if rid in self._last_dominant
            }

    def _eviction_index(self):
        if self.policy == "support":
            return super()._eviction_index()
        residents = [self._ids[id(rule)] for rule in self.rules[:-1]]
        if not residents:
            return 0
        if self.policy == "fifo":
            return min(range(len(residents)), key=residents.__getitem__)
        return min(range(len(residents)), key=lambda i: (
            self._last_dominant.get(residents[i], 0), residents[i]
        ))
