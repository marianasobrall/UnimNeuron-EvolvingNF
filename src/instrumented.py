"""Observe the original Safe model through its existing update methods."""

import numpy as np

from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe


class _ObservedRules(list):
    """Capture insertion and eviction, including a candidate evicted immediately."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def append(self, rule):
        super().append(rule)
        self.model._register_rule(rule)

    def __delitem__(self, index):
        rule = self[index]
        super().__delitem__(index)
        self.model._emit("evict", rule)
        del self.model._ids[id(rule)]


class InstrumentedSafe(EvolvingNeuroFuzzyUnimSafe):
    """Log rule events and prediction activations without changing the base model."""

    def __init__(self, *args, on_event=None, on_activation=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_event = on_event
        self._on_activation = on_activation
        self._ids = {}
        self._next_id = 1
        self._predicting = False
        self.rules = _ObservedRules(self)

    def _register_rule(self, rule):
        self._ids[id(rule)] = self._next_id
        self._next_id += 1
        self._emit("create", rule)

    def _emit(self, event, rule, *, related_rule_id=None, previous_class=None):
        if self._on_event is None:
            return
        self._on_event({
            "sample": self.K,
            "event": event,
            "rule_id": self._ids[id(rule)],
            "related_rule_id": related_rule_id,
            "previous_class": previous_class,
            "rule_class": int(np.argmax(rule.class_counts)),
            "support": rule.support,
            "center": rule.center.tolist(),
            "sigma": rule.sigma,
            "class_counts": rule.class_counts.tolist(),
        })

    def _maybe_merge_last_rule(self):
        if not self.rules:
            return super()._maybe_merge_last_rule()

        absorbed = self.rules[-1]
        if self._on_event is not None:
            before = [(rule.support, int(np.argmax(rule.class_counts))) for rule in self.rules]
        count = len(self.rules)
        super()._maybe_merge_last_rule()
        if len(self.rules) < count:
            if self._on_event is not None:
                survivor, old_class = next(
                    (rule, old_class) for rule, (support, old_class) in zip(self.rules, before)
                    if rule.support != support
                )
                absorbed_id = self._ids[id(absorbed)]
                survivor_id = self._ids[id(survivor)]
                self._emit("merge", survivor, related_rule_id=absorbed_id)
                self._emit("absorbed", absorbed, related_rule_id=survivor_id)
                if old_class != int(np.argmax(survivor.class_counts)):
                    self._emit("class_change", survivor, previous_class=old_class)
            del self._ids[id(absorbed)]

    def _update_one(self, x, y):
        if self._on_event is None:
            return super()._update_one(x, y)

        self._supports_before_update = {
            self._ids[id(rule)]: rule.support for rule in self.rules
        }
        self._next_id_before_update = self._next_id
        super()._update_one(x, y)

    def _safe_update_consequents(self, x, y):
        if self._on_event is None:
            return super()._safe_update_consequents(x, y)

        # The base update either adapts an antecedent or creates a candidate.
        if self._next_id == self._next_id_before_update:
            for rule in self.rules:
                if rule.support != self._supports_before_update[self._ids[id(rule)]]:
                    self._emit("adapt", rule)

        before = [rule.class_counts.copy() for rule in self.rules]
        super()._safe_update_consequents(x, y)
        for rule, old_counts in zip(self.rules, before):
            if np.array_equal(rule.class_counts, old_counts):
                continue
            self._emit("update_consequent", rule)
            old_class = int(np.argmax(old_counts))
            if old_class != int(np.argmax(rule.class_counts)):
                self._emit("class_change", rule, previous_class=old_class)

    def _predict_one(self, x):
        self._predicting = True
        try:
            return super()._predict_one(x)
        finally:
            self._predicting = False

    def _compute_rule_activations_and_dists(self, x):
        activations, distances = super()._compute_rule_activations_and_dists(x)
        if self._predicting and self._on_activation is not None and activations.size:
            winner = int(np.argmax(activations)) if np.any(activations > 0) else -1
            for index, (rule, activation) in enumerate(zip(self.rules, activations)):
                self._on_activation({
                    "sample": self.K + 1,
                    "rule_id": self._ids[id(rule)],
                    "activation": float(activation),
                    "dominant": int(index == winner),
                })
        return activations, distances
