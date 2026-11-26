"""Passive rule-memory measurements from the instrumented Safe model's logs.

The event log is ordered by sample and then by event. Its create/adapt/merge
rows contain the post-event antecedent; evict/absorbed remove a rule. This
reconstructs the rule base at an observation boundary without retraining.
"""

import csv
import gzip
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from math import dist, isfinite, log


@dataclass(frozen=True)
class RuleMemory:
    center: tuple[float, ...]
    sigma: float
    support: int

    @classmethod
    def from_event(cls, row):
        center = tuple(float(value) for value in json.loads(row["center"]))
        sigma = float(row["sigma"])
        support = int(row["support"])
        if not center or not all(map(isfinite, center)) or not isfinite(sigma) or sigma <= 0:
            raise ValueError(f"Invalid rule geometry at sample {row['sample']}")
        if support < 1:
            raise ValueError(f"Invalid rule support at sample {row['sample']}")
        return cls(center, sigma, support)


def _open_csv(path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if str(path).endswith(".gz") else open(
        path, "r", encoding="utf-8", newline="")


def rule_snapshots(event_path, checkpoints):
    """Return active rule states after each requested training observation.

    Checkpoints with no event still receive a snapshot. Updates to consequents
    and class-change events do not alter center, sigma or support.
    """
    checkpoints = sorted(set(int(t) for t in checkpoints))
    if not checkpoints or checkpoints[0] < 0:
        raise ValueError("Provide non-negative observation checkpoints")
    active = {}
    snapshots = {}
    last_sample = 0
    next_checkpoint = 0
    with _open_csv(event_path) as file:
        for row in csv.DictReader(file):
            sample = int(row["sample"])
            if sample < last_sample:
                raise ValueError("Events must be ordered by sample")
            while next_checkpoint < len(checkpoints) and checkpoints[next_checkpoint] < sample:
                snapshots[checkpoints[next_checkpoint]] = active.copy()
                next_checkpoint += 1
            rule_id = int(row["rule_id"])
            event = row["event"]
            if event == "create":
                if rule_id in active:
                    raise ValueError(f"Rule {rule_id} created twice")
                active[rule_id] = RuleMemory.from_event(row)
            elif event in ("adapt", "merge"):
                if rule_id not in active:
                    raise ValueError(f"Unknown rule {rule_id} updated")
                active[rule_id] = RuleMemory.from_event(row)
            elif event in ("evict", "absorbed"):
                if rule_id not in active:
                    raise ValueError(f"Unknown rule {rule_id} removed")
                del active[rule_id]
            elif event not in ("update_consequent", "class_change"):
                raise ValueError(f"Unknown event: {event}")
            last_sample = sample
    while next_checkpoint < len(checkpoints):
        snapshots[checkpoints[next_checkpoint]] = active.copy()
        next_checkpoint += 1
    return snapshots


def compare_snapshots(before, after):
    """Compare geometry only for the same persistent rule IDs.

    New/removed rules have no meaningful parameter delta and are reported
    separately. A parameter shift describes changed representation, not by
    itself whether a previous concept was forgotten.
    """
    rows = []
    for rule_id in sorted(before.keys() | after.keys()):
        old, new = before.get(rule_id), after.get(rule_id)
        row = {"rule_id": rule_id,
               "status": "survived" if old and new else "removed" if old else "born",
               "center_l2": "", "sigma_before": "", "sigma_after": "",
               "sigma_ratio": "", "support_before": "", "support_after": "",
               "support_delta": ""}
        if old:
            row.update(sigma_before=old.sigma, support_before=old.support)
        if new:
            row.update(sigma_after=new.sigma, support_after=new.support)
        if old and new:
            if len(old.center) != len(new.center):
                raise ValueError(f"Rule {rule_id} changed feature dimension")
            row.update(center_l2=dist(old.center, new.center),
                       sigma_ratio=new.sigma / old.sigma,
                       support_delta=new.support - old.support)
        rows.append(row)
    return rows


def dominant_activation_by_phase(samples_path, phases):
    """Summarize pre-update winning-rule activations by known stream phase.

    The existing drift CSV stores only the winning rule and its raw activation.
    These are dominance counts, not counts of all rules firing and not a
    prediction-contribution or prediction-correctness measure.
    """
    phases = dict(phases)
    sizes = Counter()
    counts = Counter()
    activation_sums = defaultdict(float)
    expected_sample = 1
    with _open_csv(samples_path) as file:
        for row in csv.DictReader(file):
            sample = int(row["sample"])
            if sample != expected_sample:
                raise ValueError(f"Expected observation {expected_sample}, found {sample}")
            expected_sample += 1
            matches = [name for name, (first, last) in phases.items() if first <= sample <= last]
            if len(matches) != 1:
                raise ValueError(f"Observation {sample} belongs to {len(matches)} phases")
            phase = matches[0]
            sizes[phase] += 1
            rule_id = row["dominant_rule_id"]
            if rule_id:
                key = (phase, int(rule_id))
                counts[key] += 1
                activation_sums[key] += float(row["dominant_activation"])
    rows = [{"phase": phase, "rule_id": rule_id,
             "dominant_count": count,
             "dominant_fraction": count / sizes[phase],
             "mean_dominant_activation": activation_sums[(phase, rule_id)] / count}
            for (phase, rule_id), count in sorted(counts.items())]
    return rows, dict(sizes)


class PhaseActivationAccumulator:
    """Aggregate *every* rule's pre-update activation without retaining samples.

    ``InstrumentedSafe`` calls the callback once per active rule, in rule order.
    A one-sample buffer permits normalization by the sum of that sample's raw
    activations. Each rule's stable ID is used even when rule positions change.
    """

    def __init__(self, phases):
        self.phases = dict(phases)
        self._sample = None
        self._buffer = []
        self._aggregates = defaultdict(lambda: {"present": 0, "raw_sum": 0.0,
                                                "relative_sum": 0.0, "dominant": 0,
                                                "last_dominant_sample": None})
        self._phase_selectivity = defaultdict(lambda: {"samples": 0, "entropy_sum": 0.0,
                                                       "winner_to_mean_sum": 0.0})

    def on_activation(self, row):
        sample = int(row["sample"])
        if self._sample is not None and sample < self._sample:
            raise ValueError("Activation samples must be ordered")
        if sample != self._sample:
            self._flush()
            self._sample = sample
        self._buffer.append(row)

    def _flush(self):
        if not self._buffer:
            return
        matches = [name for name, (first, last) in self.phases.items()
                   if first <= self._sample <= last]
        if len(matches) != 1:
            raise ValueError(f"Observation {self._sample} belongs to {len(matches)} phases")
        phase = matches[0]
        values = [float(row["activation"]) for row in self._buffer]
        if any(not isfinite(value) or value < 0 for value in values):
            raise ValueError(f"Invalid activation at observation {self._sample}")
        if sum(int(row["dominant"]) for row in self._buffer) != int(any(v > 0 for v in values)):
            raise ValueError(f"Invalid dominant rule at observation {self._sample}")
        if len({int(row["rule_id"]) for row in self._buffer}) != len(self._buffer):
            raise ValueError(f"Duplicate rule ID at observation {self._sample}")
        total = sum(values)
        if len(values) >= 2 and total > 0:
            weights = [value / total for value in values]
            statistics = self._phase_selectivity[phase]
            statistics["samples"] += 1
            statistics["entropy_sum"] += -sum(weight * log(weight) for weight in weights
                                                if weight > 0) / log(len(weights))
            statistics["winner_to_mean_sum"] += max(values) * len(values) / total
        for row, value in zip(self._buffer, values):
            aggregate = self._aggregates[(phase, int(row["rule_id"]))]
            aggregate["present"] += 1
            aggregate["raw_sum"] += value
            aggregate["relative_sum"] += value / total if total else 0.0
            if int(row["dominant"]):
                aggregate["dominant"] += 1
                aggregate["last_dominant_sample"] = self._sample
        self._buffer.clear()

    def rows(self):
        """Return per-rule phase summaries, including both exposure denominators."""
        self._flush()
        rows = []
        for (phase, rule_id), values in sorted(self._aggregates.items()):
            phase_size = self.phases[phase][1] - self.phases[phase][0] + 1
            present = values["present"]
            rows.append({"phase": phase, "rule_id": rule_id,
                         "samples_present": present,
                         "mean_raw_activation_when_present": values["raw_sum"] / present,
                         "mean_relative_activation_when_present": values["relative_sum"] / present,
                         "phase_mean_relative_activation": values["relative_sum"] / phase_size,
                         "dominant_count": values["dominant"],
                         "dominant_fraction_when_present": values["dominant"] / present,
                         "dominant_fraction_of_phase": values["dominant"] / phase_size,
                         "last_dominant_sample": values["last_dominant_sample"] or ""})
        return rows

    def phase_selectivity(self):
        """Summarize how concentrated the activations are within each sample.

        Normalized entropy is 1 for equal activations, 0 when one rule receives
        all activation. The winner-to-mean ratio is 1 for equal activations.
        Single-rule predictions are excluded from both diagnostics.
        """
        self._flush()
        return {phase: {"samples_with_multiple_rules": values["samples"],
                        "mean_normalized_entropy": values["entropy_sum"] / values["samples"],
                        "mean_winner_to_mean_ratio": values["winner_to_mean_sum"] / values["samples"]}
                for phase, values in self._phase_selectivity.items() if values["samples"]}


def memory_summary(comparisons):
    """Counts and median shifts for a pair of snapshots."""
    import statistics

    survivors = [row for row in comparisons if row["status"] == "survived"]
    return {"survived": len(survivors),
            "born": sum(row["status"] == "born" for row in comparisons),
            "removed": sum(row["status"] == "removed" for row in comparisons),
            "median_center_l2": statistics.median(row["center_l2"] for row in survivors) if survivors else None,
            "median_sigma_ratio": statistics.median(row["sigma_ratio"] for row in survivors) if survivors else None}
