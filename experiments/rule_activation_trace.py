"""Replay the T16 stream to summarize all rules' pre-update activations.

The existing T16 sample CSV is checked observation by observation. This is
slower than the event-log replay in rule_memory.py (about three minutes for
both configurations), but records the activation of every active rule while
keeping only per-phase aggregates in memory.
"""

import csv
import json
import platform
from collections import Counter
from time import perf_counter

import numpy as np
import river

from baseline_sine import PARAMETERS, ROOT
from drift_agrawal import agrawal_drift_data
from rule_memory import OUT, PHASES, SOURCE, write_rows
from src.instrumented import InstrumentedSafe
from src.memory_metrics import PhaseActivationAccumulator

FIELDS = ("phase", "rule_id", "samples_present",
          "mean_raw_activation_when_present", "mean_relative_activation_when_present",
          "phase_mean_relative_activation", "dominant_count",
          "dominant_fraction_when_present", "dominant_fraction_of_phase",
          "last_dominant_sample")


def trace(run, limit, x, y):
    accumulator = PhaseActivationAccumulator(PHASES)
    expected_winners = Counter()
    model = InstrumentedSafe(x.shape[1], 2, max_rules=limit,
                             on_activation=accumulator.on_activation, **PARAMETERS)
    started = perf_counter()
    with (SOURCE / f"{run}_samples.csv").open(newline="", encoding="utf-8") as file:
        reference = csv.DictReader(file)
        for index, (features, label) in enumerate(zip(x, y), start=1):
            expected = next(reference, None)
            if expected is None or int(expected["sample"]) != index or int(expected["actual"]) != label:
                raise AssertionError(f"{run}: source stream differs at observation {index}")
            if expected["dominant_rule_id"]:
                phase = next(name for name, (first, last) in PHASES.items() if first <= index <= last)
                expected_winners[(phase, int(expected["dominant_rule_id"]))] += 1
            prediction = int(model.predict(features[None, :])[0])
            model.partial_fit(features[None, :], y[index - 1:index])
            if (prediction != int(expected["prediction"]) or
                    model.n_rules_ != int(expected["rules_after_update"])):
                raise AssertionError(f"{run}: model diverges at observation {index}")
        if next(reference, None) is not None:
            raise AssertionError(f"{run}: source log has extra observations")

    rows = accumulator.rows()
    actual_winners = {(row["phase"], row["rule_id"]): row["dominant_count"]
                      for row in rows if row["dominant_count"]}
    if actual_winners != dict(expected_winners):
        raise AssertionError(f"{run}: dominant rule IDs differ from T16")
    write_rows(OUT / f"{run}_all_activations.csv", FIELDS, rows)
    # Relative weights across active rules sum to one on each prediction with
    # positive total activation. The first prediction has no rules yet.
    phase_weight_sums = {phase: sum(row["phase_mean_relative_activation"]
                                    for row in rows if row["phase"] == phase)
                         for phase in PHASES}
    for phase, weight_sum in phase_weight_sums.items():
        expected = 1 - (1 / (PHASES[phase][1] - PHASES[phase][0] + 1)
                        if phase == "before" else 0)
        if abs(weight_sum - expected) > 1e-9:
            raise AssertionError(f"{run}: normalized activations do not sum to {expected}")
    return {"samples": len(y), "exact_match_to_T16": True,
            "final_rules": model.n_rules_, "seconds": round(perf_counter() - started, 1),
            "phase_weight_sums": phase_weight_sums,
            "activation_selectivity": accumulator.phase_selectivity(),
            "rule_phase_rows": len(rows)}


def plot_activation(runs):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    phases = tuple(PHASES)
    for column, (run, rows) in enumerate(runs.items()):
        by_id = {}
        for row in rows:
            by_id.setdefault(row["rule_id"], {})[row["phase"]] = row
        stable = [rule_id for rule_id, observations in by_id.items()
                  if all(phase in observations for phase in phases)]
        top = sorted(stable,
                     key=lambda rule_id: by_id[rule_id]["before"]["dominant_count"],
                     reverse=True)[:3]
        for rule_id in top:
            axes[0, column].plot(phases,
                                 [by_id[rule_id][phase]["mean_raw_activation_when_present"]
                                  for phase in phases], marker="o", label=f"rule {rule_id}")
            axes[1, column].plot(phases,
                                 [by_id[rule_id][phase]["dominant_fraction_when_present"]
                                  for phase in phases], marker="o", label=f"rule {rule_id}")
        axes[0, column].set_title("max_rules=5" if run == "capped" else "no rule limit")
        axes[0, column].set_ylabel("Mean raw activation when present")
        axes[1, column].set_ylabel("Fraction dominant when present")
        axes[1, column].set_xlabel("Stream phase")
        for ax in axes[:, column]:
            ax.set_ylim(0, 1)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Three pre-drift rules: activation and dominance by phase")
    fig.savefig(OUT / "all_rule_activations.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, _ = agrawal_drift_data()
    report = {"stream": "ConceptDrift_Agrawal functions 0->2",
              "phases": PHASES,
              "versions": {"python": platform.python_version(), "numpy": np.__version__,
                           "river": river.__version__},
              "semantics": "all active rules, pre-update; relative activation is raw activation divided by the sample total",
              "runs": {}}
    plots = {}
    for run, limit in (("capped", 5), ("uncapped", None)):
        report["runs"][run] = trace(run, limit, x, y)
        with (OUT / f"{run}_all_activations.csv").open(newline="", encoding="utf-8") as file:
            plots[run] = [{"phase": row["phase"], "rule_id": int(row["rule_id"]),
                           "dominant_count": int(row["dominant_count"]),
                           "mean_raw_activation_when_present": float(row["mean_raw_activation_when_present"]),
                           "dominant_fraction_when_present": float(row["dominant_fraction_when_present"])}
                          for row in csv.DictReader(file)]
        print(run, report["runs"][run], flush=True)
    plot_activation(plots)
    (OUT / "all_activations_summary.json").write_text(json.dumps(report, indent=2) + "\n",
                                                      encoding="utf-8")


if __name__ == "__main__":
    main()
