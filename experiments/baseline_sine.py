"""Reproduce the capped Sine baseline and verify uncapped rule growth."""
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import river
from river.datasets import synth

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe

OUT = ROOT / "results" / "baseline_sine"
SEED = 42
PARAMETERS = dict(safe_threshold=0.07, min_firing_for_update=0.05,
                  tau_merge=0.62, drift_sensitivity=0.35)


def sine_data(size):
    # Match the original experiment: first-seen label encoding and stream-wide min-max.
    rows, labels = zip(*synth.Sine(seed=SEED).take(size))
    features = tuple(rows[0])
    x = np.array([[row[key] for key in features] for row in rows], dtype=float)
    codes = {label: i for i, label in enumerate(dict.fromkeys(labels))}
    span = np.ptp(x, axis=0)
    span[span == 0] = 1
    return (x - x.min(axis=0)) / span, np.array([codes[y] for y in labels])


def run(x, y, limit):
    model = EvolvingNeuroFuzzyUnimSafe(x.shape[1], len(np.unique(y)),
                                     max_rules=limit, **PARAMETERS)
    predictions = np.empty_like(y)
    rules = np.empty_like(y)
    start = perf_counter()
    for i in range(len(y)):
        sample = x[i:i + 1]
        predictions[i] = model.predict(sample)[0] if i else 0
        model.partial_fit(sample, y[i:i + 1])
        rules[i] = model.n_rules_
    return predictions, rules, perf_counter() - start


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    fig, ax = plt.subplots(constrained_layout=True)
    for name, size, limit in (("capped_full", 40000, 5),
                              ("capped_quick", 5000, 5),
                              ("uncapped_quick", 5000, None)):
        x, y = sine_data(size)
        predictions, rules, elapsed = run(x, y, limit)
        x_repeat, y_repeat = sine_data(size)
        predictions_repeat, rules_repeat, repeat_elapsed = run(x_repeat, y_repeat, limit)
        np.testing.assert_array_equal(x, x_repeat)
        np.testing.assert_array_equal(y, y_repeat)
        np.testing.assert_array_equal(predictions, predictions_repeat)
        np.testing.assert_array_equal(rules, rules_repeat)
        samples = np.arange(1, size + 1)
        accuracy = np.cumsum(predictions == y) / samples
        with (OUT / f"{name}.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["sample", "prediction", "actual", "rules_after_update",
                             "cumulative_accuracy"])
            writer.writerows(zip(samples, predictions, y, rules, accuracy))
        result = dict(run=name, samples=size, max_rules=limit, accuracy=float(accuracy[-1]),
                      final_rules=int(rules[-1]), peak_rules=int(rules.max()),
                      seconds=elapsed, repeat_seconds=repeat_elapsed, exact_repeat=True)
        summary.append(result)
        print(result, flush=True)
        if size == 5000:
            ax.step(samples, rules, where="post", label=f"max_rules={limit}")
    ax.set(xlabel="Observation", ylabel="Active rules after update",
           title="UnimNeuron-Safe on Sine_synth (seed 42)")
    ax.legend()
    fig.savefig(OUT / "rule_evolution.png", dpi=160)
    plt.close(fig)
    reference_path = ROOT / "experiments" / "results_repro" / "repro_summary.csv"
    with reference_path.open(newline="") as file:
        reference = next(row for row in csv.DictReader(file)
                         if row["dataset"] == "Sine_synth" and row["model"] == "ENF_Unim_Safe")
    report = dict(dataset="Sine_synth", seed=SEED, parameters=PARAMETERS,
                  python=platform.python_version(), numpy=np.__version__,
                  river=river.__version__, matplotlib=matplotlib.__version__,
                  revision=subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                   cwd=ROOT, text=True).strip(),
                  preprocessing="First-seen labels; stream-wide min-max scaling",
                  protocol="Predict before update; first prediction is encoded class 0",
                  historical_capped_accuracy=float(reference["acc_final"]),
                  historical_capped_final_rules=int(reference["rules_final"]), runs=summary)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
