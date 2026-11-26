"""T29 point 3: inspect rule geometry and dominance around the known AGRAWAL drift.

Uses T16's existing passive event and sample logs. No training, predictions or
baseline code are changed. Run ``python experiments/rule_memory.py`` from repo root.
Outputs are written to results/4_rule_memory/.
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.memory_metrics import (compare_snapshots, dominant_activation_by_phase,
                                memory_summary, rule_snapshots)

SOURCE = ROOT / "results" / "3_drift_agrawal"
OUT = ROOT / "results" / "4_rule_memory"
CHECKPOINTS = tuple(sorted(set(range(250, 20001, 250)) | {1999}))
COMPARISONS = {"through_drift": (1999, 3000),
               "early_post_drift": (1999, 3500)}
PHASES = {"before": (1, 1999), "drift": (2000, 3000),
          "after": (3001, 20000)}


def write_rows(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_memory(traces):
    """Show the three most frequently dominant pre-drift rules in each run."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from math import dist

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
    for column, (run, (snapshots, activations)) in enumerate(traces.items()):
        before = snapshots[1999]
        ranked = sorted((row for row in activations
                         if row["phase"] == "before" and row["rule_id"] in before),
                        key=lambda row: row["dominant_count"], reverse=True)[:3]
        for row in ranked:
            rule_id = row["rule_id"]
            times = [t for t in CHECKPOINTS if 1999 <= t <= 4500 and rule_id in snapshots[t]]
            axes[0, column].plot(times,
                                 [dist(before[rule_id].center, snapshots[t][rule_id].center)
                                  for t in times], marker=".", ms=3, label=f"rule {rule_id}")
            axes[1, column].plot(times,
                                 [snapshots[t][rule_id].sigma / before[rule_id].sigma
                                  for t in times], marker=".", ms=3, label=f"rule {rule_id}")
        axes[0, column].set_title("max_rules=5" if run == "capped" else "no rule limit")
        axes[0, column].set_ylabel("Center shift (L2 from t=1999)")
        axes[1, column].set_ylabel("Sigma / sigma at t=1999")
        axes[1, column].set_xlabel("Observation after training")
        for ax in axes[:, column]:
            ax.axvspan(2000, 3000, color="grey", alpha=0.17, label="drift window")
            ax.axvline(2500, color="black", ls="--", lw=0.8)
            ax.set_xlim(1900, 4500)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Rule antecedent memory around the AGRAWAL drift")
    fig.savefig(OUT / "rule_memory_drift.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((SOURCE / "summary.json").read_text(encoding="utf-8"))
    if (baseline["drift_window"] != [2000, 3000] or
            baseline["samples"] != 20000 or
            not all(run["exact_match"] for run in baseline["runs"].values())):
        raise ValueError("The source experiment does not match these phase boundaries")

    report = {"source": "results/3_drift_agrawal", "checkpoints": CHECKPOINTS,
              "phases": PHASES, "activation_semantics": "pre-update dominant rule only; raw activation",
              "comparison_semantics": "same persistent rule IDs; geometry shift is not proof of forgetting",
              "runs": {}}
    traces = {}
    for run in ("capped", "uncapped"):
        snapshots = rule_snapshots(SOURCE / f"{run}_events.csv.gz", CHECKPOINTS)
        if len(snapshots[20000]) != baseline["runs"][run]["final_rules"]:
            raise AssertionError(f"{run}: reconstructed final rule count differs from baseline")
        snapshot_rows = [{"sample": sample, "rule_id": rule_id,
                          "center": json.dumps(state.center), "sigma": state.sigma,
                          "support": state.support}
                         for sample, rules in snapshots.items()
                         for rule_id, state in sorted(rules.items())]
        write_rows(OUT / f"{run}_snapshots.csv",
                   ("sample", "rule_id", "center", "sigma", "support"), snapshot_rows)

        comparisons = []
        summaries = {}
        for name, (start, end) in COMPARISONS.items():
            rows = compare_snapshots(snapshots[start], snapshots[end])
            comparisons.extend({"comparison": name, **row} for row in rows)
            summaries[name] = memory_summary(rows)
        write_rows(OUT / f"{run}_changes.csv",
                   ("comparison", "rule_id", "status", "center_l2", "sigma_before",
                    "sigma_after", "sigma_ratio", "support_before", "support_after",
                    "support_delta"), comparisons)

        activations, sizes = dominant_activation_by_phase(SOURCE / f"{run}_samples.csv", PHASES)
        if sizes != {phase: end - start + 1 for phase, (start, end) in PHASES.items()}:
            raise AssertionError(f"{run}: missing observations in activation log")
        write_rows(OUT / f"{run}_dominant_activations.csv",
                   ("phase", "rule_id", "dominant_count", "dominant_fraction",
                    "mean_dominant_activation"), activations)
        traces[run] = (snapshots, activations)
        report["runs"][run] = {"active_rules_at_checkpoints":
                                {sample: len(rules) for sample, rules in snapshots.items()},
                                "changes": summaries,
                                "phase_observations": sizes,
                                "rules_ever_dominant_by_phase":
                                {phase: sum(row["phase"] == phase for row in activations)
                                 for phase in PHASES}}

    plot_memory(traces)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["runs"], indent=2))


if __name__ == "__main__":
    main()
