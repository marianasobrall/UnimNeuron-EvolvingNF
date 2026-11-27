"""Evaluate rule retention on matched AGRAWAL streams."""

import csv
import json
import platform
from collections import Counter
from time import perf_counter

import numpy as np
import river
from river.datasets import synth

from baseline_sine import PARAMETERS, ROOT, plt
from drift_agrawal import agrawal_drift_data
from src.eviction import POLICIES, RetentionSafe

OUT = ROOT / "results" / "6_retention"
PHASE = 5000
WINDOW = 250
REFERENCE = 1000
FRACTION = 0.90
NAMES = dict(support="Suporte", lru="LRU", fifo="FIFO")


def recurrent_data(seed):
    x, y = [], []
    for offset, function in enumerate((0, 2, 0)):
        stream = synth.Agrawal(classification_function=function, seed=3 * seed + offset)
        for features, label in stream.take(PHASE):
            x.append(list(features.values()))
            y.append(label)
    x = np.asarray(x, dtype=float)
    span = np.ptp(x, axis=0)
    return (x - x.min(axis=0)) / np.where(span == 0, 1, span), np.asarray(y, dtype=int)


def macro_f1(y, pred):
    scores = []
    for label in (0, 1):
        true = y == label
        predicted = pred == label
        denom = np.count_nonzero(true) + np.count_nonzero(predicted)
        scores.append(2 * np.count_nonzero(true & predicted) / denom if denom else 0.0)
    return sum(scores) / len(scores)


def recovery(y, pred, start, end, reference_end):
    threshold = FRACTION * macro_f1(y[reference_end - REFERENCE:reference_end],
                                    pred[reference_end - REFERENCE:reference_end])
    return recovery_at_target(y, pred, start, end, threshold)


def recovery_at_target(y, pred, start, end, threshold):
    consecutive = 0
    for stop in range(start + WINDOW, end + 1, WINDOW):
        passes = macro_f1(y[stop - WINDOW:stop], pred[stop - WINDOW:stop]) >= threshold
        consecutive = consecutive + 1 if passes else 0
        if consecutive == 2:
            return dict(threshold=threshold, delay=stop - start)
    return dict(threshold=threshold, delay=None)


def run(x, y, budget, policy, phases):
    events, samples = [], []
    cohort, before_return = set(), set()
    candidate = winner = None
    old_weight = total_weight = 0.0

    def on_event(row):
        nonlocal candidate
        event = row["event"]
        if event == "create":
            candidate = row["rule_id"]
        if event in ("create", "evict", "merge", "absorbed"):
            events.append((row["sample"], event, row["rule_id"], candidate, row["related_rule_id"]))

    def on_activation(row):
        nonlocal winner, old_weight, total_weight
        rid, weight = row["rule_id"], row["activation"]
        if row["dominant"]:
            winner = rid
        total_weight += weight
        if rid in cohort:
            old_weight += weight

    model = RetentionSafe(x.shape[1], 2, max_rules=budget, policy=policy,
                          on_event=on_event, on_activation=on_activation, **PARAMETERS)
    pred = np.empty(len(y), dtype=int)
    start_time = perf_counter()
    for i, xi in enumerate(x):
        if "A2" in phases and i == 2 * PHASE:
            before_return = set(model._ids.values())
        winner, old_weight, total_weight = None, 0.0, 0.0
        pred[i] = model.predict(xi[None, :])[0]
        model.partial_fit(xi[None, :], y[i:i + 1])
        if "A2" in phases and i == PHASE - 1:
            cohort = set(model._ids.values())
        samples.append((i + 1, int(y[i]), int(pred[i]), model.n_rules_, winner,
                        old_weight / total_weight if total_weight else 0.0))

    counts = Counter(row[1] for row in events)
    immediate = sum(event == "evict" and rid == new for _, event, rid, new, _ in events)
    assert max(row[3] for row in samples) <= budget
    assert counts["create"] - counts["evict"] - counts["absorbed"] == model.n_rules_
    report = dict(macro_f1=macro_f1(y, pred), final_rules=model.n_rules_,
                  events=dict(counts), immediate_evictions=immediate,
                  replacements=counts["evict"] - immediate, seconds=perf_counter() - start_time,
                  phase_f1={name: macro_f1(y[a:b], pred[a:b]) for name, (a,b) in phases.items()})
    if "A2" in phases:
        a2 = 2 * PHASE
        returned = samples[a2:]
        report["memory"] = dict(
            cohort_ids=sorted(cohort), survived_ids=sorted(cohort & before_return),
            reused_ids=sorted({row[4] for row in returned if row[4] in cohort}),
            dominance_share=sum(row[4] in cohort for row in returned) / len(returned),
            activation_share=float(np.mean([row[5] for row in returned])),
            first_1000_f1=macro_f1(y[a2:a2 + REFERENCE], pred[a2:a2 + REFERENCE]),
        )
        report["recovery"] = dict(B=recovery(y, pred, PHASE, a2, PHASE),
                                  A2=recovery(y, pred, a2, len(y), PHASE))
    else:
        report["recovery"] = dict(B=recovery(y, pred, 3000, len(y), 1999))
    return report, samples, events


def add_common_recovery(case, traces):
    recurring = case["scenario"] == "aba"
    bounds = {"B": case["phases"]["B"]}
    if recurring:
        bounds["A2"] = case["phases"]["A2"]
    runs = case["runs"]
    for phase, (start, end) in bounds.items():
        threshold = max(run["recovery"][phase]["threshold"] for run in runs.values())
        for policy, run in runs.items():
            y, pred = np.array([(row[1], row[2]) for row in traces[policy]]).T
            if "common_recovery" not in run:
                run["common_recovery"] = {}
            run["common_recovery"][phase] = recovery_at_target(y, pred, start, end, threshold)


def verify_reference(samples):
    path = ROOT / "results" / "3_drift_agrawal" / "capped_samples.csv"
    with path.open(newline="", encoding="utf-8") as file:
        expected = [tuple(int(row[k]) for k in ("sample", "actual", "prediction", "rules_after_update"))
                    for row in csv.DictReader(file)]
    assert [row[:4] for row in samples] == expected, "Support baseline differs from original drift run"


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def plot(report, curves):
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7), layout="constrained")
    for name, ax, title in (
        ("ab_42_5", axes[0, 0], "A→B gradual · 5 regras"),
        ("aba_42_5", axes[0, 1], "A–B–A abrupto · 5 regras · seed 42"),
    ):
        for policy, rows in curves[name].items():
            y, p = np.array([(r[1], r[2]) for r in rows]).T
            ends = list(range(WINDOW, len(rows) + 1, WINDOW))
            ax.plot(ends, [macro_f1(y[t-WINDOW:t], p[t-WINDOW:t]) for t in ends], label=NAMES[policy])
        if name.startswith("ab_"):
            ax.axvspan(2000, 3000, color="0.85", zorder=0)
        else:
            for boundary in (5000, 10000):
                ax.axvline(boundary, color="0.5", linestyle=":")
        ax.set(title=title, xlabel="Observação", ylabel="Macro-F1 por 250 observações", ylim=(0, 1))
        ax.legend()

    aba = [case for case in report["cases"] if case["scenario"] == "aba"]
    for i, policy in enumerate(POLICIES):
        for j, case in enumerate(c for c in aba if c["budget"] == 5):
            memory = case["runs"][policy]["memory"]
            axes[1, 0].scatter(i + (j-1)*0.1, memory["activation_share"],
                               marker=("o", "s", "^")[j], color=f"C{j}",
                               label=f'Seed {case["seed"]}' if i == 0 else None)
        selected = sorted((c for c in aba if c["seed"] == 42), key=lambda c: c["budget"])
        axes[1, 1].plot([c["budget"] for c in selected],
                        [c["runs"][policy]["phase_f1"]["A2"] for c in selected],
                        marker="o", label=NAMES[policy])
    axes[1, 0].set(title="A2 · peso das regras presentes no fim de A1",
                   xticks=range(3), xticklabels=list(NAMES.values()),
                   ylabel="Fração média da ativação", ylim=(-0.03, 1.03))
    axes[1, 1].set(title="A2 · sensibilidade ao orçamento · seed 42",
                   xlabel="Limite de regras", ylabel="Macro-F1", xticks=[5, 10, 20], ylim=(0, 1))
    for ax in axes[1]:
        ax.legend()
    fig.savefig(OUT / "retention.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = dict(versions=dict(python=platform.python_version(), numpy=np.__version__,
                                river=river.__version__),
                  parameters=PARAMETERS,
                  recovery=dict(window=WINDOW, reference_length=REFERENCE,
                                threshold_fraction=FRACTION, consecutive_windows=2,
                                common_threshold="Maximum of the three individual thresholds within each case"),
                  cases=[])
    curves = {}
    for scenario, seed, budget in (("ab", 42, 5), ("aba", 42, 5), ("aba", 43, 5),
                                   ("aba", 44, 5), ("aba", 42, 10), ("aba", 42, 20)):
        if scenario == "aba":
            x, y = recurrent_data(seed)
            phases = dict(A1=(0, PHASE), B=(PHASE, 2*PHASE), A2=(2*PHASE, 3*PHASE))
        else:
            x, y, _ = agrawal_drift_data()
            phases = {"A": (0, 1999), "Transição": (1999, 3000), "B": (3000, len(y))}
        case = dict(scenario=scenario, seed=seed, budget=budget, samples=len(y), phases=phases, runs={})
        key = f"{scenario}_{seed}_{budget}"
        curves[key] = {}
        for policy in POLICIES:
            metrics, samples, events = run(x, y, budget, policy, phases)
            if scenario == "ab" and policy == "support":
                verify_reference(samples)
                report["support_matches_ab_baseline"] = True
            case["runs"][policy] = metrics
            curves[key][policy] = samples
            write_csv(OUT / f"{key}_{policy}_samples.csv",
                      ("sample", "actual", "prediction", "rules_after_update", "dominant_rule_id",
                       "a1_cohort_activation_share"), samples)
            write_csv(OUT / f"{key}_{policy}_events.csv",
                      ("sample", "event", "rule_id", "candidate_id", "related_rule_id"), events)
            print(key, policy, f'F1={metrics["macro_f1"]:.4f}', flush=True)
        add_common_recovery(case, curves[key])
        report["cases"].append(case)
    plot(report, curves)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
