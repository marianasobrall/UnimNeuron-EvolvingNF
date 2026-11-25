"""T16 minimum viable experiment: rule-evolution timeline around a known concept drift.

Stream: ConceptDrift_Agrawal with Agrawal function 0 before the drift and function 2
after it (gradual drift centred at t=2,500, width 1,000, 20,000 samples). The baseline
repository uses function 0 on both sides, so its labels never change; this is the only
change to the stream, as described in the M5 proposal.

For max_rules=5 and max_rules=None, the original UnimNeuron-Safe and the instrumented
copy (src/instrumented.py) run side by side on the same samples. The script stops if a
prediction or a rule count ever differs, so every logged event belongs to the unchanged
baseline model.

Outputs go to results/3_drift_agrawal/.
"""

import csv
import gzip
import json
from collections import Counter
from time import perf_counter

import numpy as np
from river.datasets import synth

from baseline_sine import PARAMETERS, ROOT, plt
from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe
from src.instrumented import InstrumentedSafe

OUT = ROOT / "results" / "3_drift_agrawal"
SIZE = 20000
DRIFT_POSITION, DRIFT_WIDTH = 2500, 1000
DRIFT_WINDOW = (2000, 3000)          # samples where the concept is changing
WINDOW = 500                          # sliding window for prequential accuracy
BLOCK = 250                           # bin width for event-rate plots
PHASES = {"before": (1, DRIFT_WINDOW[0] - 1),
          "drift": DRIFT_WINDOW,
          "after": (DRIFT_WINDOW[1] + 1, SIZE)}
# update_consequent events are counted but not written: without a rule limit they
# would add millions of rows. Every other event is written with the rule state.
WRITTEN_EVENTS = {"create", "evict", "adapt", "merge", "absorbed", "class_change"}
EVENT_FIELDS = ("sample", "event", "rule_id", "related_rule_id", "previous_class",
                "rule_class", "support", "center", "sigma", "class_counts")


def agrawal_drift_data():
    """Same preparation as exp_sota_comparison: first-seen label codes, stream-wide min-max."""
    stream = synth.ConceptDriftStream(
        stream=synth.Agrawal(classification_function=0, seed=1),
        drift_stream=synth.Agrawal(classification_function=2, seed=2),
        seed=42, position=DRIFT_POSITION, width=DRIFT_WIDTH,
    )
    rows, labels = zip(*stream.take(SIZE))
    features = tuple(rows[0])
    x = np.array([[row[key] for key in features] for row in rows], dtype=float)
    codes = {label: i for i, label in enumerate(dict.fromkeys(labels))}
    span = np.ptp(x, axis=0)
    span[span == 0] = 1
    return (x - x.min(axis=0)) / span, np.array([codes[y] for y in labels]), features


def run(name, x, y, limit):
    events = Counter()
    lifetimes = {}
    dominant = {}

    with gzip.open(OUT / f"{name}_events.csv.gz", "wt", compresslevel=6,
                   encoding="utf-8", newline="") as event_file:
        writer = csv.DictWriter(event_file, fieldnames=EVENT_FIELDS)
        writer.writeheader()

        def on_event(row):
            event, sample, rule_id = row["event"], row["sample"], row["rule_id"]
            events[event] += 1
            if event in WRITTEN_EVENTS:
                writer.writerow(row)
            if event == "create":
                lifetimes[rule_id] = dict(rule_id=rule_id, created_at=sample, ended_at=None,
                                          end_event=None, positive_activations=0,
                                          last_positive_at=None, dominant_count=0,
                                          last_dominant_at=None, class_changes=0)
            elif event in ("evict", "absorbed"):
                lifetimes[rule_id].update(ended_at=sample, end_event=event)
                if event == "evict" and lifetimes[rule_id]["created_at"] == sample:
                    events["immediate_eviction"] += 1
            elif event == "class_change":
                lifetimes[rule_id]["class_changes"] += 1

        def on_activation(row):
            rule = lifetimes[row["rule_id"]]
            if row["activation"] > 0:
                rule["positive_activations"] += 1
                rule["last_positive_at"] = row["sample"]
            if row["dominant"]:
                rule["dominant_count"] += 1
                rule["last_dominant_at"] = row["sample"]
                dominant[row["sample"]] = (row["rule_id"], row["activation"])

        original = EvolvingNeuroFuzzyUnimSafe(x.shape[1], 2, max_rules=limit, **PARAMETERS)
        traced = InstrumentedSafe(x.shape[1], 2, max_rules=limit, on_event=on_event,
                                  on_activation=on_activation, **PARAMETERS)

        samples = []
        start = perf_counter()
        for i in range(len(y)):
            xi, yi = x[i:i + 1], y[i:i + 1]
            expected = int(original.predict(xi)[0]) if i else 0
            predicted = int(traced.predict(xi)[0]) if i else 0
            original.partial_fit(xi, yi)
            traced.partial_fit(xi, yi)
            if expected != predicted or original.n_rules_ != traced.n_rules_:
                raise AssertionError(f"{name}: instrumented model differs at sample {i + 1}")
            rule_id, activation = dominant.get(i + 1, ("", ""))
            samples.append(dict(sample=i + 1, prediction=predicted, actual=int(y[i]),
                                correct=int(predicted == y[i]),
                                rules_after_update=traced.n_rules_,
                                dominant_rule_id=rule_id, dominant_activation=activation))
        seconds = perf_counter() - start

    if events["create"] - events["evict"] - events["absorbed"] != traced.n_rules_:
        raise AssertionError(f"{name}: rule lifecycle counts do not reconcile")

    with (OUT / f"{name}_samples.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    with (OUT / f"{name}_lifetimes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(lifetimes[1]))
        writer.writeheader()
        writer.writerows(lifetimes.values())
    return samples, lifetimes, events, seconds


def event_samples(name, kinds):
    """Sample index of every written event of the given kinds."""
    found = {kind: [] for kind in kinds}
    with gzip.open(OUT / f"{name}_events.csv.gz", "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["event"] in found:
                found[row["event"]].append(int(row["sample"]))
    return {kind: np.array(values, dtype=int) for kind, values in found.items()}


def balanced_accuracy(actual, predicted):
    return float(np.mean([np.mean(predicted[actual == c] == c) for c in np.unique(actual)]))


def phase_metrics(samples):
    actual = np.array([s["actual"] for s in samples])
    predicted = np.array([s["prediction"] for s in samples])
    result = {}
    for phase, (first, last) in {"all": (1, SIZE), **PHASES}.items():
        a, p = actual[first - 1:last], predicted[first - 1:last]
        result[phase] = dict(samples=len(a), accuracy=round(float(np.mean(a == p)), 4),
                             balanced_accuracy=round(balanced_accuracy(a, p), 4),
                             majority_rate=round(float(np.bincount(a).max() / len(a)), 4))
    return result


def event_rates(times):
    """Events per 1,000 samples in each phase."""
    rates = {}
    for kind, t in times.items():
        rates[kind] = {phase: round(1000 * np.sum((t >= first) & (t <= last)) / (last - first + 1), 2)
                       for phase, (first, last) in PHASES.items()}
    return rates


def windowed_accuracy(samples):
    correct = np.array([s["correct"] for s in samples], dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(correct)))
    ends = np.arange(1, len(correct) + 1)
    starts = np.maximum(0, ends - WINDOW)
    return (cumulative[ends] - cumulative[starts]) / (ends - starts)


def mark_drift(ax):
    ax.axvspan(*DRIFT_WINDOW, color="grey", alpha=0.18, lw=0)
    ax.axvline(DRIFT_POSITION, color="black", ls="--", lw=0.9)


def plot_timeline(runs, times):
    edges = np.arange(0, SIZE + BLOCK, BLOCK)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    colours = {"capped": "#c0392b", "uncapped": "#2471a3"}
    labels = {"capped": "max_rules=5", "uncapped": "max_rules=None"}

    for name, (samples, *_rest) in runs.items():
        axes[0].plot(np.arange(1, SIZE + 1), windowed_accuracy(samples),
                     color=colours[name], lw=1.1, label=labels[name])
        axes[1].step(np.arange(1, SIZE + 1), [s["rules_after_update"] for s in samples],
                     where="post", color=colours[name], lw=1.1, label=labels[name])
    axes[0].set(ylabel=f"Accuracy (last {WINDOW})", ylim=(0, 1))
    axes[1].set(ylabel="Active rules", yscale="log")

    for ax, name in zip(axes[2:], ("capped", "uncapped")):
        for kind, style in (("create", "-"), ("evict", "--"), ("adapt", "-"), ("class_change", "-")):
            counts, _ = np.histogram(times[name][kind], bins=edges)
            ax.stairs(counts, edges, ls=style, label=kind.replace("_", " "))
        ax.set(ylabel=f"Events per {BLOCK} samples", title=labels[name])
        ax.legend(ncol=4, fontsize=8, loc="upper right")

    for ax in axes:
        mark_drift(ax)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower right")
    axes[0].set_title("UnimNeuron-Safe on ConceptDrift_Agrawal (function 0 → 2); "
                      "shaded: drift window, dashed: drift centre")
    axes[-1].set(xlabel="Sample", xlim=(0, SIZE))
    fig.savefig(OUT / "drift_timeline.png", dpi=160)
    plt.close(fig)


def plot_lifetimes(lifetimes):
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for rule in lifetimes.values():
        end = rule["ended_at"] or SIZE
        ax.hlines(rule["rule_id"], rule["created_at"], end, color="#2471a3", lw=1)
    mark_drift(ax)
    ax.set(title="Rule lifetimes without a rule limit (one line per rule ID)",
           xlabel="Sample", ylabel="Rule ID", xlim=(0, SIZE))
    ax.grid(alpha=0.3)
    fig.savefig(OUT / "rule_lifetimes_uncapped.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x, y, features = agrawal_drift_data()
    report = dict(dataset="ConceptDrift_Agrawal", functions="0 -> 2", samples=SIZE,
                  drift_position=DRIFT_POSITION, drift_width=DRIFT_WIDTH,
                  drift_window=DRIFT_WINDOW, accuracy_window=WINDOW, features=features,
                  parameters=PARAMETERS, written_events=sorted(WRITTEN_EVENTS), runs={})
    runs, times = {}, {}
    for name, limit in (("capped", 5), ("uncapped", None)):
        samples, lifetimes, events, seconds = run(name, x, y, limit)
        runs[name] = (samples, lifetimes, events)
        times[name] = event_samples(name,("create", "evict", "adapt", "class_change"))
        report["runs"][name] = dict(
            max_rules=limit, exact_match=True, seconds=round(seconds, 1),
            final_rules=samples[-1]["rules_after_update"],
            rules_ever_created=len(lifetimes),
            rules_that_ever_predicted=sum(r["positive_activations"] > 0 for r in lifetimes.values()),
            events=dict(events), event_rate_per_1000=event_rates(times[name]),
            predictive=phase_metrics(samples))
        print(name, json.dumps(report["runs"][name]["predictive"]), flush=True)
    plot_timeline(runs, times)
    plot_lifetimes(runs["uncapped"][1])
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
