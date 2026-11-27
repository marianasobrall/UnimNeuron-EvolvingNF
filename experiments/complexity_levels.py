"""Point 4: rule evolution under three levels of data complexity.

Three stationary streams (simple, medium, overlap) of 30,000 samples. For each one,
the original UnimNeuron-Safe and the instrumented copy run side by side, with
max_rules=5 and without a limit. The script stops if their predictions differ.

Scaling is the same as in the other scripts: min-max over the whole stream.

Run:          python experiments/complexity_levels.py
Quick check:  python experiments/complexity_levels.py --quick
Results:      results/5_complexity/
"""

import csv
import gzip
import json
import sys
from collections import Counter
from time import perf_counter

import numpy as np

from baseline_sine import PARAMETERS, ROOT, plt
from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe
from src.instrumented import InstrumentedSafe
from src.streams import COMPLEXITY_LEVELS, bayes_accuracy, gaussian_clouds

OUT = ROOT / "results" / "5_complexity"
SIZE = 30000
SEED = 42
WINDOW = 500
BLOCK = 5000
LIMITS = {"capped": 5, "uncapped": None}
SAVED_EVENTS = {"create", "evict", "adapt", "merge", "absorbed", "class_change"}
EVENT_FIELDS = ("sample", "event", "rule_id", "related_rule_id", "previous_class",
                "rule_class", "support", "center", "sigma", "class_counts")
COLOURS = {"simple": "#1e8449", "medium": "#d68910", "overlap": "#c0392b"}


def load(level, size):
    x, y = gaussian_clouds(size, COMPLEXITY_LEVELS[level], SEED)
    span = np.ptp(x, axis=0)
    span[span == 0] = 1
    return (x - x.min(axis=0)) / span, y


def run(tag, x, y, limit):
    events = Counter()
    rules_info = {}

    with gzip.open(OUT / f"{tag}_events.csv.gz", "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        writer.writeheader()

        def on_event(row):
            kind, sample, rule_id = row["event"], row["sample"], row["rule_id"]
            events[kind] += 1
            if kind in SAVED_EVENTS:
                writer.writerow(row)
            if kind == "create":
                rules_info[rule_id] = dict(rule_id=rule_id, created_at=sample, ended_at=None,
                                           end_event=None, times_dominant=0, class_changes=0)
            elif kind in ("evict", "absorbed"):
                rules_info[rule_id].update(ended_at=sample, end_event=kind)
                if kind == "evict" and rules_info[rule_id]["created_at"] == sample:
                    events["immediate_eviction"] += 1
            elif kind == "class_change":
                rules_info[rule_id]["class_changes"] += 1

        def on_activation(row):
            if row["dominant"]:
                rules_info[row["rule_id"]]["times_dominant"] += 1

        original = EvolvingNeuroFuzzyUnimSafe(2, 2, max_rules=limit, **PARAMETERS)
        traced = InstrumentedSafe(2, 2, max_rules=limit, on_event=on_event,
                                  on_activation=on_activation, **PARAMETERS)

        pred = np.zeros(len(y), dtype=int)
        rules = np.zeros(len(y), dtype=int)
        start = perf_counter()
        for i in range(len(y)):
            xi, yi = x[i:i + 1], y[i:i + 1]
            p_original = int(original.predict(xi)[0]) if i else 0
            p_traced = int(traced.predict(xi)[0]) if i else 0
            original.partial_fit(xi, yi)
            traced.partial_fit(xi, yi)
            if p_original != p_traced or original.n_rules_ != traced.n_rules_:
                raise AssertionError(f"{tag}: models differ at sample {i + 1}")
            pred[i] = p_traced
            rules[i] = traced.n_rules_
        seconds = perf_counter() - start

    if events["create"] - events["evict"] - events["absorbed"] != traced.n_rules_:
        raise AssertionError(f"{tag}: rule counts do not add up")

    with (OUT / f"{tag}_samples.csv").open("w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["sample", "prediction", "actual", "correct", "rules_after_update"])
        out.writerows(zip(range(1, len(y) + 1), pred, y, (pred == y).astype(int), rules))
    with (OUT / f"{tag}_rules.csv").open("w", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rules_info[1]))
        out.writeheader()
        out.writerows(rules_info.values())
    return pred, rules, rules_info, events, seconds


def f1_per_class(actual, pred, c):
    tp = np.sum((pred == c) & (actual == c))
    fp = np.sum((pred == c) & (actual != c))
    fn = np.sum((pred != c) & (actual == c))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def macro_f1(actual, pred):
    return float(np.mean([f1_per_class(actual, pred, c) for c in (0, 1)]))


def rolling_macro_f1(actual, pred):
    def rolling_sum(flags):
        total = np.concatenate(([0], np.cumsum(flags)))
        ends = np.arange(1, len(flags) + 1)
        return total[ends] - total[np.maximum(0, ends - WINDOW)]

    scores = []
    for c in (0, 1):
        tp = rolling_sum((pred == c) & (actual == c))
        fp = rolling_sum((pred == c) & (actual != c))
        fn = rolling_sum((pred != c) & (actual == c))
        den = 2 * tp + fp + fn
        scores.append(np.divide(2 * tp, den, out=np.zeros(len(tp)), where=den > 0))
    return np.mean(scores, axis=0)


def growth(rules, rules_info, size):
    final = int(rules[-1])
    created = [r["created_at"] for r in rules_info.values()]
    return {
        "samples_to_50pct_of_final": int(np.argmax(rules >= np.ceil(0.5 * final)) + 1),
        "samples_to_90pct_of_final": int(np.argmax(rules >= np.ceil(0.9 * final)) + 1),
        "rules_created_per_block": np.histogram(created, bins=np.arange(0, size + BLOCK, BLOCK))[0].tolist(),
        "block_size": BLOCK,
    }


def plot_data(size):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True,
                             constrained_layout=True)
    for ax, level in zip(axes, COMPLEXITY_LEVELS):
        x, y = load(level, size)
        x, y = x[:2000], y[:2000]
        ax.scatter(*x[y == 0].T, s=3, alpha=0.5, label="class 0")
        ax.scatter(*x[y == 1].T, s=3, alpha=0.5, label="class 1")
        d = COMPLEXITY_LEVELS[level]
        ax.set(title=f"{level} (d = {d}, best accuracy {bayes_accuracy(d):.3f})",
               xlabel="$x_1$ (scaled)")
        ax.grid(alpha=0.3)
    axes[0].set(ylabel="$x_2$ (scaled)")
    axes[0].legend(markerscale=4, fontsize=8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"complexity_data.{ext}", dpi=200)
    plt.close(fig)


def plot_results(curves, size):
    t = np.arange(1, size + 1)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    for level, colour in COLOURS.items():
        axes[0].step(t, curves[level]["uncapped"]["rules"], where="post", color=colour,
                     lw=1.2, label=level)
        for ax, limit in ((axes[1], "uncapped"), (axes[2], "capped")):
            ax.plot(t, curves[level][limit]["f1"], color=colour, lw=1.0, label=level)
            ax.axhline(bayes_accuracy(COMPLEXITY_LEVELS[level]), color=colour, ls=":", lw=1.0)
    axes[0].set(ylabel="Active rules", title="Without a rule limit")
    axes[1].set(ylabel=f"Macro-F1 (last {WINDOW})", ylim=(0, 1.02), title="Without a rule limit")
    axes[2].set(ylabel=f"Macro-F1 (last {WINDOW})", ylim=(0, 1.02),
                title="With the original limit (5 rules)", xlabel="Sample", xlim=(0, size))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("UnimNeuron-Safe on three streams of increasing class overlap "
                 "(dotted: best possible accuracy)", fontsize=10)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"complexity_rule_evolution.{ext}", dpi=200)
    plt.close(fig)


def latex_table(report):
    rows = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r" & & \multicolumn{2}{c}{Macro-F1} & \multicolumn{2}{c}{Rules created} & Final rules \\",
            r"Level & Best acc. & limit 5 & no limit & limit 5 & no limit & no limit \\",
            r"\midrule"]
    for level, info in report["levels"].items():
        c, u = info["runs"]["capped"], info["runs"]["uncapped"]
        rows.append(f"{level} & {info['bayes_accuracy']:.3f} & {c['macro_f1']:.3f} & "
                    f"{u['macro_f1']:.3f} & {c['rules_created']:,} & {u['rules_created']:,} & "
                    f"{u['final_rules']} \\\\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "complexity_table.tex").write_text("\n".join(rows) + "\n")


def main():
    global OUT
    size = SIZE
    if "--quick" in sys.argv:
        size, OUT = 3000, OUT / "quick"
    OUT.mkdir(parents=True, exist_ok=True)

    report = dict(samples=size, seed=SEED, scaling="min-max over the whole stream",
                  f1_window=WINDOW, parameters=PARAMETERS, levels={})
    curves = {}
    for level, d in COMPLEXITY_LEVELS.items():
        x, y = load(level, size)
        info = dict(separation=d, bayes_accuracy=round(bayes_accuracy(d), 4), runs={})
        curves[level] = {}
        for name, limit in LIMITS.items():
            tag = f"{level}_{name}"
            pred, rules, rules_info, events, seconds = run(tag, x, y, limit)
            curves[level][name] = dict(rules=rules, f1=rolling_macro_f1(y, pred))
            info["runs"][name] = dict(
                max_rules=limit, seconds=round(seconds, 1),
                accuracy=round(float(np.mean(pred == y)), 4),
                macro_f1=round(macro_f1(y, pred), 4),
                final_rules=int(rules[-1]),
                rules_created=len(rules_info),
                rules_ever_dominant=sum(r["times_dominant"] > 0 for r in rules_info.values()),
                immediate_eviction_share=round(events["immediate_eviction"] / events["create"], 4),
                events=dict(events),
                growth=growth(rules, rules_info, size))
            print(tag, "macro-F1", info["runs"][name]["macro_f1"],
                  "| final rules", int(rules[-1]), "| created", len(rules_info), flush=True)
        report["levels"][level] = info

    plot_data(size)
    plot_results(curves, size)
    latex_table(report)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()