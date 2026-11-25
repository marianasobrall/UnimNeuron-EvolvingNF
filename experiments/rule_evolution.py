"""Validate and export the rule timeline using the Sine baseline from point 1."""

import csv
import gzip
import json
from collections import Counter

import numpy as np

from baseline_sine import OUT as BASELINE, PARAMETERS, ROOT, SEED, plt, sine_data
from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe
from src.instrumented import InstrumentedSafe

OUT = ROOT / "results" / "rule_evolution"
EVENT_FIELDS = ("sample", "event", "rule_id", "related_rule_id", "previous_class",
                "rule_class", "support", "center", "sigma", "class_counts")
ACTIVATION_FIELDS = ("sample", "rule_id", "activation", "dominant")


def evaluate(model, x, y, baseline_file):
    with baseline_file.open(newline="") as file:
        reference = csv.DictReader(file)
        for sample, (row, label) in enumerate(zip(x, y), start=1):
            expected = next(reference, None)
            prediction = int(model.predict(row[None, :])[0])
            model.partial_fit(row[None, :], np.array([label]))
            actual = (sample, prediction, int(label), model.n_rules_)
            if expected is None or actual != tuple(int(expected[key]) for key in
                    ("sample", "prediction", "actual", "rules_after_update")):
                raise AssertionError(f"{baseline_file.stem}: mismatch at sample {sample}")
        if next(reference, None) is not None:
            raise AssertionError(f"{baseline_file.stem}: unexpected extra baseline rows")


def trace(name, size, limit):
    x, y = sine_data(size)
    events = Counter()
    lifetimes = {}
    first_rule = []
    bin_width = 250
    churn = np.zeros((2, (size + bin_width - 1) // bin_width), dtype=int)
    with (
        gzip.open(OUT / f"{name}_events.csv.gz", "wt", compresslevel=3, encoding="utf-8", newline="") as event_file,
        gzip.open(OUT / f"{name}_activations.csv.gz", "wt", compresslevel=3, encoding="utf-8", newline="") as act_file,
    ):
        event_writer = csv.DictWriter(event_file, fieldnames=EVENT_FIELDS)
        act_writer = csv.DictWriter(act_file, fieldnames=ACTIVATION_FIELDS)
        event_writer.writeheader()
        act_writer.writeheader()

        def on_event(row):
            event_writer.writerow(row)
            event, sample, rule_id = row["event"], row["sample"], row["rule_id"]
            events[event] += 1
            bucket = (sample - 1) // bin_width
            if event == "create":
                lifetimes[rule_id] = dict(rule_id=rule_id, created_at=sample, ended_at=None,
                                          end_event=None, positive_activations=0,
                                          last_positive_at=None, dominant_count=0,
                                          last_dominant_at=None)
                churn[0, bucket] += 1
            if event in ("evict", "absorbed"):
                lifetimes[rule_id].update(ended_at=sample, end_event=event)
                churn[1, bucket] += 1
                if lifetimes[rule_id]["created_at"] == sample and event == "evict":
                    events["immediate_eviction"] += 1
            if rule_id == 1 and event in ("create", "adapt", "merge", "evict", "absorbed"):
                first_rule.append(row)

        def on_activation(row):
            act_writer.writerow(row)
            rule = lifetimes[row["rule_id"]]
            if row["activation"] > 0:
                rule["positive_activations"] += 1
                rule["last_positive_at"] = row["sample"]
            if row["dominant"]:
                rule["dominant_count"] += 1
                rule["last_dominant_at"] = row["sample"]

        model = InstrumentedSafe(x.shape[1], len(np.unique(y)), max_rules=limit,
                                 on_event=on_event, on_activation=on_activation, **PARAMETERS)
        evaluate(model, x, y, BASELINE / f"{name}.csv")

    if events["create"] - events["evict"] - events["absorbed"] != model.n_rules_:
        raise AssertionError("Rule lifecycle counts do not reconcile")

    with (OUT / f"{name}_lifetimes.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(lifetimes[1]))
        writer.writeheader()
        writer.writerows(lifetimes.values())

    summary = dict(run=name, samples=size, max_rules=limit, final_rules=model.n_rules_,
                   exact_match=True, events=dict(events))
    return summary, lifetimes, churn, first_rule


def check_merge():
    # The Sine runs use the defaults. Here we relax the merge guards to test logging.
    x, y = sine_data(30)
    params = dict(min_rules_for_merge=1, min_support_for_merge=1, tau_merge=0.0)
    events = []
    base = EvolvingNeuroFuzzyUnimSafe(x.shape[1], len(np.unique(y)), **params)
    traced = InstrumentedSafe(x.shape[1], len(np.unique(y)), on_event=events.append, **params)
    for index in range(len(y)):
        xi, yi = x[index:index + 1], y[index:index + 1]
        np.testing.assert_array_equal(base.predict(xi), traced.predict(xi))
        base.partial_fit(xi, yi)
        traced.partial_fit(xi, yi)
        assert base.n_rules_ == traced.n_rules_
        for expected, actual in zip(base.rules, traced.rules):
            assert (expected.support, expected.sigma) == (actual.support, actual.sigma)
            for field in ("center", "class_counts", "w"):
                np.testing.assert_array_equal(getattr(expected, field), getattr(actual, field))

    merges = [(row["sample"], row["rule_id"], row["related_rule_id"])
              for row in events if row["event"] == "merge"]
    absorbed = [(row["sample"], row["related_rule_id"], row["rule_id"])
                for row in events if row["event"] == "absorbed"]
    assert merges and merges == absorbed
    return len(merges)


def plot(short_runs):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), constrained_layout=True)
    for ax, (summary, _, churn, _) in zip(axes[0], short_runs):
        edges = 250 * np.arange(churn.shape[1] + 1)
        ax.stairs(churn[0], edges, baseline=None, label="Criações")
        ax.stairs(churn[1], edges, baseline=None, linestyle="--", label="Saídas")
        ax.set(title="Limite de 5" if summary["max_rules"] else "Sem limite",
               xlabel="Observação", ylabel="Eventos por 250 observações", xlim=(0, 5000))
        ax.legend()
    ax = axes[1, 0]
    summary, lifetimes, _, first_rule = short_runs[1]
    for rule in lifetimes.values():
        end = rule["ended_at"] or summary["samples"]
        ax.hlines(rule["rule_id"], rule["created_at"], end, color="#286078")
    ax.set(title="Tempo de vida das regras sem limite", xlabel="Observação", ylabel="ID da regra")
    ax = axes[1, 1]
    samples = [row["sample"] for row in first_rule]
    ax.plot(samples, [row["center"][0] for row in first_rule], label="Centro (1.ª dimensão)")
    ax.plot(samples, [row["center"][1] for row in first_rule], label="Centro (2.ª dimensão)")
    ax.plot(samples, [row["sigma"] for row in first_rule], label="Sigma")
    ax.set(title="Parâmetros da regra 1 sem limite", xlabel="Observação", ylabel="Valor")
    ax.legend()
    fig.savefig(OUT / "timeline.png", dpi=155)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    short_runs = []
    report = dict(dataset="Sine_synth", seed=SEED, parameters=PARAMETERS, runs=[])
    for name, size, limit in (("capped_quick", 5000, 5), ("uncapped_quick", 5000, None)):
        result = trace(name, size, limit)
        short_runs.append(result)
        report["runs"].append(result[0])
        print(result[0], flush=True)

    x, y = sine_data(40000)
    full_events = Counter()

    def count_event(row):
        full_events[row["event"]] += 1

    def count_activation(row):
        full_events["prediction_activation"] += 1

    full = InstrumentedSafe(x.shape[1], len(np.unique(y)), max_rules=5,
                            on_event=count_event, on_activation=count_activation, **PARAMETERS)
    evaluate(full, x, y, BASELINE / "capped_full.csv")
    report["full_check"] = dict(samples=len(y), final_rules=full.n_rules_, exact_match=True,
                                logged_counts=dict(full_events))
    report["technical_merge_check"] = dict(samples=30, merges=check_merge(), exact_match=True)
    plot(short_runs)
    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["full_check"], report["technical_merge_check"], flush=True)


if __name__ == "__main__":
    main()
