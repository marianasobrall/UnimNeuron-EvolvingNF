"""
M3 - Documented reproduction of the UnimNeuron baseline (full stream lengths)
================================================================

Runs the eight streams of Experiment 1 of the official repository, with the same
stream lengths as its "complete" mode, WITHOUT modifying any baseline file:

  * the official exp_sota_comparison.py is imported as a library (its data
    loading, scaling, prequential loop and rule export are reused as-is);
  * the two UnimNeuron models are wrapped by thin *instrumented* subclasses
    that only COUNT structural events (rule creation, eviction by max_rules,
    merging). They do not change any decision of the model;
  * the paper hyper-parameters are used (max_rules=5, tau_merge=0.62/0.68).

Extra (diagnostic) runs:
  * an online majority-class predictor, to put accuracies in context;
  * UnimNeuron-Safe with max_rules=None, to inspect uncapped rule growth.

Usage (from the project root):
    .venv/Scripts/python experiments/reproduce_baseline.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
OUTDIR = os.path.join(HERE, "results_repro")
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Import the official experiment script as a library.
# It contains a non-optional import of a module that is NOT in the repository
# (`enfs_uni0_evolving`), so we register an empty stub before importing.
# The module also creates "results_exp/" in the CWD at import time, so we
# import it from inside OUTDIR to keep the baseline folder untouched.
# ---------------------------------------------------------------------------
_stub = types.ModuleType("enfs_uni0_evolving")
_stub.ENFSUni0Evolving = None
sys.modules["enfs_uni0_evolving"] = _stub
sys.path.insert(0, REPO)

_cwd = os.getcwd()
os.chdir(OUTDIR)
import exp_sota_comparison as base  # noqa: E402
from ENF_UnimNeuron_Safe import EvolvingNeuroFuzzyUnimSafe  # noqa: E402
from ENF_UnimNeuron_PA import EvolvingNeuroFuzzyUnimPA  # noqa: E402
os.chdir(_cwd)
try:
    os.rmdir(os.path.join(OUTDIR, "results_exp"))  # empty side-effect folder
except OSError:
    pass

import matplotlib.pyplot as plt  # noqa: E402  (backend already set to Agg)
from river import datasets  # noqa: E402
from river.datasets import synth  # noqa: E402


# ---------------------------------------------------------------------------
# Instrumentation (observation only)
# ---------------------------------------------------------------------------
class _EventCounterMixin:
    """Counts structural events without altering the model's behaviour."""

    def _init_events(self):
        self.events = dict(created=0, evicted=0, new_rule_evicted=0, merged=0)

    def _wrap_create(self, create_fn, *args):
        before = {id(r) for r in self.rules}
        n_before = len(self.rules)
        create_fn(*args)
        self.events["created"] += 1
        if self.max_rules is not None and n_before + 1 > self.max_rules:
            self.events["evicted"] += 1
            if not ({id(r) for r in self.rules} - before):
                # the rule that was just created was the one removed
                self.events["new_rule_evicted"] += 1


class InstrumentedSafe(_EventCounterMixin, EvolvingNeuroFuzzyUnimSafe):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._init_events()

    def _create_rule(self, x, y, sigma):
        self._wrap_create(super()._create_rule, x, y, sigma)

    def _maybe_merge_last_rule(self):
        n_before = self.n_rules_
        super()._maybe_merge_last_rule()
        if self.n_rules_ < n_before:
            self.events["merged"] += 1


class InstrumentedPA(_EventCounterMixin, EvolvingNeuroFuzzyUnimPA):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._init_events()

    def _create_rule(self, x, sigma):
        self._wrap_create(super()._create_rule, x, sigma)


class OnlineMajority:
    """Predicts the most frequent label seen so far (sanity baseline)."""

    def __init__(self, n_classes):
        self.counts = np.zeros(n_classes)

    def predict(self, X):
        return np.array([int(np.argmax(self.counts))])

    def partial_fit(self, X, y):
        self.counts[int(y[0])] += 1


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Same stream definitions and lengths as the official "complete" mode
# (STREAM_LENGTHS_COMPLETE; Phishing has only 1250 samples), except RandomRBFDrift, whose
# official definition has no seed (non-reproducible) and change_speed=0
# (i.e. no drift). We fix the seeds and keep change_speed=0 to stay faithful.
STREAMS = {
    "Elec2_real": (datasets.Elec2(), 10000),
    "CreditCard_real": (datasets.CreditCard(), 5000),
    "Phishing_real": (datasets.Phishing(), 1250),
    "Bananas_real": (datasets.Bananas(), 5000),
    "ConceptDrift_Agrawal": (
        synth.ConceptDriftStream(
            stream=synth.Agrawal(seed=1), drift_stream=synth.Agrawal(seed=2),
            seed=42, position=2500, width=1000,
        ),
        20000,
    ),
    "Agrawal_synth": (synth.Agrawal(seed=42), 20000),
    "Sine_synth": (synth.Sine(seed=42), 40000),
    "RandomRBFDrift_synth": (synth.RandomRBFDrift(seed_model=42, seed_sample=42), 20000),
}

# Hyper-parameters copied from exp_sota_comparison.main().
# NOTE: safe_threshold, min_firing_for_update, drift_sensitivity, use_PA_I,
# margin_threshold and decay_rate are NOT parameters of the models: they are
# silently swallowed by **kwargs. We pass them anyway to be faithful.
SAFE_KW = dict(max_rules=5, safe_threshold=0.07, min_firing_for_update=0.05,
               tau_merge=0.62, drift_sensitivity=0.35)
PA_KW = dict(max_rules=5, C=0.25, use_PA_I=True, margin_threshold=0.8,
             decay_rate=1e-2, tau_merge=0.68)


def balanced_accuracy(y, y_pred):
    recalls = [np.mean(y_pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(recalls))


def build_models(n_features, n_classes):
    return {
        "ENF_Unim_Safe": InstrumentedSafe(n_features, n_classes, **SAFE_KW),
        "ENF_Unim_PA": InstrumentedPA(n_features, n_classes, **PA_KW),
        "Safe_uncapped": InstrumentedSafe(n_features, n_classes, **{**SAFE_KW, "max_rules": None}),
        "Majority": OnlineMajority(n_classes),
    }


def main():
    rows = []
    for ds_name, (ds, n_max) in STREAMS.items():
        print(f"\n=== {ds_name} (N={n_max}) ===")
        X_raw, y, feat = base.load_stream_to_numpy(ds, n_max=n_max)
        X, _, _ = base.min_max_scale(X_raw)  # official (non-causal) scaling
        n, d = X.shape
        C = len(np.unique(y))
        ds_out = os.path.join(OUTDIR, ds_name)
        os.makedirs(ds_out, exist_ok=True)

        curves = {}
        for m_name, model in build_models(d, C).items():
            t0 = time.time()
            res = base.prequential_nf_generic(model, X, y, desc=f"{m_name}/{ds_name}")
            dt = time.time() - t0
            curves[m_name] = res
            ev = getattr(model, "events", {})
            row = dict(
                dataset=ds_name, model=m_name, N=n, d=d, classes=C,
                acc_final=round(float(res["accuracy"][-1]), 4),
                acc_mean=round(float(res["accuracy"].mean()), 4),
                bal_acc=round(balanced_accuracy(y, res["y_pred"]), 4),
                majority_rate=round(float(np.bincount(y).max() / n), 4),
                rules_final=int(res["n_rules_final"]) if m_name != "Majority" else 0,
                rules_max=int(np.nanmax(res["n_rules_time"])) if m_name != "Majority" else 0,
                created=ev.get("created", 0), evicted=ev.get("evicted", 0),
                new_rule_evicted=ev.get("new_rule_evicted", 0), merged=ev.get("merged", 0),
                time_s=round(dt, 2),
            )
            # step at which the capped rule base became full
            rt = res["n_rules_time"]
            row["t_full"] = int(np.argmax(rt >= 5)) if m_name in ("ENF_Unim_Safe", "ENF_Unim_PA") and np.any(rt >= 5) else ""
            rows.append(row)
            print("[RESULT]", json.dumps(row))

            if m_name in ("ENF_Unim_Safe", "ENF_Unim_PA"):
                base.export_rules_latex(model=model, model_name=m_name, dataset_outdir=ds_out,
                                        feature_names=feat, max_rules_to_write=5,
                                        top_features_per_rule=6)
                if m_name == "ENF_Unim_Safe":
                    with open(os.path.join(ds_out, "rules_ENF_Unim_Safe_classprobs.txt"), "w") as f:
                        for i, r in enumerate(model.rules, 1):
                            f.write(f"Rule {i}: support={r.support} sigma={r.sigma:.3f} "
                                    f"class_probs={np.round(r.class_probs(), 3).tolist()} "
                                    f"w={np.round(r.w, 2).tolist()}\n")

        # plots: accuracy and number of rules over time
        fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        for m_name, res in curves.items():
            axes[0].plot(res["accuracy"], lw=1.1, label=m_name)
            if m_name != "Majority":
                axes[1].plot(res["n_rules_time"], lw=1.1, label=m_name)
        if ds_name == "ConceptDrift_Agrawal":
            for ax in axes:
                ax.axvspan(2000, 3000, color="grey", alpha=0.15)
                ax.axvline(2500, color="grey", ls="--", lw=0.8)
        axes[0].set_ylabel("Prequential accuracy")
        axes[1].set_ylabel("# rules")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Samples")
        axes[0].legend(fontsize=7)
        axes[0].set_title(ds_name)
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(ds_out, f"{ds_name}_repro.png"), dpi=200)
        plt.close(fig)

    with open(os.path.join(OUTDIR, "repro_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nSaved:", os.path.join(OUTDIR, "repro_summary.csv"))


if __name__ == "__main__":
    main()
