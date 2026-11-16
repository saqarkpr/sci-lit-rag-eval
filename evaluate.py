#!/usr/bin/env python3
"""Compute trustworthy-evaluation metrics from the out-of-fold predictions
written by train.py, for both retrievers, and produce results/report.md plus
reliability diagrams.

Sections produced:
  1. Headline table (retrieval Recall@k/MRR, QA accuracy, ECE, Brier,
     hallucination / false-abstention / evidence-grounding rates), each rate
     with a 95% bootstrap confidence interval.
  2. Accuracy by answer type, with exact-match (EM) alongside the lenient
     fuzzy score for extractive items.
  3. Efficiency (CPU-only).
  4. CV-seed stability: the whole cascade re-run under N different fold
     splits, reporting mean +- std of the headline rates.
  5. Negation-cue ablation: generic (headline) vs. benchmark-tuned cue list.
  6. Robustness probes: negation flip (classifier) and query perturbation
     (retrieval).

Usage: python evaluate.py [--config configs/default.yaml] [--skip-sweeps]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data_utils import load_corpus, load_qa, corpus_index_by_id, load_config
from src.retrieval import build_retriever
from src.generator import fuzzy_match, exact_match, set_negation_cues
from src.evaluate_trust import (
    expected_calibration_error, brier_score, hallucination_rate,
    false_abstention_rate, evidence_grounding_rate, bootstrap_ci, accuracy_by_type,
)
from src.robustness import negation_flip_probe, query_perturbation_probe
from train import build_dataset, run_cv

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RETRIEVER_NAMES = ["bm25", "tfidf_lsa"]


def is_correct(rec, strict=False):
    gt, pt = rec["gold_answer_type"], rec["predicted_answer_type"]
    if gt == "unanswerable":
        return pt == "unanswerable"
    if pt != gt:
        return False
    if gt == "yes_no":
        return rec["predicted_value"] == rec["gold_answer"]
    if gt == "extractive":
        match = exact_match if strict else fuzzy_match
        return match(rec["predicted_value"] or "", rec["gold_answer"] or "")
    return False


def reliability_plot(confidences, corrects, out_path, title, n_bins=10):
    ece, bins = expected_calibration_error(confidences, corrects, n_bins=n_bins)
    xs = [(b["lo"] + b["hi"]) / 2 for b in bins if b["count"] > 0]
    accs = [b["acc"] for b in bins if b["count"] > 0]
    confs = [b["conf"] for b in bins if b["count"] > 0]
    counts = [b["count"] for b in bins if b["count"] > 0]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="perfect calibration")
    ax.bar(xs, accs, width=1.0 / n_bins * 0.8, alpha=0.7, label="observed accuracy", color="#4C78A8")
    ax.scatter(confs, accs, color="#F58518", zorder=5, label="bin (conf, acc)")
    for x, a, c in zip(xs, accs, counts):
        ax.annotate(f"n={c}", (x, a), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=6)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE={ece:.3f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return ece, bins


def _rate_with_ci(records, rate_fn, indicator_fn):
    """rate_fn gives (rate, numerator, denominator); indicator_fn maps a
    record in the denominator population to 0/1 so we can bootstrap it."""
    rate, num, den = rate_fn(records)
    if rate is None:
        return {"rate": None, "count": f"{num}/{den}", "ci95": [None, None]}
    pop = [indicator_fn(r) for r in records if indicator_fn(r) is not None]
    _, lo, hi = bootstrap_ci(pop)
    return {"rate": rate, "count": f"{num}/{den}", "ci95": [lo, hi]}


def _hall_ind(r):
    return (r["predicted_answer_type"] != "unanswerable") if r["gold_answer_type"] == "unanswerable" else None


def _fa_ind(r):
    return (r["predicted_answer_type"] == "unanswerable") if r["gold_answer_type"] != "unanswerable" else None


def _ground_ind(r):
    if r["predicted_answer_type"] != "unanswerable" and r["gold_arxiv_id"] is not None:
        return r["top1_id"] == r["gold_arxiv_id"]
    return None


def metrics_from_records(records, n_bins=10):
    """All classifier-side metrics for one list of prediction records."""
    corrects = np.array([1.0 if is_correct(r) else 0.0 for r in records])
    corrects_strict = np.array([1.0 if is_correct(r, strict=True) else 0.0 for r in records])
    confidences = np.array([r["confidence"] for r in records])
    acc, acc_lo, acc_hi = bootstrap_ci(corrects)
    ece, _ = expected_calibration_error(confidences, corrects, n_bins=n_bins)
    return {
        "accuracy": acc, "accuracy_ci95": [acc_lo, acc_hi],
        "accuracy_strict": float(corrects_strict.mean()),
        "accuracy_by_type": accuracy_by_type(records, corrects),
        "accuracy_by_type_strict": accuracy_by_type(records, corrects_strict),
        "ece": ece,
        "brier": brier_score(confidences, corrects),
        "hallucination": _rate_with_ci(records, hallucination_rate, _hall_ind),
        "false_abstention": _rate_with_ci(records, false_abstention_rate, _fa_ind),
        "evidence_grounding": _rate_with_ci(records, evidence_grounding_rate, _ground_ind),
    }


def summarize(retriever_name, n_bins=10):
    with open(RESULTS / f"records_{retriever_name}.json") as f:
        payload = json.load(f)
    run_meta, records = payload["run"], payload["records"]
    with open(RESULTS / f"retrieval_metrics_{retriever_name}.json") as f:
        retrieval = json.load(f)["metrics"]
    with open(RESULTS / f"efficiency_{retriever_name}.json") as f:
        efficiency = {k: v for k, v in json.load(f).items() if k != "run"}

    corrects = np.array([1.0 if is_correct(r) else 0.0 for r in records])
    confidences = np.array([r["confidence"] for r in records])
    reliability_plot(confidences, corrects, RESULTS / f"reliability_{retriever_name}.png",
                     f"Reliability diagram ({retriever_name})", n_bins=n_bins)

    return {"retriever": retriever_name, "run": run_meta,
            **metrics_from_records(records, n_bins=n_bins),
            "retrieval": retrieval, "efficiency": efficiency}


def seed_stability_sweep(qa, docs, docs_by_id, cfg):
    """Re-run the full cascade under several CV fold-split seeds (same
    retrieval features) and report mean/std of the headline rates. This
    separates 'what the pipeline does' from 'which fold split we drew'."""
    out = {}
    for name in RETRIEVER_NAMES:
        retriever = build_retriever(name, docs, n_components=cfg["tfidf_lsa"]["n_components"])
        rows = build_dataset(qa, retriever, docs_by_id, k=cfg["top_k"])
        per_seed = []
        for seed in cfg["robustness"]["seeds"]:
            preds = run_cv(rows, seed=seed, n_splits=cfg["n_splits"])
            m = metrics_from_records(preds, n_bins=cfg["ece_bins"])
            per_seed.append({"seed": seed, "accuracy": m["accuracy"], "ece": m["ece"],
                             "hallucination_rate": m["hallucination"]["rate"],
                             "false_abstention_rate": m["false_abstention"]["rate"]})
        agg = {}
        for key in ("accuracy", "ece", "hallucination_rate", "false_abstention_rate"):
            vals = np.array([p[key] for p in per_seed], dtype=float)
            agg[key] = {"mean": float(vals.mean()), "std": float(vals.std()),
                        "min": float(vals.min()), "max": float(vals.max())}
        out[name] = {"n_seeds": len(per_seed), "per_seed": per_seed, "aggregate": agg}
    return out


def negation_cue_ablation(qa, docs, docs_by_id, cfg):
    """Headline (generic cues) vs. the original benchmark-tuned cue list."""
    out = {}
    try:
        for mode in ("generic", "tuned"):
            set_negation_cues(mode)
            out[mode] = {}
            for name in RETRIEVER_NAMES:
                retriever = build_retriever(name, docs, n_components=cfg["tfidf_lsa"]["n_components"])
                rows = build_dataset(qa, retriever, docs_by_id, k=cfg["top_k"])
                preds = run_cv(rows, seed=cfg["seed"], n_splits=cfg["n_splits"])
                m = metrics_from_records(preds, n_bins=cfg["ece_bins"])
                out[mode][name] = {
                    "accuracy": m["accuracy"],
                    "yes_no_accuracy": m["accuracy_by_type"]["yes_no"]["accuracy"],
                    "hallucination_rate": m["hallucination"]["rate"],
                    "false_abstention_rate": m["false_abstention"]["rate"],
                    "ece": m["ece"],
                }
    finally:
        set_negation_cues(cfg["negation_cues"])
    return out


def _fmt_ci(d):
    if d["rate"] is None:
        return "n/a"
    return f"{d['rate']:.3f} ({d['count']}) [{d['ci95'][0]:.2f}, {d['ci95'][1]:.2f}]"


def write_report(summaries, stability, ablation, negation, perturbation, cfg):
    L = ["# Results summary\n",
         f"Seed {cfg['seed']}, {cfg['n_splits']}-fold stratified CV, k={cfg['top_k']}, "
         f"negation cues = `{cfg['negation_cues']}`. Rates are followed by their count and a 95% "
         "percentile-bootstrap confidence interval (2000 resamples).\n"]
    L.append("| Retriever | Recall@1 | Recall@3 | Recall@5 | MRR | QA acc. [95% CI] | ECE | Brier | Hallucination rate | False-abstention rate | Evidence-grounding rate |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        r = s["retrieval"]
        L.append(
            f"| {s['retriever']} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | {r['recall@5']:.3f} | {r['mrr']:.3f} | "
            f"{s['accuracy']:.3f} [{s['accuracy_ci95'][0]:.2f}, {s['accuracy_ci95'][1]:.2f}] | "
            f"{s['ece']:.3f} | {s['brier']:.3f} | {_fmt_ci(s['hallucination'])} | "
            f"{_fmt_ci(s['false_abstention'])} | {_fmt_ci(s['evidence_grounding'])} |"
        )

    L.append("\n## Accuracy by answer type\n")
    L.append("Extractive items are scored two ways: *fuzzy* (prediction and gold are substrings of each other, "
             "case-insensitive; used in the headline table) and *EM* (exact match after case/hyphen/whitespace "
             "normalisation).\n")
    L.append("| Retriever | yes/no (n) | extractive fuzzy (n) | extractive EM | unanswerable (n) | overall fuzzy | overall EM |")
    L.append("|---|---|---|---|---|---|---|")
    for s in summaries:
        bt, bts = s["accuracy_by_type"], s["accuracy_by_type_strict"]
        L.append(
            f"| {s['retriever']} | {bt['yes_no']['accuracy']:.3f} ({bt['yes_no']['n']}) | "
            f"{bt['extractive']['accuracy']:.3f} ({bt['extractive']['n']}) | {bts['extractive']['accuracy']:.3f} | "
            f"{bt['unanswerable']['accuracy']:.3f} ({bt['unanswerable']['n']}) | "
            f"{s['accuracy']:.3f} | {s['accuracy_strict']:.3f} |"
        )

    L.append("\n## Efficiency (CPU-only)\n")
    L.append("Timings are machine-dependent; treat the *ratio* between retrievers as the result, not the absolute values.\n")
    L.append("| Retriever | Build time (s) | Mean latency (ms) | P95 latency (ms) | Approx. memory (KB) |")
    L.append("|---|---|---|---|---|")
    for s in summaries:
        e = s["efficiency"]
        L.append(f"| {s['retriever']} | {e['build_time_s']:.4f} | {e['mean_latency_ms']:.3f} | "
                 f"{e['p95_latency_ms']:.3f} | {e['approx_memory_kb']:.1f} |")

    if stability:
        n = next(iter(stability.values()))["n_seeds"]
        L.append(f"\n## CV-seed stability ({n} fold-split seeds, mean +- std [min, max])\n")
        L.append("| Retriever | QA acc. | ECE | Hallucination rate | False-abstention rate |")
        L.append("|---|---|---|---|---|")
        for name, st in stability.items():
            a = st["aggregate"]
            cells = [f"{a[k]['mean']:.3f} +- {a[k]['std']:.3f} [{a[k]['min']:.2f}, {a[k]['max']:.2f}]"
                     for k in ("accuracy", "ece", "hallucination_rate", "false_abstention_rate")]
            L.append(f"| {name} | " + " | ".join(cells) + " |")

    if ablation:
        L.append("\n## Negation-cue ablation (seed %d)\n" % cfg["seed"])
        L.append("`generic` is a domain-agnostic cue list written without looking at the benchmark (headline). "
                 "`tuned` is the original list that was assembled by reading the benchmark's yes/no questions and "
                 "contains phrases that each single out one gold-\"No\" item; it is reported only to quantify "
                 "how much that leakage inflated the numbers.\n")
        L.append("| Cue set | Retriever | QA acc. | yes/no acc. | Hallucination rate | False-abstention rate | ECE |")
        L.append("|---|---|---|---|---|---|---|")
        for mode, per in ablation.items():
            for name, m in per.items():
                L.append(f"| {mode} | {name} | {m['accuracy']:.3f} | {m['yes_no_accuracy']:.3f} | "
                         f"{m['hallucination_rate']:.3f} | {m['false_abstention_rate']:.3f} | {m['ece']:.3f} |")

    L.append("\n## Robustness probe 1: negation flip test (BM25 features, full-data yes/no model)\n")
    L.append(f"- Questions probed: {negation['n_probed']}")
    L.append(f"- Flipped prediction on negated paraphrase: {negation['n_flipped_on_negation']}")
    L.append(f"- Negation flip rate: {negation['negation_flip_rate']:.3f} "
             "(higher = more logically consistent; 1.0 = always flips as it should)")

    if perturbation:
        L.append("\n## Robustness probe 2: retrieval under query perturbation\n")
        p0 = next(iter(perturbation.values()))
        L.append(f"Each answerable question is perturbed {p0['perturbed'][0]['n_perturbations']} times per setting "
                 "(`drop` = tokens removed, `typo` = adjacent-character swap inside tokens); Recall@1 is the mean "
                 "over perturbations (+- std).\n")
        header = "| Retriever | clean R@1 | " + " | ".join(
            f"{p['mode']} {int(p['rate']*100)}%" for p in p0["perturbed"]) + " |"
        L.append(header)
        L.append("|" + "---|" * (2 + len(p0["perturbed"])))
        for name, pr in perturbation.items():
            cells = [f"{p['recall@1_mean']:.3f} +- {p['recall@1_std']:.3f}" for p in pr["perturbed"]]
            L.append(f"| {name} | {pr['clean']['recall@1']:.3f} | " + " | ".join(cells) + " |")

    report = "\n".join(L) + "\n"
    with open(RESULTS / "report.md", "w") as f:
        f.write(report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--skip-sweeps", action="store_true",
                    help="skip the seed-stability sweep, cue ablation and perturbation probe (faster)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    set_negation_cues(cfg["negation_cues"])

    summaries = [summarize(name, n_bins=cfg["ece_bins"]) for name in RETRIEVER_NAMES]
    with open(RESULTS / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    docs = load_corpus(ROOT / cfg["data"]["corpus"])
    qa = load_qa(ROOT / cfg["data"]["qa_benchmark"])
    docs_by_id = corpus_index_by_id(docs)
    bm25 = build_retriever("bm25", docs)

    negation = negation_flip_probe(qa, bm25, docs_by_id, k=cfg["top_k"])
    with open(RESULTS / "robustness_negation.json", "w") as f:
        json.dump(negation, f, indent=2)

    stability = ablation = perturbation = None
    if not args.skip_sweeps:
        stability = seed_stability_sweep(qa, docs, docs_by_id, cfg)
        with open(RESULTS / "seed_stability.json", "w") as f:
            json.dump(stability, f, indent=2)
        ablation = negation_cue_ablation(qa, docs, docs_by_id, cfg)
        with open(RESULTS / "ablation_negation_cues.json", "w") as f:
            json.dump(ablation, f, indent=2)
        perturbation = {}
        for name in RETRIEVER_NAMES:
            retriever = build_retriever(name, docs, n_components=cfg["tfidf_lsa"]["n_components"])
            perturbation[name] = query_perturbation_probe(
                qa, retriever, rates=cfg["robustness"]["perturbation_rates"],
                n_perturbations=cfg["robustness"]["n_perturbations"], seed=cfg["seed"], k=cfg["top_k"])
        with open(RESULTS / "robustness_perturbation.json", "w") as f:
            json.dump(perturbation, f, indent=2)

    print(write_report(summaries, stability, ablation, negation, perturbation, cfg))


if __name__ == "__main__":
    main()
