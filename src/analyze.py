"""Statistical analysis, tables and figures for Assignment 4. CPU only (no torch import).

Reads results/<model>/{prefilter.json, seq/*, single/*} and writes results/analysis/:
  main_results.csv / .md     every condition x primary & secondary metrics (mean +/- SD over seeds, 95% CI)
  h1_portability.csv         H1 tests (Wilcoxon signed-rank, Holm)
  h2_locality.csv            H2 tests (per-seed Wilcoxon signed-rank on per-probe LOC, N=1 vs N=n_max; slopes vs log10 N)
  ro4_pairwise.csv           ERS pairwise comparisons (stratified paired bootstrap, Holm)
  subgroups.csv              portability by type / entity popularity / prior knowledge; RS
  efficiency.csv             seconds per edit, total hours, peak memory
  curves.csv                 ER/LOC/PPL per milestone
  error_table.csv            error categories (manual labels if present, else automatic)
  numbers.json               every number above, for traceability
  fig*.png                   figures

Usage: python src/analyze.py [--config configs/experiment.yaml]
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
import yaml
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METHODS = ["none", "FT", "ROME", "MEMIT"]
LABEL = {"none": "No edit", "FT": "FT-M", "ROME": "ROME", "MEMIT": "MEMIT"}
COLOR = {"none": "#8a8a8a", "FT": "#d1495b", "ROME": "#2e86ab", "MEMIT": "#3b8b5a"}
PTYPES = ["two_hop", "inverse_relation", "other_logical", "alias"]


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def hm(a, b):
    return 0.0 if a + b == 0 else 2 * a * b / (a + b)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def boot_ci(x, B, rng, stat=np.mean):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (np.nan, np.nan)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def wilcoxon_paired(before, after):
    """Wilcoxon signed-rank test on paired per-probe scores (N=1 vs N=n_max).
    Returns (p, matched-pairs rank-biserial r (positive = decline), number of probes that changed)."""
    d = np.asarray(before, float) - np.asarray(after, float)
    nz = d[d != 0]
    if len(nz) == 0:
        return 1.0, 0.0, 0
    res = stats.wilcoxon(nz)
    ranks = stats.rankdata(np.abs(nz))
    r_rb = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()
    return float(res.pvalue), float(r_rb), int(len(nz))


def fmt(m, s=None, pct=True):
    k = 100 if pct else 1
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "n/a"
    return f"{m * k:.1f}" + ("" if s is None or np.isnan(s) else f" ± {s * k:.1f}")


# ----------------------------------------------------------------------------------------------
def load_runs(res_dir, model):
    runs = {}
    for d in sorted(glob.glob(os.path.join(res_dir, model, "seq", "*_seed*"))):
        name = os.path.basename(d)
        if name.endswith("_dev"):
            continue
        method, seed = name.split("_seed")
        if not os.path.exists(os.path.join(d, "milestones.jsonl")):
            continue
        runs[(method, int(seed))] = {
            "edits": [r for r in read_jsonl(os.path.join(d, "edits.jsonl")) if not r.get("diverged")],
            "ms": {m["n"]: m for m in read_jsonl(os.path.join(d, "milestones.jsonl"))},
            "summary": json.load(open(os.path.join(d, "run_summary.json"))) if os.path.exists(os.path.join(d, "run_summary.json")) else {},
            "config": json.load(open(os.path.join(d, "run_config.json"))),
        }
    single = {}
    for d in sorted(glob.glob(os.path.join(res_dir, model, "single", "*_seed*"))):
        method = os.path.basename(d).split("_seed")[0]
        if os.path.exists(os.path.join(d, "edits.jsonl")):
            single[method] = read_jsonl(os.path.join(d, "edits.jsonl"))
    return runs, single


def port_types(rec):
    out = defaultdict(list)
    for p in rec["port"]:
        key = p["subtype"] if p["type"] == "reverse" else p["type"]
        out[key].append(p["pass"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "experiment.yaml"))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    res_dir = os.path.join(ROOT, cfg.get("results_dir", "results"))
    out_dir = os.path.join(res_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    B = int(cfg["analysis"]["bootstrap"])
    rng = np.random.default_rng(2026)
    edits_meta = {e["case_id"]: e for e in read_jsonl(os.path.join(ROOT, cfg["data_dir"], "edits_test.jsonl"))}
    loc_meta = {p["probe_id"]: p for p in read_jsonl(os.path.join(ROOT, cfg["data_dir"], "locality_pool.jsonl"))}
    models = [m for m in cfg["models"] if os.path.isdir(os.path.join(res_dir, m, "seq"))]
    numbers = {}
    main_rows, h1_rows, h2_rows, ro4_rows, sub_rows, eff_rows, curve_rows = [], [], [], [], [], [], []

    for model in models:
        runs, single = load_runs(res_dir, model)
        pf = json.load(open(os.path.join(res_dir, model, "prefilter.json")))
        numbers[model] = {"prefilter": pf.get("summary", {})}
        seeds = sorted({s for (_, s) in runs})
        nmax_by_run = {k: max(v["ms"]) for k, v in runs.items()}
        methods = [m for m in METHODS if any(k[0] == m for k in runs)]

        # ---------------- main table -----------------
        for method in methods:
            per_seed = defaultdict(list)
            pooled = defaultdict(list)
            for s in seeds:
                if (method, s) not in runs:
                    continue
                r = runs[(method, s)]
                n_end = nmax_by_run[(method, s)]
                ms = r["ms"][n_end]
                eff = [e["eff_em"] for e in r["edits"]]
                port_all = [p["pass"] for e in r["edits"] for p in e["port"]]
                rs = [x["agree"] for e in r["edits"] for x in e["rs"]]                       # prediction unchanged
                rs_known = [x["pass"] for e in r["edits"] for x in e["rs"] if x["known"]]   # known fact still correct
                pre_em = pf["locality"]["pre_em"]
                loc_known = [v for v, k in zip(ms["loc_em"], pre_em) if k]
                per_seed["eff"].append(np.mean(eff)); pooled["eff"] += eff
                per_seed["port"].append(np.mean(port_all) if port_all else np.nan); pooled["port"] += port_all
                for t in PTYPES:
                    v = [p for e in r["edits"] for p in port_types(e).get(t, [])]
                    per_seed["port_" + t].append(np.mean(v) if v else np.nan); pooled["port_" + t] += v
                per_seed["rs"].append(np.mean(rs) if rs else np.nan); pooled["rs"] += rs
                per_seed["rs_known"].append(np.mean(rs_known) if rs_known else np.nan); pooled["rs_known"] += rs_known
                per_seed["LOC_known"].append(np.mean(loc_known) if loc_known else np.nan); pooled["LOC_known"] += loc_known
                per_seed["ER"].append(ms["ER"]); pooled["ER"] += ms["er"]
                per_seed["LOC"].append(ms["LOC"]); pooled["LOC"] += ms["loc"]
                per_seed["ERS"].append(hm(ms["ER"], ms["LOC"]))
                per_seed["PPL"].append(ms["ppl"])
                per_seed["n_end"].append(n_end)
                per_seed["edit_s"].append(np.mean([e["edit_s"] for e in r["edits"]]))
            if not per_seed:
                continue
            row = {"model": model, "method": LABEL[method], "seeds": len(per_seed["ER"]), "N": int(np.min(per_seed["n_end"]))}
            for k in ["ERS", "eff", "port", "port_two_hop", "port_inverse_relation", "port_other_logical", "port_alias",
                      "rs", "rs_known", "ER", "LOC", "LOC_known"]:
                v = np.array(per_seed[k], float)
                m_, s_ = np.nanmean(v), (np.nanstd(v, ddof=1) if np.sum(~np.isnan(v)) > 1 else np.nan)
                ci = boot_ci(pooled[k], B // 5, rng) if k in pooled else (np.nan, np.nan)
                row[k] = fmt(m_, s_)
                row[k + "_ci"] = f"[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]" if not np.isnan(ci[0]) else ""
                row[k + "_n"] = len(pooled.get(k, []))
                numbers[model].setdefault(method, {})[k] = {"mean": m_, "sd": s_, "ci95": ci, "per_seed": v.tolist(),
                                                            "n_items_pooled": len(pooled.get(k, []))}
            v = np.array(per_seed["PPL"]); row["PPL"] = f"{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}"
            numbers[model][method]["PPL"] = {"mean": float(v.mean()), "per_seed": v.tolist()}
            main_rows.append(row)

            eff_rows.append({"model": model, "method": LABEL[method],
                             "mean_s_per_edit": round(float(np.mean(per_seed["edit_s"])), 2),
                             "hours_per_run": round(float(np.mean([sum(e["edit_s"] + e.get("eval_s", 0) for e in runs[(method, s)]["edits"])
                                                                   for s in seeds if (method, s) in runs])) / 3600, 2),
                             "peak_gpu_mem_gb": max([runs[(method, s)]["summary"].get("peak_gpu_mem_gb", np.nan)
                                                     for s in seeds if (method, s) in runs] or [np.nan]),
                             "n_edit_errors": int(sum(sum(1 for e in runs[(method, s)]["edits"] if e.get("edit_error"))
                                                      for s in seeds if (method, s) in runs)),
                             "status": ";".join(sorted({runs[(method, s)]["summary"].get("status", "?") for s in seeds if (method, s) in runs}))})

            # curves
            for s in seeds:
                if (method, s) in runs:
                    for n, ms in sorted(runs[(method, s)]["ms"].items()):
                        curve_rows.append({"model": model, "method": method, "seed": s, "n": n, "ER": ms["ER"],
                                           "LOC": ms["LOC"], "ERS": hm(ms["ER"], ms["LOC"]), "PPL": ms["ppl"]})

        # ---------------- H1: immediate efficacy vs portability (paired per edit) -----------------
        for method in [m for m in methods if m != "none"]:
            diffs, effs, ports = [], [], []
            by_type = defaultdict(lambda: ([], []))
            for s in seeds:
                if (method, s) not in runs:
                    continue
                for e in runs[(method, s)]["edits"]:
                    if not e["port"]:
                        continue
                    pm = np.mean([p["pass"] for p in e["port"]])
                    effs.append(e["eff_em"]); ports.append(pm); diffs.append(e["eff_em"] - pm)
                    for t, v in port_types(e).items():
                        by_type[t][0].append(e["eff_em"]); by_type[t][1].append(np.mean(v))
            if not diffs:
                continue
            d = np.array(diffs)
            nz = d[d != 0]
            W, p = (stats.wilcoxon(nz).statistic, stats.wilcoxon(nz).pvalue) if len(nz) > 0 else (np.nan, 1.0)
            r_rb = (np.sum(nz > 0) - np.sum(nz < 0)) / len(nz) if len(nz) else np.nan  # matched-pairs rank-biserial (sign-based)
            ci = boot_ci(d, B, rng)
            h1_rows.append({"model": model, "method": LABEL[method], "n_edit_obs": len(d), "eff_imm": np.mean(effs),
                            "portability": np.mean(ports), "diff_pp": d.mean() * 100,
                            "diff_ci95": f"[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]", "W": W, "p": p, "r_rb": r_rb,
                            **{f"port_{t}": (np.mean(by_type[t][1]) if by_type[t][1] else np.nan) for t in PTYPES}})
        if h1_rows:
            todo = [r for r in h1_rows if r["model"] == model]
            for r_, ph in zip(todo, holm([r["p"] for r in todo])):
                r_["p_holm"] = ph

        # ---------------- H2: locality decline with edit volume -----------------
        for method in [m for m in methods if m != "none"]:
            ps, slopes, er_slopes, drops, rbs, n_changed = [], [], [], [], [], []
            for s in seeds:
                if (method, s) not in runs:
                    continue
                ms = runs[(method, s)]["ms"]
                ns = sorted(ms)
                p, r_rb, n_nz = wilcoxon_paired(ms[ns[0]]["loc"], ms[ns[-1]]["loc"])
                ps.append(p); rbs.append(r_rb); n_changed.append(n_nz)
                drops.append(ms[ns[0]]["LOC"] - ms[ns[-1]]["LOC"])
                x = np.log10(ns)
                slopes.append(stats.linregress(x, [ms[n]["LOC"] for n in ns]).slope)
                er_slopes.append(stats.linregress(x, [ms[n]["ER"] for n in ns]).slope)
            if not ps:
                continue
            single_loc = (float(np.mean([np.mean(r["loc"]) for r in single[method]])) if method in single else np.nan)
            h2_rows.append({"model": model, "method": LABEL[method], "seeds": len(ps),
                            "LOC_drop_pp_mean": np.mean(drops) * 100, "LOC_drop_pp_sd": (np.std(drops, ddof=1) * 100 if len(drops) > 1 else np.nan),
                            "slope_LOC_per_decade_pp": np.mean(slopes) * 100,
                            "slope_LOC_sd_pp": (np.std(slopes, ddof=1) * 100 if len(slopes) > 1 else np.nan),
                            "slope_ER_per_decade_pp": np.mean(er_slopes) * 100,
                            "probes_changed_N1_vs_Nmax": "; ".join(str(n) for n in n_changed),
                            "wilcoxon_p_per_seed": "; ".join(f"{p:.2g}" for p in ps),
                            "rank_biserial_per_seed": "; ".join(f"{r:.2f}" for r in rbs),
                            "p_iut": max(ps),  # intersection-union: significant only if every seed is
                            "all_slopes_negative": bool(all(s_ < 0 for s_ in slopes)),
                            "LOC_single_edit_mean": single_loc})
        todo = [r for r in h2_rows if r["model"] == model]
        for r_, ph in zip(todo, holm([r["p_iut"] for r in todo])):
            r_["p_iut_holm"] = ph

        # ---------------- RO4: pairwise ERS at n_max (stratified paired bootstrap) -----------------
        pairs = [("MEMIT", "ROME"), ("MEMIT", "FT"), ("ROME", "FT")]
        tmp = []
        for a, b in pairs:
            common = [s for s in seeds if (a, s) in runs and (b, s) in runs]
            if not common:
                continue
            obs, boots = [], np.zeros(B)
            for s in common:
                ma = runs[(a, s)]["ms"][nmax_by_run[(a, s)]]; mb = runs[(b, s)]["ms"][nmax_by_run[(b, s)]]
                if len(ma["er"]) != len(mb["er"]):
                    continue  # unequal stream length (e.g. a diverged run): not comparable
                era, erb = np.array(ma["er"]), np.array(mb["er"]); la, lb = np.array(ma["loc"]), np.array(mb["loc"])
                obs.append(hm(era.mean(), la.mean()) - hm(erb.mean(), lb.mean()))
                ie = rng.integers(0, len(era), (B, len(era))); il = rng.integers(0, len(la), (B, len(la)))
                A = 2 * era[ie].mean(1) * la[il].mean(1) / np.maximum(era[ie].mean(1) + la[il].mean(1), 1e-12)
                Bv = 2 * erb[ie].mean(1) * lb[il].mean(1) / np.maximum(erb[ie].mean(1) + lb[il].mean(1), 1e-12)
                boots += (A - Bv)
            if not obs:
                continue
            boots /= len(obs)
            p = float(min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean())))
            tmp.append({"model": model, "comparison": f"{LABEL[a]} vs {LABEL[b]}", "seeds": len(obs),
                        "ERS_diff_pp": float(np.mean(obs) * 100),
                        "ERS_diff_sd_pp": float(np.std(obs, ddof=1) * 100) if len(obs) > 1 else np.nan,
                        "ci95_pp": f"[{np.percentile(boots, 2.5) * 100:.1f}, {np.percentile(boots, 97.5) * 100:.1f}]",
                        "p_boot": max(p, 1.0 / B), "same_sign_all_seeds": bool(all(o > 0 for o in obs) or all(o < 0 for o in obs))})
        ro4_rows += tmp

        # ---------------- subgroups -----------------
        for method in [m for m in methods if m != "none"]:
            groups = defaultdict(list)
            for s in seeds:
                if (method, s) not in runs:
                    continue
                for e in runs[(method, s)]["edits"]:
                    meta = edits_meta[e["case_id"]]; pfe = pf["edits"][e["case_id"]]
                    groups[("efficacy", "source=" + meta["source"])].append(e["eff_em"])
                    groups[("efficacy", "true_fact_known=" + str(pfe["true_known"]))].append(e["eff_em"])
                    for p in e["port"]:
                        t = p["subtype"] if p["type"] == "reverse" else p["type"]
                        groups[("portability", "type=" + t)].append(p["pass"])
                        groups[("portability", "source=" + meta["source"])].append(p["pass"])
                        groups[("portability", "true_fact_known=" + str(pfe["true_known"]))].append(p["pass"])
                        groups[("portability", "edit_succeeded=" + str(e["eff_em"]))].append(p["pass"])
                    for x in e["rs"]:
                        groups[("rel_specificity_unchanged", "true_fact_known=" + str(pfe["true_known"]))].append(x["agree"])
                    pos = e["i"]
                    bucket = "1-100" if pos <= 100 else ("101-500" if pos <= 500 else "501+")
                    groups[("efficacy", "stream_position=" + bucket)].append(e["eff_em"])
                    for p in e["port"]:
                        groups[("portability", "stream_position=" + bucket)].append(p["pass"])
                # locality at n_max by probe kind / relation shared with an edit in the stream
                ms = runs[(method, s)]["ms"][nmax_by_run[(method, s)]]
                rels = {edits_meta[c]["relation"] for c in runs[(method, s)]["config"]["stream_case_ids"]}
                for pid, v in zip(pf["locality"]["probe_ids"], ms["loc"]):
                    pm = loc_meta[pid]
                    groups[("locality@Nmax", "probe_kind=" + pm["kind"])].append(v)
                    groups[("locality@Nmax", "relation_edited_in_stream=" + str(pm.get("relation") in rels))].append(v)
            for (metric, g), v in sorted(groups.items()):
                ci = boot_ci(v, B // 10, rng)
                sub_rows.append({"model": model, "method": LABEL[method], "metric": metric, "subgroup": g, "n": len(v),
                                 "mean_pct": round(100 * float(np.mean(v)), 1), "ci95": f"[{ci[0] * 100:.1f}, {ci[1] * 100:.1f}]"})

        # ---------------- figures -----------------
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for ax, key, ttl in zip(axes, ["ER", "LOC", "PPL"], ["Edit retention (ER)", "Locality: untouched predictions unchanged (LOC)", "Perplexity (wikitext-103)"]):
            for method in methods:
                cr = [c for c in curve_rows if c["model"] == model and c["method"] == method]
                ns = sorted({c["n"] for c in cr})
                if not ns:
                    continue
                M = np.array([[c[key] for c in cr if c["n"] == n] for n in ns], dtype=object)
                mu = np.array([np.mean(x) for x in M], float); sd = np.array([np.std(x) for x in M], float)
                k = 1 if key == "PPL" else 100
                ax.plot(ns, mu * k, marker="o", ms=3, color=COLOR[method], label=LABEL[method])
                ax.fill_between(ns, (mu - sd) * k, (mu + sd) * k, color=COLOR[method], alpha=0.15)
            ax.set_xscale("log"); ax.set_xlabel("Number of sequential edits N (log scale)")
            ax.set_ylabel("%" if key != "PPL" else "perplexity"); ax.set_title(ttl); ax.grid(alpha=0.3)
        axes[0].legend(frameon=False)
        fig.suptitle(f"{model}: sequential editing (mean ± SD over seeds)")
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f"fig1_curves_{model}.png"), dpi=200); plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.2))
        cats = ["eff", "port_two_hop", "port_inverse_relation", "port_other_logical", "port_alias", "rs"]
        names = ["Efficacy\n(immediate)", "Two-hop", "Inverse\nrelation", "Other\nlogical", "Subject\nalias", "Relation specif.\n(unchanged)"]
        w = 0.8 / max(len(methods), 1)
        for j, method in enumerate(methods):
            vals = [numbers[model].get(method, {}).get(c, {}).get("mean", np.nan) * 100 for c in cats]
            cis = [numbers[model].get(method, {}).get(c, {}).get("ci95", (np.nan, np.nan)) for c in cats]
            err = np.array([[v - ci[0] * 100, ci[1] * 100 - v] for v, ci in zip(vals, cis)]).T
            ax.bar(np.arange(len(cats)) + j * w - 0.4 + w / 2, vals, w, color=COLOR[method], label=LABEL[method],
                   yerr=np.nan_to_num(err), capsize=2)
        ax.set_xticks(np.arange(len(cats))); ax.set_xticklabels(names); ax.set_ylabel("% correct (95% bootstrap CI)")
        ax.set_title(f"{model}: immediate efficacy vs. portability and specificity"); ax.legend(frameon=False, ncol=4)
        ax.grid(axis="y", alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"fig2_portability_{model}.png"), dpi=200); plt.close(fig)

    # ---------------- error analysis table -----------------
    err_rows = []
    for model in models:
        p = os.path.join(res_dir, model, "error_analysis_candidates.csv")
        if not os.path.exists(p):
            continue
        rows = [r for r in csv.DictReader(open(p, encoding="utf-8")) if r["outcome"] == "fail"]
        manual = [r for r in rows if r.get("manual_label", "").strip()]
        use = manual if manual else rows
        key = "manual_label" if manual else "auto_label"
        agree = (np.mean([r["manual_label"].strip() == r["auto_label"] for r in manual]) if manual else np.nan)
        cnt = defaultdict(int)
        ex = {}
        for r in use:
            k = (r["kind"], r[key].strip())
            cnt[k] += 1
            ex.setdefault(k, f"{r['method']}: '{r['prompt']}' -> '{r['generation'][:60]}' (gold: {r['gold'][:40]})")
        tot = sum(cnt.values())
        for (kind, lab), c in sorted(cnt.items(), key=lambda x: (x[0][0], -x[1])):
            err_rows.append({"model": model, "label_source": key, "kind": kind, "error_type": lab, "count": c,
                             "pct_of_errors": round(100 * c / tot, 1), "example": ex[(kind, lab)],
                             "auto_manual_agreement": (round(agree, 3) if manual else "")})
        numbers.setdefault(model, {})["error_analysis"] = {"n_failures_labelled": tot, "label_source": key,
                                                           "auto_manual_agreement": agree}

    def dump(rows, name):
        if not rows:
            return
        cols = list(dict.fromkeys(k for r in rows for k in r))
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
            for r in rows:
                w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    for rows, name in ((main_rows, "main_results.csv"), (h1_rows, "h1_portability.csv"), (h2_rows, "h2_locality.csv"),
                       (ro4_rows, "ro4_pairwise.csv"), (sub_rows, "subgroups.csv"), (eff_rows, "efficiency.csv"),
                       (curve_rows, "curves.csv"), (err_rows, "error_table.csv")):
        dump(rows, name)
    # Holm across all RO4 comparisons
    if ro4_rows:
        for r_, ph in zip(ro4_rows, holm([r["p_boot"] for r in ro4_rows])):
            r_["p_holm"] = ph
        dump(ro4_rows, "ro4_pairwise.csv")
    numbers["h1"], numbers["h2"], numbers["ro4"] = h1_rows, h2_rows, ro4_rows
    numbers["efficiency"] = eff_rows

    # markdown version of the main table
    if main_rows:
        cols = ["model", "method", "seeds", "N", "ERS", "eff", "port", "port_two_hop", "port_inverse_relation",
                "port_alias", "rs", "rs_known", "ER", "LOC", "LOC_known", "PPL"]
        with open(os.path.join(out_dir, "main_results.md"), "w") as f:
            f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
            for r in main_rows:
                f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")

    # ERS bar figure across models
    if main_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        for i, model in enumerate(models):
            for j, method in enumerate([m for m in METHODS if m in numbers.get(model, {})]):
                v = numbers[model][method]["ERS"]
                ax.bar(i + j * 0.2 - 0.3, v["mean"] * 100, 0.2, yerr=(v["sd"] * 100 if not np.isnan(v["sd"]) else 0),
                       color=COLOR[method], label=LABEL[method] if i == 0 else None, capsize=3)
        ax.set_xticks(range(len(models))); ax.set_xticklabels(models); ax.set_ylabel("ERS at N = n_max (%)")
        ax.set_title("Primary metric: Editing Reliability Score (mean ± SD over seeds)")
        ax.legend(frameon=False); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig3_ers.png"), dpi=200); plt.close(fig)

    with open(os.path.join(out_dir, "numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("analysis written to", out_dir)


if __name__ == "__main__":
    main()
