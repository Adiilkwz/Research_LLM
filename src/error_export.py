"""Collect stored failure/success examples of the final sequential runs, add the UNEDITED model's
generation for each portability example, assign a heuristic (automatic) error category, and write a
CSV for MANUAL verification (Appendix B of the report requires >= 30 manually categorised failures).

The manual_label column must be filled by a person; analyze.py uses manual_label when present
and reports auto_label agreement.

Usage (GPU recommended, CPU works but is slow): python src/error_export.py --model gpt2-xl
"""
import argparse
import csv
import glob
import json
import os
import re

from common import load_config, load_model, read_jsonl, results_dir, greedy_generate, ROOT


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (s or "").lower())).strip()


def contains(gen, ans):
    a = norm(ans)
    return bool(a) and a in norm(gen)


def degenerate(gen):
    w = norm(gen).split()
    return len(w) == 0 or (len(w) >= 4 and len(set(w)) / len(w) < 0.4)


def auto_label(row, stream_targets):
    g = row["generation"]
    if degenerate(g):
        return "degenerate_output"
    gold_hit = any(contains(g, a) for a in row["gold"])
    if row["kind"] == "portability":
        if gold_hit:
            return "measurement_artifact_gold_in_generation"
        if row.get("pre_generation") is not None and norm(g) == norm(row["pre_generation"]):
            return "edit_not_propagated_unchanged_output"
        if contains(g, row["target_new"]):
            return "echoes_edited_object_no_second_hop"
        if row.get("target_true") and contains(g, row["target_true"]):
            return "reverts_to_old_fact"
        return "other_wrong_answer"
    if row["kind"] == "retention":
        if gold_hit:
            return "measurement_artifact_gold_in_generation"
        if row.get("target_true") and contains(g, row["target_true"]):
            return "reverted_to_original_fact"
        if any(contains(g, t) for t in stream_targets if t not in row["gold"]):
            return "overwritten_by_other_edit_target"
        return "other_wrong_answer"
    # locality
    if gold_hit:
        return "measurement_artifact_gold_in_generation"
    if any(contains(g, t) for t in stream_targets):
        return "contaminated_by_edited_object"
    return "drift_to_other_answer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no_pre", action="store_true", help="skip unedited-model generations (no GPU)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    base = os.path.join(results_dir(cfg), args.model, "seq")
    edits = {e["case_id"]: e for e in read_jsonl(os.path.join(ROOT, cfg["data_dir"], "edits_test.jsonl"))}

    rows = []
    for run in sorted(glob.glob(os.path.join(base, "*_seed*"))):
        name = os.path.basename(run)
        if name.endswith("_dev") or name.startswith("none_"):
            continue
        method, seed = name.split("_seed")
        ex_path = os.path.join(run, "examples.jsonl")
        if not os.path.exists(ex_path):
            continue
        rc = json.load(open(os.path.join(run, "run_config.json")))
        stream_targets = sorted({edits[c]["target_new"] for c in rc["stream_case_ids"]})
        for r in read_jsonl(ex_path):
            r.update({"model": args.model, "method": method, "seed": int(seed)})
            r["_stream_targets"] = stream_targets
            rows.append(r)

    if not args.no_pre:
        model, tok = load_model(cfg, args.model)
        cache = {}
        for r in rows:
            if r["kind"] == "portability":
                if r["prompt"] not in cache:
                    cache[r["prompt"]] = greedy_generate(model, tok, r["prompt"])
                r["pre_generation"] = cache[r["prompt"]]

    out_path = os.path.join(results_dir(cfg), args.model, "error_analysis_candidates.csv")
    cols = ["model", "method", "seed", "kind", "outcome", "type", "subtype", "case_id", "edit_prompt", "prompt", "gold",
            "target_new", "target_true", "generation", "pre_generation", "auto_label", "manual_label", "note"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            r["auto_label"] = auto_label(r, r.pop("_stream_targets")) if r.get("outcome") == "fail" else "success_example"
            r["gold"] = " | ".join(r["gold"])
            w.writerow({c: r.get(c, "") for c in cols})
    print(f"wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
