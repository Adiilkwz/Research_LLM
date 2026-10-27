"""Pre-edit screening with the UNEDITED model (run once per model, shared by every method and seed).

Follows the RippleEdits protocol (Cohen et al., 2024): a ripple test is only executed if the
unedited model passes its condition queries; a relation-specificity test is only counted if the
unedited model already knew the fact. The global locality probe set = the first K candidate facts
(fixed shuffled order from prepare_data.py) that the unedited model answers correctly.

Output: results/<model>/prefilter.json
Usage:  python src/prefilter.py --model gpt2-xl
"""
import argparse
import os

from common import (load_config, load_model, ppl_passages, read_jsonl, results_dir, score_queries, set_seed,
                    test_passes, write_json, Timer, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(0)
    model, tok = load_model(cfg, args.model)
    bs = cfg["models"][args.model]["eval_batch"]
    mat = cfg["max_answer_tokens"]
    proc = os.path.join(ROOT, cfg["data_dir"])

    out = {"model": args.model, "edits": {}, "locality": {}}
    with Timer() as t:
        for split in ("test", "dev"):
            edits = read_jsonl(os.path.join(proc, f"edits_{split}.jsonl"))
            # gather every query that has to be scored, then score in one batched pass
            queries, refs = [], []

            def add(q, ref):
                queries.append(q); refs.append(ref)

            for e in edits:
                cid = e["case_id"]
                add({"prompt": e["prompt"], "answers": [e["target_true"]]}, (cid, "true"))
                add({"prompt": e["prompt"], "answers": [e["target_new"]]}, (cid, "new"))
                for ti, t_ in enumerate(e["portability"]):
                    for ci, c in enumerate(t_["conditions"]):
                        add(c, (cid, "pcond", ti, ci))
                    for qi, q in enumerate(t_["queries"]):
                        add(q, (cid, "pquery", ti, qi))
                for ti, t_ in enumerate(e["rel_spec"]):
                    for qi, q in enumerate(t_["queries"]):
                        add(q, (cid, "rs", ti, qi))
            res = score_queries(model, tok, queries, bs, mat)
            byref = {r: s for r, s in zip(refs, res)}

            for e in edits:
                cid = e["case_id"]
                port_valid, port_pre = [], []
                for ti, t_ in enumerate(e["portability"]):
                    conds = [byref[(cid, "pcond", ti, ci)] for ci in range(len(t_["conditions"]))]
                    ok = True if not conds else bool(test_passes(conds, t_["condition_mode"]))
                    port_valid.append(int(ok))
                    port_pre.append(test_passes([byref[(cid, "pquery", ti, qi)] for qi in range(len(t_["queries"]))],
                                                t_["condition_mode"]))
                rs_valid = []
                for ti, t_ in enumerate(e["rel_spec"]):
                    rs_valid.append(test_passes([byref[(cid, "rs", ti, qi)] for qi in range(len(t_["queries"]))],
                                                t_["condition_mode"]))
                out["edits"][cid] = {
                    "split": split,
                    "true_known": byref[(cid, "true")]["em"],
                    "new_known": byref[(cid, "new")]["em"],
                    "port_valid": port_valid,      # condition satisfied -> test is executed
                    "port_pre": port_pre,          # unedited model already gives the post-edit answer
                    "rs_valid": rs_valid,          # fact known before the edit -> counted
                }

        loc = read_jsonl(os.path.join(proc, "locality_pool.jsonl"))
        lres = score_queries(model, tok, loc, bs, mat)
        known = [p["probe_id"] for p, r in zip(loc, lres) if r["em"]]
        K = int(cfg["locality_probes"])
        out["locality"] = {"candidates": len(loc), "known": len(known), "probe_ids": known[:K]}

    e = out["edits"]
    test_ids = [k for k, v in e.items() if v["split"] == "test"]
    out["summary"] = {
        "seconds": round(t.s, 1),
        "test_edits": len(test_ids),
        "true_fact_known_rate": round(sum(e[k]["true_known"] for k in test_ids) / len(test_ids), 4),
        "new_fact_known_rate": round(sum(e[k]["new_known"] for k in test_ids) / len(test_ids), 4),
        "portability_tests_total": sum(len(e[k]["port_valid"]) for k in test_ids),
        "portability_tests_valid": sum(sum(e[k]["port_valid"]) for k in test_ids),
        "rel_spec_tests_total": sum(len(e[k]["rs_valid"]) for k in test_ids),
        "rel_spec_tests_valid": sum(sum(e[k]["rs_valid"]) for k in test_ids),
        "locality_known": len(known), "locality_used": len(out["locality"]["probe_ids"]),
    }
    out["summary"]["ppl_passages"] = len(ppl_passages(cfg))  # creates data/processed/ppl_passages.jsonl once
    print(out["summary"])
    write_json(os.path.join(results_dir(cfg), args.model, "prefilter.json"), out)


if __name__ == "__main__":
    main()
