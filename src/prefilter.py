"""Pre-edit screening with the UNEDITED model (run once per model, shared by every method and seed).

Portability follows the RippleEdits protocol (Cohen et al., 2024): a ripple test is only executed
if the unedited model passes its condition queries.

Specificity follows EasyEdit / Yao et al. (2023): an out-of-scope query is preserved when the edited
model's argmax prediction (teacher-forced on the gold answer positions) is IDENTICAL to the unedited
model's. This needs the unedited predictions, stored here:
  * relation specificity: every RippleEdits Relation_Specificity test of every edit (first query, first answer)
  * locality: the first K facts of locality_pool.jsonl (fixed shuffled order, same probes for every model)
Whether the unedited model knew each fact (exact match) is stored too (secondary "known-fact" metrics).

Output: results/<model>/prefilter.json
Usage:  python src/prefilter.py --model gpt2-xl
"""
import argparse
import os

from common import (load_config, load_model, ppl_passages, predict_pairs, read_jsonl, results_dir, score_queries,
                    set_seed, test_passes, write_json, Timer, ROOT)


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
            rs_pairs, rs_refs = [], []
            for e in edits:
                for ti, t_ in enumerate(e["rel_spec"]):
                    q0 = t_["queries"][0]
                    rs_pairs.append((q0["prompt"], q0["answers"][0])); rs_refs.append((e["case_id"], ti))
            rs_pred = dict(zip(rs_refs, predict_pairs(model, tok, rs_pairs, bs, mat)))

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
                    "rs_valid": rs_valid,          # fact known (exact match) before the edit
                    "rs_pre_pred": [rs_pred[(cid, ti)] for ti in range(len(e["rel_spec"]))],
                }

        K = int(cfg["locality_probes"])
        loc = read_jsonl(os.path.join(proc, "locality_pool.jsonl"))[:K]
        lres = score_queries(model, tok, loc, bs, mat)
        lpred = predict_pairs(model, tok, [(p["prompt"], p["answers"][0]) for p in loc], bs, mat)
        out["locality"] = {"probe_ids": [p["probe_id"] for p in loc], "pre_pred": lpred,
                           "pre_em": [r["em"] for r in lres]}
        known = [p for p, r in zip(loc, lres) if r["em"]]

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
        "locality_probes": len(out["locality"]["probe_ids"]), "locality_probes_known": len(known),
    }
    out["summary"]["ppl_passages"] = len(ppl_passages(cfg))  # creates data/processed/ppl_passages.jsonl once
    print(out["summary"])
    write_json(os.path.join(results_dir(cfg), args.model, "prefilter.json"), out)


if __name__ == "__main__":
    main()
