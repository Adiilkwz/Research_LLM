"""Run one (model, method, seed) condition.

Sequential mode (default): apply n_max edits one after another to the SAME model; after every edit
measure the edit itself, its portability (ripple) tests and its relation-specificity tests
("immediate" metrics); at every milestone N measure edit retention over all N edits so far (ER),
the global locality probe set (LOC) and perplexity (PPL).

Single mode (--mode single): each edit is applied to the unedited model, evaluated (immediate
metrics + LOC) and then reverted, i.e. the classical N=1 setting.

Examples:
  python src/run_experiment.py --model gpt2-xl --method ROME --seed 0
  python src/run_experiment.py --model gpt2-xl --method ROME --seed 0 --resume
  python src/run_experiment.py --model gpt2-xl --method MEMIT --seed 0 --split dev --n_max 5   # pilot
  python src/run_experiment.py --model gpt2-xl --method FT --mode single
"""
import argparse
import os
import platform
import shutil

import numpy as np
import torch
import transformers

from common import (Timer, append_jsonl, greedy_generate, load_config, load_model, perplexity, ppl_passages,
                    read_jsonl, results_dir, score_queries, set_seed, test_passes, write_json, ROOT)
from editors import Editor, load_hparams


def build_stream(cfg, split, seed, n):
    pool = read_jsonl(os.path.join(ROOT, cfg["data_dir"], f"edits_{split}.jsonl"))
    order = np.random.default_rng(seed).permutation(len(pool))[:n]
    return [pool[i] for i in order]


class Evaluator:
    def __init__(self, cfg, model_key, model, tok, prefilter, probes, passages):
        self.model, self.tok, self.pf, self.probes, self.passages = model, tok, prefilter, probes, passages
        self.bs = cfg["models"][model_key]["eval_batch"]
        self.mat = cfg["max_answer_tokens"]
        self.ppl_max = cfg["ppl_max_tokens"]

    def immediate(self, e):
        pf = self.pf["edits"][e["case_id"]]
        qs, tags = [], []
        qs.append({"prompt": e["prompt"], "answers": [e["target_new"]]}); tags.append(("eff",))
        qs.append({"prompt": e["prompt"], "answers": [e["target_true"]]}); tags.append(("true",))
        for ti, t in enumerate(e["portability"]):
            if pf["port_valid"][ti]:
                for qi, q in enumerate(t["queries"]):
                    qs.append(q); tags.append(("port", ti, qi))
        for ti, t in enumerate(e["rel_spec"]):
            if pf["rs_valid"][ti]:
                for qi, q in enumerate(t["queries"]):
                    qs.append(q); tags.append(("rs", ti, qi))
        res = score_queries(self.model, self.tok, qs, self.bs, self.mat)
        by = dict(zip(tags, res))
        rec = {"eff_em": by[("eff",)]["em"], "eff_tok_acc": by[("eff",)]["tok_acc"], "eff_logp": by[("eff",)]["logp"],
               "true_em": by[("true",)]["em"], "port": [], "rs": []}
        for ti, t in enumerate(e["portability"]):
            if pf["port_valid"][ti]:
                r = [by[("port", ti, qi)] for qi in range(len(t["queries"]))]
                rec["port"].append({"ti": ti, "type": t["type"], "subtype": t.get("subtype"),
                                    "pass": test_passes(r, t["condition_mode"]), "pre": pf["port_pre"][ti],
                                    "tok_acc": max(x["tok_acc"] for x in r)})
        for ti, t in enumerate(e["rel_spec"]):
            if pf["rs_valid"][ti]:
                r = [by[("rs", ti, qi)] for qi in range(len(t["queries"]))]
                rec["rs"].append({"ti": ti, "pass": test_passes(r, t["condition_mode"])})
        return rec

    def retention(self, edits):
        res = score_queries(self.model, self.tok, [{"prompt": e["prompt"], "answers": [e["target_new"]]} for e in edits],
                            self.bs, self.mat)
        return [r["em"] for r in res], [r["tok_acc"] for r in res]

    def locality(self):
        return [r["em"] for r in score_queries(self.model, self.tok, self.probes, self.bs, self.mat)]

    def ppl(self):
        return perplexity(self.model, self.tok, self.passages, self.ppl_max)


def env_info():
    info = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "platform": platform.platform()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--method", required=True, choices=["none", "FT", "ROME", "MEMIT"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="seq", choices=["seq", "single"])
    ap.add_argument("--split", default="test", choices=["test", "dev"])
    ap.add_argument("--n_max", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_max = args.n_max or (cfg["n_max"] if args.mode == "seq" else cfg["n_single"])
    milestones = sorted({m for m in cfg["milestones"] if m <= n_max} | {n_max})
    name = f"{args.method}_seed{args.seed}" + ("" if args.split == "test" else f"_{args.split}")
    run_dir = os.path.join(results_dir(cfg), args.model, args.mode, name)
    progress_path = os.path.join(run_dir, "progress.json")
    if os.path.exists(progress_path) and not args.resume:
        if not args.overwrite:
            raise SystemExit(f"{run_dir} already has results. Use --resume to continue or --overwrite to restart.")
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    set_seed(args.seed)
    model, tok = load_model(cfg, args.model)
    pf = __import__("json").load(open(os.path.join(results_dir(cfg), args.model, "prefilter.json")))
    pool = {p["probe_id"]: p for p in read_jsonl(os.path.join(ROOT, cfg["data_dir"], "locality_pool.jsonl"))}
    probes = [pool[i] for i in pf["locality"]["probe_ids"]]
    ev = Evaluator(cfg, args.model, model, tok, pf, probes, ppl_passages(cfg))
    stream = build_stream(cfg, args.split, args.seed, n_max)

    editor = None
    if args.method != "none":
        hp = load_hparams(cfg, args.model, args.method)
        editor = Editor(args.method, model, tok, hp, os.path.join(run_dir, "editor_stdout.log"))
        hp_dict = hp.to_dict()
    else:
        hp_dict = {}

    write_json(os.path.join(run_dir, "run_config.json"), {
        "model": args.model, "method": args.method, "seed": args.seed, "mode": args.mode, "split": args.split,
        "n_max": n_max, "milestones": milestones, "hparams": hp_dict, "env": env_info(),
        "stream_case_ids": [e["case_id"] for e in stream], "n_locality_probes": len(probes)})

    edits_path = os.path.join(run_dir, "edits.jsonl")
    ms_path = os.path.join(run_dir, "milestones.jsonl")
    fail_path = os.path.join(run_dir, "examples.jsonl")
    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    start = 0
    if args.resume and os.path.exists(progress_path):
        prog = __import__("json").load(open(progress_path))
        start = prog["done"]
        if editor is not None and start > 0:
            ck = torch.load(ckpt_path, map_location="cpu")
            assert ck["done"] == start
            editor.load_state(ck["state"])
            Editor.set_template_caches(ck["templates"])
        # drop log rows written after the last checkpoint
        for p, key in ((edits_path, "i"), (ms_path, "n"), (fail_path, "i")):
            if os.path.exists(p):
                rows = [r for r in read_jsonl(p) if r.get(key, 0) <= start]
                open(p, "w").close()
                for r in rows:
                    append_jsonl(p, r)
        print(f"Resuming {run_dir} after edit {start}")

    rng = np.random.default_rng(args.seed + 12345)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    status = "completed"

    if args.mode == "single":
        for i in range(start + 1, n_max + 1):
            e = stream[i - 1]
            err = None
            with Timer() as te:
                try:
                    orig = editor.apply(e, keep_original=True) if editor else {}
                except Exception as ex:  # the method failed on this edit: logged, evaluated as-is
                    err, orig = f"{type(ex).__name__}: {ex}"[:300], {}
                    torch.cuda.empty_cache()
            finite = editor.weights_finite() if editor else True
            with Timer() as tv:
                rec = ev.immediate(e)
                loc = ev.locality()
            if editor:
                editor.restore(orig)
            rec.update({"i": i, "case_id": e["case_id"], "edit_s": round(te.s, 3), "eval_s": round(tv.s, 3),
                        "finite": int(finite), "loc": loc, "edit_error": err})
            append_jsonl(edits_path, rec)
            write_json(progress_path, {"done": i})
            print(f"[single {args.model} {args.method}] {i}/{n_max} eff={rec['eff_em']} "
                  f"loc={np.mean(loc):.3f} edit={te.s:.1f}s", flush=True)
    else:
        consecutive_errors = 0
        n_port_examples = sum(1 for r in read_jsonl(fail_path)) if (args.resume and os.path.exists(fail_path)) else 0
        for i in range(start + 1, n_max + 1):
            e = stream[i - 1]
            err = None
            with Timer() as te:
                if editor:
                    try:
                        editor.apply(e)
                    except Exception as ex:  # the method failed on this edit: logged, evaluated as-is
                        err = f"{type(ex).__name__}: {ex}"[:300]
                        torch.cuda.empty_cache()
            consecutive_errors = consecutive_errors + 1 if err else 0
            if consecutive_errors >= 5:
                status = f"aborted_5_consecutive_edit_errors_at_{i}"
                print(f"!! 5 consecutive edit errors (last: {err}); stopping run", flush=True)
                break
            if editor and not editor.weights_finite():
                status = f"diverged_at_{i}"
                append_jsonl(edits_path, {"i": i, "case_id": e["case_id"], "edit_s": round(te.s, 3), "diverged": 1})
                print(f"!! non-finite weights after edit {i}; stopping run", flush=True)
                break
            with Timer() as tv:
                rec = ev.immediate(e)
            rec.update({"i": i, "case_id": e["case_id"], "edit_s": round(te.s, 3), "eval_s": round(tv.s, 3),
                        "edit_error": err})
            append_jsonl(edits_path, rec)
            if err:
                print(f"edit {i} failed: {err}", flush=True)

            # a few portability success/failure examples with free-text generations (every 5th edit)
            if editor and i % 5 == 0 and n_port_examples < 2 * cfg["failure_samples"]:
                for pr in rec["port"][:1]:
                    t = e["portability"][pr["ti"]]
                    append_jsonl(fail_path, {
                        "kind": "portability", "i": i, "case_id": e["case_id"], "outcome": "pass" if pr["pass"] else "fail",
                        "type": pr["type"], "subtype": pr["subtype"], "prompt": t["queries"][0]["prompt"],
                        "gold": t["queries"][0]["answers"], "edit_prompt": e["prompt"], "target_new": e["target_new"],
                        "target_true": e["target_true"], "generation": greedy_generate(model, tok, t["queries"][0]["prompt"])})
                    n_port_examples += 1

            if i in milestones:
                with Timer() as tm:
                    er, er_acc = ev.retention(stream[:i])
                    loc = ev.locality()
                    ppl = ev.ppl()
                ms = {"n": i, "er": er, "er_tok_acc_mean": float(np.mean(er_acc)), "loc": loc, "ppl": ppl,
                      "ER": float(np.mean(er)), "LOC": float(np.mean(loc)), "eval_s": round(tm.s, 1)}
                append_jsonl(ms_path, ms)
                print(f"[{args.model} {args.method} s{args.seed}] N={i} ER={ms['ER']:.3f} LOC={ms['LOC']:.3f} "
                      f"PPL={ppl:.2f} last_edit={te.s:.1f}s", flush=True)
                if i == n_max and editor:
                    k = cfg["failure_samples"]
                    for kind, vec, items in (("retention", er, stream[:i]), ("locality", loc, probes)):
                        fails = [j for j, v in enumerate(vec) if not v]
                        for j in sorted(rng.choice(fails, size=min(k, len(fails)), replace=False).tolist()) if fails else []:
                            it = items[j]
                            append_jsonl(fail_path, {
                                "kind": kind, "i": i, "outcome": "fail", "index": j, "case_id": it.get("case_id", it.get("probe_id")),
                                "prompt": it["prompt"], "gold": [it["target_new"]] if kind == "retention" else it["answers"],
                                "target_true": it.get("target_true"), "generation": greedy_generate(model, tok, it["prompt"])})
                if editor:
                    torch.save({"done": i, "state": editor.state(), "templates": Editor.template_caches()}, ckpt_path + ".tmp")
                    os.replace(ckpt_path + ".tmp", ckpt_path)  # atomic: a kill never leaves a half-written checkpoint
                write_json(progress_path, {"done": i})

    summary = {"status": status, "n_done": __import__("json").load(open(progress_path))["done"] if os.path.exists(progress_path) else 0}
    if torch.cuda.is_available():
        summary["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    rows = read_jsonl(edits_path) if os.path.exists(edits_path) else []
    if rows:
        summary["mean_edit_s"] = round(float(np.mean([r["edit_s"] for r in rows])), 3)
    write_json(os.path.join(run_dir, "run_summary.json"), summary)
    print("DONE", run_dir, summary)


if __name__ == "__main__":
    main()
