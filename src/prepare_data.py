"""Build the frozen dataset splits used in Assignment 4.

Source: RippleEdits benchmark (Cohen et al., 2024), GitHub edenbiran/RippleEdits,
commit 54f3b88af4895a3aacb580ec63ce7ae857185040, files data/benchmark/popular.json
and data/benchmark/random.json (both are counterfactual edits with a known original fact).

Output (data/processed/):
  edits_test.jsonl      edit pool for the final experiment (sequential streams are sampled from it)
  edits_dev.jsonl       development pool (pilot / timing / debugging only, never reported as results)
  locality_pool.jsonl   candidate "untouched" facts for the global specificity probe set
  split_summary.json    counts
  SHA256SUMS            hashes of the three jsonl files (recorded in the protocol freeze form)

Usage: python src/prepare_data.py --raw data/raw --out data/processed
"""
import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict

SPLIT_SEED = 2026
SOURCES = ["popular", "random"]
PORTABILITY_KEYS = {
    "Compositionality_I": "two_hop",
    "Compositionality_II": "two_hop",
    "Logical_Generalization": "reverse",
    "Subject_Aliasing": "alias",
}
MAX_ANSWERS = 4          # canonical value + up to 3 aliases
MAX_ANSWER_CHARS = 60
MAX_SUBJECT_CHARS = 60


def load_templates(relation_py):
    src = open(relation_py, encoding="utf-8").read()
    pat = re.compile(r"^\s+([A-Z_]+) = \('(?:P?\d*)', (['\"])(.+?)\2", re.M)
    return {m.group(1): m.group(3) for m in pat.finditer(src)}


def latin_ok(s):
    # GPT-2 / Llama prompts in this study are English; drop strings with non-Latin scripts.
    return all(ord(c) < 0x250 for c in s)


def parse_fact(prompt, template):
    """prompt = template.replace('<subject>', S) + ' ' + T + '.'  ->  (S, T, phrase)"""
    if "<subject>" not in template or not prompt.endswith("."):
        return None
    pre, post = template.split("<subject>")
    body = prompt[:-1]
    if not body.startswith(pre):
        return None
    rest = body[len(pre):]
    # find the split point: rest = S + post + ' ' + T ; choose the first occurrence that rebuilds exactly
    key = post + " "
    start = 0
    while True:
        k = rest.find(key, start)
        if k < 0:
            return None
        subj, tgt = rest[:k], rest[k + len(key):]
        if subj and tgt and (pre + subj + post + " " + tgt + ".") == prompt:
            return subj.strip(), tgt.strip(), (pre + subj + post)
        start = k + 1


def answers_of(q):
    out = []
    for a in q.get("answers", []):
        cands = [a.get("value", "")] + list(a.get("aliases", []))
        for c in cands:
            c = (c or "").strip()
            if c and len(c) <= MAX_ANSWER_CHARS and latin_ok(c) and c not in out:
                out.append(c)
    return out[:MAX_ANSWERS]


def convert_query(q):
    ans = answers_of(q)
    if not ans or not q.get("prompt") or not latin_ok(q["prompt"]):
        return None
    return {"prompt": q["prompt"].strip(), "answers": ans, "relation": q.get("relation"),
            "q_subject_id": q.get("subject_id") if isinstance(q.get("subject_id"), str) else (q.get("subject_id") or [None])[0]}


def convert_tests(tests, kind, min_one=True):
    out = []
    for t in tests or []:
        tq = [convert_query(q) for q in t.get("test_queries", [])]
        tq = [q for q in tq if q]
        if not tq:
            continue
        test_prompts = {q["prompt"] for q in tq}
        conds = [convert_query(q) for q in t.get("condition_queries", [])]
        conds = [c for c in conds if c]
        if kind != "rel_spec":
            # A condition identical to the test query would require the *unedited* model to
            # already know the post-edit answer; such conditions are dropped (see protocol).
            conds = [c for c in conds if c["prompt"] not in test_prompts]
        out.append({"queries": tq, "conditions": conds, "condition_mode": t.get("test_condition", "OR")})
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    templates = load_templates(os.path.join(args.raw, "ripple_relation.py"))

    drop = Counter()
    edits, seen = [], set()
    for src in SOURCES:
        data = json.load(open(os.path.join(args.raw, f"{src}.json"), encoding="utf-8"))
        for idx, ex in enumerate(data):
            e = ex["edit"]
            rel = e["relation"]
            if "original_fact" not in e:
                drop["no_original_fact"] += 1; continue
            tpl = templates.get(rel)
            if tpl is None:
                drop["no_template"] += 1; continue
            p_new = parse_fact(e["prompt"], tpl)
            p_old = parse_fact(e["original_fact"]["prompt"], tpl)
            if not p_new or not p_old:
                drop["parse_failed"] += 1; continue
            subj, tgt_new, phrase = p_new
            subj_o, tgt_true, phrase_o = p_old
            if subj != subj_o or phrase != phrase_o:
                drop["subject_mismatch"] += 1; continue
            if not (latin_ok(phrase) and latin_ok(tgt_new) and latin_ok(tgt_true)):
                drop["non_latin"] += 1; continue
            if len(subj) > MAX_SUBJECT_CHARS or "{" in phrase or "}" in phrase or tgt_new == tgt_true:
                drop["other_filter"] += 1; continue
            if len(tgt_new) > MAX_ANSWER_CHARS:
                drop["long_target"] += 1; continue
            key = (e["subject_id"], rel)
            if key in seen:
                drop["duplicate_subject_relation"] += 1; continue
            seen.add(key)

            port = []
            for k, ptype in PORTABILITY_KEYS.items():
                for t in convert_tests(ex.get(k), ptype):
                    t["type"] = ptype
                    if ptype == "reverse":
                        # Logical_Generalization mixes true inverse/symmetric relations (query asked
                        # about the NEW TARGET entity) with other logical implications.
                        is_inv = any(q["q_subject_id"] == e["target_id"] for q in t["queries"])
                        t["subtype"] = "inverse_relation" if is_inv else "other_logical"
                    t["source_key"] = k
                    port.append(t)
            rel_spec = convert_tests(ex.get("Relation_Specificity"), "rel_spec")
            edits.append({
                "case_id": f"{src}-{idx}",
                "source": src,
                "subject_id": e["subject_id"],
                "relation": rel,
                "subject": subj,
                "prompt": phrase,                       # the edit prompt (subject included)
                "prompt_template": phrase.replace(subj, "{}", 1),
                "target_new": tgt_new,
                "target_true": tgt_true,
                "target_id": e["target_id"],
                "portability": port,
                "rel_spec": rel_spec,
            })

    # ---- split by SUBJECT (no subject appears in two splits) ----
    subjects = sorted({e["subject_id"] for e in edits})
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(subjects)
    n = len(subjects)
    n_dev, n_loc = int(0.05 * n), int(0.25 * n)
    dev_s = set(subjects[:n_dev]); loc_s = set(subjects[n_dev:n_dev + n_loc])
    dev = [e for e in edits if e["subject_id"] in dev_s]
    loc_src = [e for e in edits if e["subject_id"] in loc_s]
    test = [e for e in edits if e["subject_id"] not in dev_s and e["subject_id"] not in loc_s]

    # ---- locality candidates: TRUE facts about held-out subjects (never edited) ----
    edit_subject_names = {e["subject"] for e in dev + test}
    edit_prompts = {e["prompt"] for e in dev + test}
    loc, loc_seen = [], set()
    for e in loc_src:
        cands = [{"prompt": e["prompt"], "answers": [e["target_true"]], "relation": e["relation"], "kind": "original_fact"}]
        for t in e["rel_spec"]:
            for q in t["queries"]:
                cands.append({"prompt": q["prompt"], "answers": q["answers"], "relation": q["relation"], "kind": "rel_spec_heldout"})
        for c in cands:
            if c["prompt"] in loc_seen or c["prompt"] in edit_prompts:
                continue
            if any(s in c["prompt"] for s in edit_subject_names if len(s) > 3):
                continue  # must not mention any edited subject
            loc_seen.add(c["prompt"])
            c["probe_id"] = f"loc-{len(loc)}"
            c["subject_id"] = e["subject_id"]
            loc.append(c)
    rng.shuffle(loc)  # fixed order; the first K known-by-model probes are used per model

    def dump(rows, name):
        path = os.path.join(args.out, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    paths = [dump(test, "edits_test.jsonl"), dump(dev, "edits_dev.jsonl"), dump(loc, "locality_pool.jsonl")]
    with open(os.path.join(args.out, "SHA256SUMS"), "w") as f:
        for p in paths:
            f.write(f"{sha256(p)}  {os.path.basename(p)}\n")

    def port_counts(rows):
        c = Counter()
        for r in rows:
            for t in r["portability"]:
                c[t["type"]] += 1
                if t.get("subtype"):
                    c[t["subtype"]] += 1
            c["rel_spec"] += len(r["rel_spec"])
            c["edits_with_any_portability"] += int(bool(r["portability"]))
        return dict(c)

    summary = {
        "split_seed": SPLIT_SEED,
        "kept_edits": len(edits), "dropped": dict(drop),
        "test_edits": len(test), "dev_edits": len(dev), "locality_candidates": len(loc),
        "test_by_source": dict(Counter(e["source"] for e in test)),
        "test_tests": port_counts(test),
        "test_relations": dict(Counter(e["relation"] for e in test).most_common()),
        "sha256": {os.path.basename(p): sha256(p) for p in paths},
    }
    json.dump(summary, open(os.path.join(args.out, "split_summary.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
