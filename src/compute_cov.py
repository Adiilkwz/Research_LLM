"""Pre-compute the key covariance statistics used by MEMIT and ROME (second moment of the inputs to the edited MLP
projection) on wikitext-103 and store them exactly where EasyEdit's layer_stats() looks for them.

Why: EasyEdit would otherwise download the full English Wikipedia dump (~20 GB) and run
100,000 forward passes per layer, which is infeasible on a free Kaggle T4. We use a fixed,
seeded sample of wikitext-103 paragraphs instead (declared as protocol deviation D4).

Usage: python src/compute_cov.py --model gpt2-xl
"""
import argparse
import os
import random

import torch
from datasets import load_dataset

from common import load_config, load_model, set_seed, write_json, ROOT
from editors import load_hparams, stats_file
from easyeditor.util import nethook
from easyeditor.util.runningstats import CombinedStat, SecondMoment, save_cached_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ccfg = cfg["cov"]
    set_seed(ccfg["seed"])

    model, tok = load_model(cfg, args.model)
    tok.padding_side = "right"
    hp = load_hparams(cfg, args.model, "MEMIT")
    hp_rome = load_hparams(cfg, args.model, "ROME")
    layer_names = sorted({hp.rewrite_module_tmp.format(l) for l in hp.layers} |
                         {hp_rome.rewrite_module_tmp.format(l) for l in hp_rome.layers})
    targets = {ln: stats_file(cfg, model, ln) for ln in layer_names}
    if all(os.path.exists(p) for p in targets.values()):
        print("All covariance files already exist:", *targets.values(), sep="\n  ")
        return

    ds = load_dataset(ccfg["hf_dataset"], ccfg["hf_config"], split="train")
    idx = [i for i, t in enumerate(ds["text"]) if len(t.strip()) >= ccfg["min_chars"]]
    rng = random.Random(ccfg["seed"])
    idx = rng.sample(idx, int(ccfg["n_samples"]))
    texts = [ds[i]["text"].strip() for i in idx]
    print(f"Using {len(texts)} wikitext paragraphs for {layer_names}")

    stats = {ln: CombinedStat(mom2=SecondMoment()) for ln in layer_names}
    dev = next(model.parameters()).device
    n_tokens = 0
    with torch.no_grad():
        for b in range(0, len(texts), args.batch_size):
            enc = tok(texts[b:b + args.batch_size], return_tensors="pt", padding=True, truncation=True,
                      max_length=int(ccfg["max_tokens"])).to(dev)
            with nethook.TraceDict(model, layer_names, retain_input=True, retain_output=False, stop=True) as tr:
                model(**enc)
            m = enc["attention_mask"].bool()
            for ln in layer_names:
                feats = tr[ln].input[m].to(torch.float32)   # (n_tokens, d_ff)
                stats[ln].add(feats)
            n_tokens += int(m.sum().item())
            if (b // args.batch_size) % 100 == 0:
                print(f"  {b + len(enc['input_ids'])}/{len(texts)} paragraphs, {n_tokens} tokens", flush=True)

    for ln, st in stats.items():
        st.to_(device="cpu")
        save_cached_state(targets[ln], st, {"sample_size": int(ccfg["n_samples"])})
        print("saved", targets[ln])
    write_json(os.path.join(os.path.dirname(targets[layer_names[0]]), "cov_meta.json"),
               {"model": args.model, "layers": layer_names, "paragraphs": len(texts), "tokens": n_tokens,
                "dataset": f"{ccfg['hf_dataset']}/{ccfg['hf_config']}", "seed": ccfg["seed"],
                "max_tokens": ccfg["max_tokens"]})


if __name__ == "__main__":
    main()
