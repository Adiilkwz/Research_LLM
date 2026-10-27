"""Shared utilities: config, model loading, data loading, teacher-forced scoring, generation."""
import json
import os
import random
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party"))  # vendored EasyEdit subset -> `import easyeditor`

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def load_config(path=None):
    path = path or os.path.join(ROOT, "configs", "experiment.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:  # half-written last line of a killed run
                    print(f"warning: skipping malformed line in {path}")
    return rows


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def results_dir(cfg):
    return os.path.join(ROOT, cfg.get("results_dir", "results"))


def load_model(cfg, model_key, device="cuda"):
    mcfg = cfg["models"][model_key]
    dtype = getattr(torch, mcfg["dtype"])
    tok = AutoTokenizer.from_pretrained(mcfg["hf_id"])
    tok.pad_token_id = tok.eos_token_id  # as in EasyEdit's BaseEditor for GPT-2 / Llama
    try:
        model = AutoModelForCausalLM.from_pretrained(mcfg["hf_id"], dtype=dtype)   # transformers >= 4.56
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(mcfg["hf_id"], torch_dtype=dtype)
    assert next(model.parameters()).dtype == dtype, "model loaded in the wrong dtype"
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def model_device(model):
    return next(model.parameters()).device


# --------------------------------------------------------------------------------------
# Teacher-forced scoring
# --------------------------------------------------------------------------------------
@torch.no_grad()
def score_pairs(model, tok, pairs, batch_size=32, max_answer_tokens=16):
    """Score (prompt, answer) pairs with teacher forcing.

    Returns a list of dicts {em, tok_acc, logp}:
      em      1 if the greedy (argmax) token equals the answer token at EVERY answer position
      tok_acc fraction of answer tokens predicted correctly (EasyEdit's token_accuracy protocol)
      logp    mean log-probability of the answer tokens
    """
    if not pairs:
        return []
    dev = model_device(model)
    enc = []
    for p, a in pairs:
        pi = tok(p, add_special_tokens=True)["input_ids"]
        ai = tok(" " + a.strip(), add_special_tokens=False)["input_ids"][:max_answer_tokens]
        enc.append((pi, ai))
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]) + len(enc[i][1]))
    out = [None] * len(enc)
    pad = tok.pad_token_id
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        L = max(len(enc[i][0]) + len(enc[i][1]) for i in idx)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        mask = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            seq = enc[i][0] + enc[i][1]
            ids[r, :len(seq)] = torch.tensor(seq)
            mask[r, :len(seq)] = 1
        logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
        for r, i in enumerate(idx):
            pi, ai = enc[i]
            pos = torch.arange(len(pi) - 1, len(pi) + len(ai) - 1, device=logits.device)
            lg = logits[r, pos].float()
            tgt = torch.tensor(ai, device=logits.device)
            pred = lg.argmax(-1)
            correct = (pred == tgt)
            lp = torch.log_softmax(lg, -1).gather(-1, tgt[:, None]).squeeze(-1)
            out[i] = {"em": int(correct.all().item()), "tok_acc": float(correct.float().mean().item()),
                      "logp": float(lp.mean().item())}
        del logits
    return out


def score_queries(model, tok, queries, batch_size=32, max_answer_tokens=16):
    """queries: list of {"prompt", "answers": [...]}. A query is correct if ANY answer string is an EM."""
    pairs, owner = [], []
    for qi, q in enumerate(queries):
        for a in q["answers"]:
            pairs.append((q["prompt"], a)); owner.append(qi)
    res = score_pairs(model, tok, pairs, batch_size, max_answer_tokens)
    best = [{"em": 0, "tok_acc": 0.0, "logp": -1e9} for _ in queries]
    for qi, r in zip(owner, res):
        b = best[qi]
        b["em"] = max(b["em"], r["em"])
        b["tok_acc"] = max(b["tok_acc"], r["tok_acc"])
        b["logp"] = max(b["logp"], r["logp"])
    return best


def test_passes(query_results, mode="OR"):
    ems = [r["em"] for r in query_results]
    return int(any(ems)) if mode == "OR" else int(all(ems))


@torch.no_grad()
def greedy_generate(model, tok, prompt, max_new_tokens=12):
    dev = model_device(model)
    enc = tok(prompt, return_tensors="pt").to(dev)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def perplexity(model, tok, passages, max_tokens=128, batch_size=8):
    dev = model_device(model)
    tot_nll, tot_tok = 0.0, 0
    encs = [tok(p, add_special_tokens=True)["input_ids"][:max_tokens] for p in passages]
    for b in range(0, len(encs), batch_size):
        chunk = encs[b:b + batch_size]
        L = max(len(e) for e in chunk)
        ids = torch.full((len(chunk), L), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for r, e in enumerate(chunk):
            ids[r, :len(e)] = torch.tensor(e); mask[r, :len(e)] = 1
        ids, mask = ids.to(dev), mask.to(dev)
        logits = model(input_ids=ids, attention_mask=mask).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None]).squeeze(-1)
        m = mask[:, 1:].float()
        tot_nll += float(-(lp * m).sum().item()); tot_tok += int(m.sum().item())
    return float(np.exp(tot_nll / max(tot_tok, 1)))


def ppl_passages(cfg):
    """Fixed perplexity passages: first N wikitext-103 *test* paragraphs with >= 200 characters."""
    path = os.path.join(ROOT, cfg["data_dir"], "ppl_passages.jsonl")
    if not os.path.exists(path):
        from datasets import load_dataset
        ds = load_dataset(cfg["cov"]["hf_dataset"], cfg["cov"]["hf_config"], split="test")
        rows = [t.strip() for t in ds["text"] if len(t.strip()) >= 200][: int(cfg["ppl_passages"])]
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"text": r}, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
    return [r["text"] for r in read_jsonl(path)]


class Timer:
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t = time.perf_counter(); return self

    def __exit__(self, *a):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.s = time.perf_counter() - self.t
