"""Thin wrappers around the vendored EasyEdit implementations of FT-L, ROME and MEMIT.

Algorithm hyper-parameters are EasyEdit's published defaults (configs/hparams/<model>/<METHOD>.yaml,
copied unchanged from EasyEdit commit 4c10987). The ONLY overrides applied here are:
  * model_name / device           -> where the model lives (no effect on the algorithm)
  * stats_dir, mom2_dataset, mom2_n_samples (ROME, MEMIT) -> covariance statistics are pre-computed
    on wikitext-103 by src/compute_cov.py (see protocol deviation D4). ROME's defaults use
    mom2_adjustment=false, so only MEMIT actually reads these statistics.
"""
import contextlib
import os

import torch

from common import ROOT

from easyeditor.util import nethook
from easyeditor.models.rome import ROMEHyperParams, apply_rome_to_model
from easyeditor.models.memit import MEMITHyperParams, apply_memit_to_model
from easyeditor.models.ft import FTHyperParams, apply_ft_to_model
import easyeditor.models.rome.rome_main as rome_main
import easyeditor.models.memit.memit_main as memit_main

HP_CLASSES = {"ROME": ROMEHyperParams, "MEMIT": MEMITHyperParams, "FT": FTHyperParams}
APPLY = {"ROME": apply_rome_to_model, "MEMIT": apply_memit_to_model, "FT": apply_ft_to_model}


def load_hparams(cfg, model_key, method):
    path = os.path.join(ROOT, cfg["models"][model_key]["hparams_dir"], f"{method}.yaml")
    hp = HP_CLASSES[method].from_hparams(path)
    hp.model_name = cfg["models"][model_key]["hf_id"]
    hp.device = 0
    if method in ("ROME", "MEMIT"):
        hp.stats_dir = os.path.join(ROOT, cfg["cov"]["stats_dir"])
        hp.mom2_dataset = cfg["cov"]["dataset_name"]
        hp.mom2_n_samples = int(cfg["cov"]["n_samples"])
    return hp


def stats_file(cfg, model, layer_name):
    """Exact path that easyeditor.models.rome.layer_stats looks up for the cached covariance."""
    model_name = model.config._name_or_path.rsplit("/")[-1]
    return os.path.join(ROOT, cfg["cov"]["stats_dir"], model_name, f"{cfg['cov']['dataset_name']}_stats",
                        f"{layer_name}_float32_mom2_{int(cfg['cov']['n_samples'])}.npz")


class Editor:
    """Applies one edit at a time to `model` in place and remembers which weights were touched."""

    def __init__(self, method, model, tok, hparams, log_path):
        assert method in APPLY
        self.method, self.model, self.tok, self.hp = method, model, tok, hparams
        # EasyEdit BaseEditor convention: right padding for ROME/MEMIT, left padding otherwise.
        tok.padding_side = "left" if method == "FT" else "right"
        self.log_path = log_path
        self.param_names = self._param_names()
        model_dtype = next(model.parameters()).dtype
        self._hooks = []
        if method == "FT" and model_dtype != torch.float32:
            self._fp32_island(model_dtype)

    # ------------------------------------------------------------------
    def _param_names(self):
        if self.method in ("ROME", "MEMIT"):
            return [f"{self.hp.rewrite_module_tmp.format(l)}.weight" for l in self.hp.layers]
        names = []
        for n, _ in self.model.named_parameters():
            if any(self.hp.rewrite_module_tmp.format(l) in n for l in self.hp.layers):
                names.append(n)
        return names

    def _fp32_island(self, model_dtype):
        """FT trains the target layer with Adam; Adam on fp16 weights under/overflows. The trained
        module is therefore kept in fp32 (inputs cast up, outputs cast back to the model dtype).
        This changes numerics only by fp16 rounding and is applied to FT on fp16 models only."""
        mods = set()
        for n in self.param_names:
            mods.add(n.rsplit(".", 1)[0])
        for m in mods:
            mod = nethook.get_module(self.model, m)
            mod.float()
            self._hooks.append(mod.register_forward_pre_hook(
                lambda _m, args: tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)))
            self._hooks.append(mod.register_forward_hook(lambda _m, _a, out: out.to(model_dtype)))

    def request(self, e):
        # FT feeds the prompt text directly; ROME/MEMIT need the subject position ("{}").
        prompt = e["prompt"] if self.method == "FT" else e["prompt_template"]
        return {"prompt": prompt, "subject": e["subject"], "target_new": e["target_new"],
                "ground_truth": e["target_true"], "case_id": e["case_id"]}

    def apply(self, e, keep_original=False):
        with open(self.log_path, "a", encoding="utf-8") as logf, contextlib.redirect_stdout(logf):
            _, orig = APPLY[self.method](self.model, self.tok, [self.request(e)], self.hp,
                                         copy=False, return_orig_weights=keep_original)
        for p in self.model.parameters():
            p.requires_grad_(False)
        return orig

    @torch.no_grad()
    def restore(self, orig):
        for k, v in orig.items():
            nethook.get_parameter(self.model, k)[...] = v.to(nethook.get_parameter(self.model, k).device)

    @torch.no_grad()
    def weights_finite(self):
        return all(torch.isfinite(nethook.get_parameter(self.model, n)).all().item() for n in self.param_names)

    # ------------------------------------------------------------------ checkpointing
    def state(self):
        return {n: nethook.get_parameter(self.model, n).detach().cpu().clone() for n in self.param_names}

    @torch.no_grad()
    def load_state(self, state):
        for n, v in state.items():
            p = nethook.get_parameter(self.model, n)
            p[...] = v.to(device=p.device, dtype=p.dtype)

    @staticmethod
    def template_caches():
        return {"rome": rome_main.CONTEXT_TEMPLATES_CACHE, "memit": memit_main.CONTEXT_TEMPLATES_CACHE}

    @staticmethod
    def set_template_caches(c):
        if c.get("rome") is not None:
            rome_main.CONTEXT_TEMPLATES_CACHE = c["rome"]
        if c.get("memit") is not None:
            memit_main.CONTEXT_TEMPLATES_CACHE = c["memit"]
