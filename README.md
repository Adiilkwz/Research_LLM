# Sequential Knowledge Editing: Portability, Specificity and Stability (RMT Assignment 4)

Adil Ramazan, Yesbol Bagyt, Nurassyl Aitbay — group SE-2415

**Research question.** To what extent do existing knowledge-editing methods maintain portability, specificity
and long-term stability during sequential knowledge updates in Large Language Models?

This repository contains everything needed to reproduce the main results table: frozen data splits, the exact
editing code (vendored EasyEdit subset), configuration, run scripts, raw logs and the analysis script.

## What is compared

| Condition | Role |
|---|---|
| `none` — unedited model | simple baseline (no intervention) |
| `FT` — FT-M, fine-tuning of one MLP layer with the loss on every target token (Zhang et al., 2024; L∞-constrained for GPT2-XL as in EasyEdit's defaults) | common literature baseline |
| `ROME` — rank-one model editing (Meng et al., 2022) | editing method under study |
| `MEMIT` — mass-editing memory in a transformer (Meng et al., 2023) | editing method under study |

Models: **GPT2-XL** (1.5B, fp32) and **Llama-3.2-3B** (fp16, `unsloth/Llama-3.2-3B`, identical weights to
`meta-llama/Llama-3.2-3B`). Every method edits the **same** stream of 1,000 counterfactual facts (per seed, 3 seeds)
one after another. Hyper-parameters are EasyEdit's published defaults (`configs/hparams/`), none were tuned, with two
changes fixed on the dev split before any test run: ROME uses the covariance adjustment of the original paper
(`mom2_adjustment: true`), and the fine-tuning baseline uses the FT-M objective (`target_new`).

## Metrics

| Metric | Definition | Measured |
|---|---|---|
| **ERS (primary)** | harmonic mean of ER and LOC at N = n_max | milestones |
| Efficacy | edited fact produced right after its edit | after every edit |
| Portability | RippleEdits ripple tests: two-hop (Compositionality I/II), inverse relation & other logical (Logical Generalization), subject aliasing | after every edit |
| Relation specificity | other facts about the edited subject (RippleEdits Relation_Specificity): share of answer tokens whose prediction is unchanged vs. the unedited model; secondary: known facts still correct | after every edit |
| ER — edit retention | share of all edits so far still produced | milestones 1, 10, 50, 100, 250, 500, 750, 1000 |
| LOC — locality | mean over 500 untouched held-out facts of the share of answer tokens whose argmax prediction is unchanged vs. the unedited model (EasyEdit locality; Yao et al., 2023); 1 − LOC = side-effect distortion rate; secondary LOC_known: facts known before editing still correct | milestones |
| PPL | perplexity on 100 wikitext-103 test paragraphs | milestones |

Predictions are compared on the gold-answer positions under teacher forcing (argmax token ids). A query counts as correct when, under teacher forcing, the model's argmax token equals the gold token at every
position of the gold answer or of one of its aliases (exact match; EasyEdit's `decoder_only_teacher_forcing`).

## Repository structure

```
configs/experiment.yaml        frozen protocol settings (models, seeds, n_max, milestones, ...)
configs/hparams/<model>/       EasyEdit default hyper-parameters (unchanged copies)
data/raw/                      RippleEdits popular.json / random.json (commit 54f3b88, MIT licence)
data/processed/                frozen splits + SHA256SUMS (built by src/prepare_data.py)
src/prepare_data.py            raw RippleEdits -> frozen splits
src/prefilter.py               pre-edit screening with the unedited model (RippleEdits condition queries)
src/compute_cov.py             MEMIT covariance statistics (wikitext-103)
src/run_experiment.py          one (model, method, seed) run, sequential or single-edit mode
src/run_queue.py               runs several jobs on one GPU, resumable, stops before Kaggle's 12 h limit
src/error_export.py            failure examples + automatic error categories -> CSV for manual labelling
src/analyze.py                 statistics, tables, figures (CPU only)
third_party/easyeditor/        ROME, MEMIT, FT from EasyEdit commit 4c10987 (MIT); one patch, see below
notebooks/kaggle_runner.ipynb  Kaggle notebook (setup / pilot / main / analysis)
results/                       raw logs of every run (edits.jsonl, milestones.jsonl, ...) and results/analysis/
```

## Reproducing the results

Hardware used: Kaggle Notebook, 2 × NVIDIA T4 (16 GB), free tier. Software: Kaggle's pre-installed PyTorch
(version recorded in every `results/**/run_config.json`) + `requirements.txt`.

```bash
git clone <this repo> ke-seq && cd ke-seq
pip install -r requirements.txt                     # does not touch torch
python src/prepare_data.py                          # optional: rebuilds data/processed, check SHA256SUMS
for M in gpt2-xl llama3.2-3b; do
  python src/prefilter.py   --model $M
  python src/compute_cov.py --model $M
  for S in 0 1 2; do for A in none FT ROME MEMIT; do
    python src/run_experiment.py --model $M --method $A --seed $S --resume
  done; done
  for A in FT ROME MEMIT; do python src/run_experiment.py --model $M --method $A --mode single --resume; done
  python src/error_export.py --model $M
done
python src/analyze.py                               # -> results/analysis/main_results.md, figures, tests
```

Only the analysis step is needed to regenerate the tables and figures from the committed raw logs:
`python src/analyze.py` (CPU, < 1 min).

On Kaggle use `notebooks/kaggle_runner.ipynb` (instructions inside; modes pilot / main / analysis, setup runs automatically). One model per account; the two GPUs of a
session run two job queues in parallel.

## Changes to third-party code

`third_party/easyeditor/models/ft/ft_main.py`, *patch 1*: target strings are tokenised with
`add_special_tokens=False`, as ROME and MEMIT already do. Without it, FT on Llama trains towards the BOS token.
`third_party/easyeditor/models/rome/compute_u.py`, *patch 2*: the inverse-covariance product is computed in fp32
(entries of C⁻¹ can exceed the fp16 range on the fp16 Llama model); no effect on the fp32 GPT2-XL model.
For FT on the fp16 model, the trained layer is kept in fp32 (`src/editors.py::_fp32_island`), because Adam updates
underflow in fp16.

## Data and licences

RippleEdits (Cohen et al., 2024) — MIT, `data/raw/RIPPLEEDITS_LICENSE.txt`. EasyEdit (Zhang et al., 2024) — MIT,
`third_party/EASYEDIT_LICENSE`. Model weights are downloaded from the Hugging Face Hub at run time and are not part
of this repository. No personal data is used.

## Error analysis (Appendix B of the report)

`results/error_analysis/labelled_failures.csv` holds a stratified sample of 84 failures (42 per model; per method
6 portability, 4 retention, 4 locality failures; seed 2026). Codebook: **degenerate_output** (repetition, gibberish,
empty), **interference_from_other_edits** (objects of other edits in the same stream; checked against the run's edit
list), **unedited_behaviour** (same answer as the unedited model), **no_second_hop** (repeats the new object instead
of applying the second relation), **partial_propagation** (related to the new fact but wrong), **partial_target**
(first tokens of the target, then drift), **wrong_entity** (fluent, specific, unrelated wrong answer). Labels were
drafted with AI assistance and verified by the three authors.

## Report figures

`python -c "import runpy; runpy.run_path('src/analyze.py', run_name='lib')['report_figures']()"` renders the
print-size figures `results/analysis/report_fig*.png` used in the report.
