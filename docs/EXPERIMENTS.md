# Running the experiments

Run every command from the **repository root** with the `reid` environment
active:

```bash
python scripts/<name>.py [options]
```

Each training script writes checkpoints to `checkpoints/` and plots to
`results/`; each evaluation script prints metrics and writes plots to
`results/`. Every script documents its own options in its header comment — read
the top of the file (or pass `--help`) for the available flags.

> Make sure the datasets are in place first — see [`DATASETS.md`](DATASETS.md).

---

## 1. Merged Terrestrial protocol (ATRW + LeopardID)

Requires both datasets.

```bash
# train — leak-free CLIP + MFA, multi-scale image branch,
#         last UNFREEZE_LAST_N_BLOCKS CLIP blocks unfrozen
python scripts/train_terrestrial.py                  # add --batch-size 32 if VRAM is tight

# evaluate — closed-set mAP / Rank-1 / Rank-5, overall and per species
python scripts/eval_terrestrial.py --checkpoint terrestrial_mfa_ft4_epoch60.pth

# qualitative figures: sample crops, retrieval grid, t-SNE
python scripts/visualize_terrestrial.py --checkpoint terrestrial_mfa_ft4_epoch60.pth
```

The checkpoint name follows the pattern `terrestrial_mfa_ft<N>_epoch<E>.pth`,
where `<N>` is `UNFREEZE_LAST_N_BLOCKS` from `config.py`.

---

## 2. ATRW — identity-prompt evaluation-leak study

```bash
# leak-free training (the individual identifier is removed from the prompt)
python scripts/train_atrw_neutral.py

# four-variant evaluation:
#   visual-only / leak prompt / constant id / neutral prompt
python scripts/eval_atrw_ablation.py --checkpoint <checkpoint>.pth
```

Run the ablation on both a leak-trained and a leak-free checkpoint to measure
the size of the leak and the leak-free accuracy on Protocol 1 (full 107-identity
split) and Protocol 2 (47-identity closed set).

---

## 3. LeopardID 2022 — identity-prompt evaluation-leak study

```bash
python scripts/train_leopard.py            # baseline (identifier in the prompt)
python scripts/train_leopard_neutral.py    # leak-free training

python scripts/eval_leopard_ablation.py --checkpoint <checkpoint>.pth   # four-variant ablation
python scripts/eval_leopard.py --checkpoint <checkpoint>.pth            # standard evaluation
```

---

## 4. Auto-derived metadata

First estimate the metadata, then train and evaluate the metadata-conditioned
models.

```bash
# (a) body-orientation labels — choose one estimator
python scripts/extract_orientation_yolo.py
python scripts/extract_orientation_transfer.py
python scripts/extract_orientation_pose.py --dataset atrw     # or --dataset leopard

# (b) circadian + temperature + orientation  ->  data/<dataset>/metadata_auto.json
python scripts/extract_metadata.py --dataset atrw
python scripts/extract_metadata.py --dataset leopard

# (c) train with auto-metadata prompts
python scripts/train_atrw_meta.py
python scripts/train_leopard_meta.py

# (d) evaluate the metadata-conditioned checkpoints
python scripts/eval_atrw_meta.py --checkpoint <checkpoint>.pth
python scripts/eval_leopard_meta.py --checkpoint <checkpoint>.pth
```

---

## Configuration

All hyper-parameters, dataset paths, and split ratios live in `config.py`:

- **Training:** `NUM_EPOCHS`, `BATCH_SIZE`, `LR`, `WEIGHT_DECAY`, loss weights
  (`LAMBDA_ID`, `LAMBDA_TRI`, `LAMBDA_IA`).
- **Backbone fine-tuning:** `UNFREEZE_LAST_N_BLOCKS`, `BACKBONE_LR_MULT`.
- **Split protocols:** the `LEOPARD_*` parameters (single-dataset leopard split)
  and the `TM_*` parameters (merged Terrestrial split).
- **Model:** `CLIP_MODEL`, `EMBED_DIM`, `NUM_HEADS_ATTN`.

Checkpoints land in `checkpoints/` and all plots / metrics in `results/`; both
directories are created automatically and are git-ignored.
