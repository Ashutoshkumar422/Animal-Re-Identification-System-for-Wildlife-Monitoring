# Metadata-Aware Animal Re-Identification with CLIP

Re-identification of individual wild animals from camera-trap photographs, built
on a frozen **CLIP ViT-B/16** backbone with a lightweight **Meta-Feature Adapter
(MFA)**. This repository contains the code for a study on two public benchmarks
— **ATRW** (Amur tiger) and **LeopardID 2022** — and on a merged "Terrestrial
Mammal" protocol that combines both species.

## What this study covers

- A re-implementation of the Meta-Feature Adapter on a frozen CLIP backbone: a
  visual feature extractor, a text/metadata encoder, a gated cross-attention
  fusion module, and a BNNeck re-identification head.
- An **evaluation-leak analysis**. When the individual identifier is written
  into the metadata prompt, gallery and query images of the same animal share a
  text anchor and the metadata branch turns into an identity oracle. Removing
  the identifier from the *training* prompt eliminates the leak and improves
  honest accuracy.
- An **auto-derived-metadata pipeline** that estimates per-image circadian
  state, temperature category, and body orientation directly from pixels, for
  datasets that ship no environmental metadata.
- A **merged-dataset protocol** (ATRW ∪ LeopardID 2022 with entity-disjoint
  splits) that enables a like-for-like comparison with prior terrestrial-mammal
  re-identification work.

## Repository layout

```
metawild-impl/
├── config.py                   # all hyper-parameters, paths, split ratios
├── data/                       # dataset builders + split protocols
│   ├── atrw_dataset.py
│   ├── leopard_dataset.py
│   └── terrestrial_dataset.py   # merged ATRW + LeopardID protocol
├── models/
│   ├── mfa.py                   # Meta-Feature Adapter + multi-scale image branch
│   └── losses.py                # identity + triplet + cross-modal contrastive losses
├── eval/
│   ├── metrics.py               # mAP / CMC, feature extraction, t-SNE
│   └── re_ranking.py            # k-reciprocal re-ranking
├── utils/
│   └── visualizer.py            # training / evaluation plots
├── scripts/                     # runnable entry points
│   ├── train_*.py
│   ├── eval_*.py
│   ├── extract_*.py             # auto-metadata estimation
│   └── visualize_terrestrial.py
└── docs/
    ├── DATASETS.md              # how to download and lay out the datasets
    └── EXPERIMENTS.md           # how to reproduce every experiment
```

Datasets, model checkpoints, and generated figures are intentionally **not**
tracked in git (see `.gitignore`). Follow [`docs/DATASETS.md`](docs/DATASETS.md)
to obtain the data; checkpoints and results are created on the first run.

## Installation

Requires Python 3.10+ and, for training, an NVIDIA GPU with CUDA (≈16 GB of
VRAM is sufficient — lower the batch size for smaller cards).

**Option A — conda**

```bash
conda env create -f environment.yml
conda activate reid
```

**Option B — pip / virtualenv**

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Both install the OpenAI CLIP package from GitHub. `ultralytics` is optional and
only needed for the YOLO-based orientation estimator.

## Datasets

Two public datasets are used; neither is redistributed here. Download them and
place them under `data/` as described in
[`docs/DATASETS.md`](docs/DATASETS.md):

- **ATRW** — Amur Tiger Re-identification in the Wild.
- **LeopardID 2022** — Wild Me / LILA BC.

## Running experiments

Run the scripts from the **repository root** (each script adds the project root
to its import path automatically, so a fresh clone works with no extra setup):

```bash
python scripts/<name>.py [options]
```

See [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) for the full list of commands
and the order in which to run them.

### Quick start — merged Terrestrial protocol

```bash
# 1. train leak-free CLIP + MFA on the merged ATRW + LeopardID dataset
python scripts/train_terrestrial.py

# 2. evaluate: closed-set mAP / Rank-1 / Rank-5, overall and per species
python scripts/eval_terrestrial.py --checkpoint terrestrial_mfa_ft4_epoch60.pth

# 3. generate the qualitative figures
python scripts/visualize_terrestrial.py --checkpoint terrestrial_mfa_ft4_epoch60.pth
```

## Results

All values are mAP (%) unless stated otherwise. "Leak-free" means the individual
identifier has been removed from the training prompt.

**Single-dataset evaluation**

| Setting                          | Visual-only | CLIP + MFA (leak-free) |
|-----------------------------------|:-----------:|:----------------------:|
| ATRW — full 107-identity split    | 60.33       | 78.44                  |
| ATRW — 47-identity closed set     | 62.15       | 84.08                  |
| LeopardID 2022 — disjoint split   | 27.11       | 36.38                  |

**Merged Terrestrial protocol** (entity-disjoint, closed-set) — mAP / Rank-1 / Rank-5:

| Species | ResNet-50 baseline      | CLIP + MFA, leak-free   |
|---------|:-----------------------:|:-----------------------:|
| Tiger   | 77.59 / 98.72 / 99.43   | 81.41 / 99.22 / 99.69   |
| Leopard | 29.92 / 68.30 / 83.40   | 30.87 / 74.14 / 88.69   |

The CLIP backbone is frozen except for its last few transformer blocks, so the
trainable parameter count (~31 M) stays below the ~38.7 M of the ResNet-50
baseline.

## References

This code re-implements and builds on:

- Y. Li, D. Zhao, T. Qiao, Y. Wu, B. Pang, Y. S. Koh. *MetaWild: A Multimodal
  Dataset for Animal Re-Identification with Environmental Metadata.* ACM
  Multimedia, 2025.
- Xu et al. *Automatic re-identification of terrestrial mammals using deep
  learning and camera-trap images.* Global Ecology and Conservation, 2026.

The ATRW and LeopardID 2022 datasets are released by their respective authors —
please cite the original dataset papers when using them.

## License

Released under the MIT License — see [`LICENSE`](LICENSE). The ATRW and
LeopardID 2022 datasets are covered by their own separate terms set by their
respective providers.
