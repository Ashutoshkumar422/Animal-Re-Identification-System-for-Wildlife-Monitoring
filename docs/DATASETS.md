# Datasets

This project uses two public animal re-identification datasets. Neither is
redistributed in this repository — download them from the original sources and
place them under `data/`.

The datasets must live under `data/` at the repository root:

- `data/atrw/`     (`config.ATRW_ROOT`)
- `data/leopard/`  (`config.LEOPARD_ROOT`)

`config.py` resolves these paths relative to its own location, so scripts work
from any working directory. Both folders are git-ignored.

---

## ATRW — Amur Tiger Re-identification in the Wild

The re-identification subset of ATRW: 1,887 images of 107 individual tigers.

**Source:** the ATRW dataset released for the CVWC challenge
(<https://cvwc2019.github.io/>), also mirrored on LILA BC
(<https://lila.science/datasets/atrw>).

**Expected layout:**

```
data/atrw/
├── reid_list_train.csv     # one row per image:  "<identity_id>, <filename>"
└── train/                  # all 1,887 tiger crops
```

The loader (`data/atrw_dataset.py`) parses the CSV, remaps the identity ids to a
contiguous range, and builds per-identity train / gallery / query splits
(ratios fixed for reproducibility).

---

## LeopardID 2022

6,795 images of African leopards with COCO-style annotations (≈431 individuals).

**Source:** Wild Me / LILA BC —
<https://lila.science/datasets/leopard-id-2022/>

The official release ships every image under `train2022/` and leaves the
`val2022` / `test2022` splits as empty stubs. The loader therefore builds its
own train/test split from the train annotations alone.

**Expected layout:**

```
data/leopard/
├── annotations/
│   ├── instances_train2022.json    # the only populated annotation file
│   ├── instances_val2022.json      # empty stub in the official release
│   └── instances_test2022.json     # empty stub in the official release
└── images/
    └── train2022/                  # all 6,795 leopard images
```

COCO fields used by `data/leopard_dataset.py`:

| Field                                          | Role                                    |
|-------------------------------------------------|-----------------------------------------|
| `images[*].id` ↔ `annotations[*].image_id`      | join images to annotations              |
| `images[*].file_name`                           | filename inside `images/train2022/`     |
| `annotations[*].name` (UUID)                    | individual leopard identity             |
| `annotations[*].bbox` (`[x, y, w, h]`)          | crop applied at load time               |
| `annotations[*].viewpoint`                      | left / right flank (used by the merged protocol) |

A few images carry two annotations (two leopards); each annotation becomes its
own sample, cropped by its own bounding box.

The split protocol keeps train and test **identities disjoint**. Identities with
fewer than `LEOPARD_MIN_IMGS_PER_ID` images are dropped. The relevant ratios
(`LEOPARD_TRAIN_ID_FRAC`, `LEOPARD_GALLERY_FRAC`, `LEOPARD_SPLIT_SEED`) live in
`config.py`.

---

## Merged "Terrestrial Mammal" protocol

`data/terrestrial_dataset.py` builds a single merged dataset from ATRW and
LeopardID 2022, so **both datasets above must be present**.

- Each leopard is split into a left-flank and a right-flank "entity", because
  rosette patterns are not symmetric.
- Entities (from both species) are split disjointly into train / validation /
  test sets.
- Evaluation is closed-set (`Query = Gallery`) with mAP / Rank-1 / Rank-5.

The behaviour is controlled by the `TM_*` parameters in `config.py`
(`TM_MIN_IMGS_PER_ENTITY`, `TM_LEOPARD_MAX_ENTITIES`, `TM_TEST_ENTITY_FRAC`,
`TM_VAL_IMG_FRAC`, `TM_SPLIT_SEED`).

---

## Auto-derived metadata (optional)

Some experiments condition the model on per-image metadata estimated from the
images themselves (circadian state, temperature category, body orientation).
Generate it with:

```bash
python scripts/extract_metadata.py --dataset atrw
python scripts/extract_metadata.py --dataset leopard
```

This writes `data/<dataset>/metadata_auto.json`. See
[`EXPERIMENTS.md`](EXPERIMENTS.md) for the full metadata pipeline.

---

## Where things are stored

| Content                       | Location              | Tracked in git? |
|--------------------------------|-----------------------|:---------------:|
| Raw datasets                   | `data/atrw/`, `data/leopard/` | no       |
| Auto-metadata JSON             | `data/<dataset>/metadata_auto.json` | no |
| Model checkpoints              | `checkpoints/`        | no              |
| Plots, figures, metrics        | `results/`            | no              |

`checkpoints/` and `results/` are created automatically on the first run.
