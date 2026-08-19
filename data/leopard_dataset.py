# data/leopard_dataset.py
# Loader for LeopardID 2022 (Wild Me / LILA BC) — mirrors data/atrw_dataset.py interface.
#
# Expected layout (matches the official release):
#   <LEOPARD_ROOT>/
#     annotations/
#       instances_train2022.json     # COCO-format — the only file with real data
#       instances_val2022.json       # empty stub in the official release (ignored)
#       instances_test2022.json      # empty stub in the official release (ignored)
#     images/
#       train2022/                   # all 6,795 leopard images live here
#       val2022/                     # empty in the official release
#       test2022/                    # empty in the official release
#
# The official release ships every image under train2022/ and leaves test/val
# empty, so we construct a disjoint-identity train/test split ourselves
# (mirrors Xu et al. 2026's open-set evaluation protocol).
#
# COCO mapping notes:
#   - identity per leopard = annotations[*].name  (a UUID string, 431 unique)
#   - bbox (COCO [x, y, w, h]) = annotations[*].bbox
#   - file path joined as: <LEOPARD_ROOT>/images/<LEOPARD_IMAGE_SUBDIR>/<file_name>
#   - some images carry 2 annotations (different leopards) — each becomes its
#     own training sample, cropped by its own bbox
#
# __getitem__ returns: (image_tensor, prompt_string, label_int, species_string)

import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

from config import (
    LEOPARD_ROOT,
    LEOPARD_IMAGE_SUBDIR,
    LEOPARD_MIN_IMGS_PER_ID,
    LEOPARD_TRAIN_ID_FRAC,
    LEOPARD_GALLERY_FRAC,
    LEOPARD_SPLIT_SEED,
    IMAGE_SIZE,
)


# ── CLIP-style transforms (kept identical to data/atrw_dataset.py) ────────────
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
])

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
])


# ── COCO JSON parsing ─────────────────────────────────────────────────────────

def _parse_coco_json(json_path: str, image_subdir: str
                    ) -> Tuple[Dict[int, List[Tuple[str, tuple]]], Dict[str, int]]:
    """
    Read a COCO-format annotations file and group (rel_path, bbox) entries
    by identity.

    Identity field is `annotations[*].name` (UUID string).
    Bbox is COCO `[x, y, w, h]`.

    Returns:
        id_to_entries : { remapped_int_id : [ (rel_path, bbox_or_None), ... ] }
        id_map        : { original_uuid_string : remapped_int_id }
    """
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    imgs_by_id = {im["id"]: im["file_name"] for im in coco.get("images", [])}

    raw: Dict[str, List[Tuple[str, tuple]]] = defaultdict(list)
    for a in coco.get("annotations", []):
        ident = a.get("name")
        if not ident:
            continue
        fname = imgs_by_id.get(a["image_id"])
        if not fname:
            continue
        rel_path = f"{image_subdir}/{fname}"
        bbox = a.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox = tuple(float(v) for v in bbox)
        else:
            bbox = None
        raw[ident].append((rel_path, bbox))

    # Drop low-population identities, then remap UUID strings → 0-indexed ints.
    kept_uuids = sorted(u for u, v in raw.items() if len(v) >= LEOPARD_MIN_IMGS_PER_ID)
    id_map = {orig: idx for idx, orig in enumerate(kept_uuids)}
    id_to_entries = {id_map[u]: raw[u] for u in kept_uuids}
    return id_to_entries, id_map


# ── Disjoint-identity split ───────────────────────────────────────────────────

def _make_disjoint_splits(id_to_entries):
    """
    Disjoint-identity protocol:
      - LEOPARD_TRAIN_ID_FRAC of identities → training (all their images)
      - Remaining identities → test, each split into gallery / query
    Each sample = (rel_path, bbox_or_None, label_int)
    """
    rng = random.Random(LEOPARD_SPLIT_SEED)
    all_ids = sorted(id_to_entries.keys())
    rng.shuffle(all_ids)

    n_train = int(round(len(all_ids) * LEOPARD_TRAIN_ID_FRAC))
    train_ids = set(all_ids[:n_train])
    test_ids  = set(all_ids[n_train:])

    train_samples, gallery_samples, query_samples = [], [], []

    # Re-index train identities to a contiguous range [0, n_train_ids) so the
    # classifier head's output dim matches the actual label values.
    train_id_remap = {tid: i for i, tid in enumerate(sorted(train_ids))}
    for old_label in train_ids:
        new_label = train_id_remap[old_label]
        for path, bbox in id_to_entries[old_label]:
            train_samples.append((path, bbox, new_label))

    # Re-index test identities to a contiguous range starting at 0
    # (train and test never share label space, so this is safe).
    test_id_remap = {tid: i for i, tid in enumerate(sorted(test_ids))}
    for tid in sorted(test_ids):
        entries = list(id_to_entries[tid])
        rng.shuffle(entries)
        n_gal = max(1, int(round(len(entries) * LEOPARD_GALLERY_FRAC)))
        gal, qry = entries[:n_gal], entries[n_gal:]
        # Ensure each test identity has at least one query image
        if len(qry) == 0 and len(gal) > 1:
            qry = [gal.pop()]
        new_label = test_id_remap[tid]
        for path, bbox in gal:
            gallery_samples.append((path, bbox, new_label))
        for path, bbox in qry:
            query_samples.append((path, bbox, new_label))

    return train_samples, gallery_samples, query_samples, len(train_ids), len(test_ids)


# ── Dataset class ─────────────────────────────────────────────────────────────

PROMPT_LEOPARD = (
    "A photo of a leopard individual {label} in cool temperature, "
    "with face direction front, captured during the day."
)


class LeopardDataset(Dataset):
    """
    samples : list of (relative_image_path, bbox_or_None, label_int)
              relative to <LEOPARD_ROOT>/images/
    img_dir : <LEOPARD_ROOT>/images/
    split   : "train" | "gallery" | "query" — controls augmentation
    """
    def __init__(self, samples, img_dir, split: str = "train"):
        self.samples   = samples
        self.img_dir   = img_dir
        self.split     = split
        self.transform = _TRAIN_TRANSFORM if split == "train" else _EVAL_TRANSFORM
        self.num_classes = len({lbl for _, _, lbl in samples})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, bbox, label = self.samples[idx]
        img_path = os.path.join(self.img_dir, rel_path)
        img = Image.open(img_path).convert("RGB")

        if bbox is not None:
            x, y, w, h = bbox
            x, y = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
            if x2 > x and y2 > y:
                img = img.crop((x, y, x2, y2))

        img = self.transform(img)
        prompt = PROMPT_LEOPARD.format(label=label)
        return img, prompt, label, "leopard"


# ── Public entry point ────────────────────────────────────────────────────────

def load_leopard(root: str = LEOPARD_ROOT, verbose: bool = True):
    """
    Returns (train_ds, gallery_ds, query_ds).
    Reads <root>/annotations/instances_train2022.json — the only annotation
    file in the official release that contains real data — and constructs
    a disjoint-identity train/test split from it.
    """
    json_path = os.path.join(root, "annotations", "instances_train2022.json")
    img_dir   = os.path.join(root, "images")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"Expected {json_path}. See README_LEOPARD.md for the expected layout."
        )

    id_to_entries, id_map = _parse_coco_json(json_path, LEOPARD_IMAGE_SUBDIR)
    train_s, gal_s, qry_s, n_train_ids, n_test_ids = _make_disjoint_splits(id_to_entries)

    train_ds   = LeopardDataset(train_s, img_dir, split="train")
    gallery_ds = LeopardDataset(gal_s,   img_dir, split="gallery")
    query_ds   = LeopardDataset(qry_s,   img_dir, split="query")

    if verbose:
        n_total_ids = len(id_map)
        print(f"[leopard] identities kept (≥{LEOPARD_MIN_IMGS_PER_ID} imgs): {n_total_ids}")
        print(f"[leopard] disjoint split — train ids: {n_train_ids} | test ids: {n_test_ids}")
        print(f"[leopard] images — train: {len(train_s)} | "
              f"gallery: {len(gal_s)} | query: {len(qry_s)}")

    # Sanity checks. Note: train labels live in [0, n_train_ids) and test labels
    # are independently re-indexed to [0, n_test_ids) — the integer label spaces
    # overlap by construction, but they belong to disjoint *real identities*
    # (the underlying UUIDs never overlap, since identity-level split happens
    # before re-indexing). The model only uses train labels for classification
    # and only within-test labels for retrieval, so this is safe.
    gal_ids = {l for _, _, l in gal_s}
    qry_ids = {l for _, _, l in qry_s}
    assert gal_ids == qry_ids, \
        "closed-set test invariant broken: every test identity must appear " \
        "in both gallery and query"

    return train_ds, gallery_ds, query_ds
