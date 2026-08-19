# data/terrestrial_dataset.py
# Reproduction of the merged "Terrestrial Mammal Dataset" from
# Xu et al. 2026 (ARNet, Global Ecology and Conservation 68 e04223),
# so CLIP+MFA results drop directly into the journal paper's Tables 3-5.
#
# Xu et al.'s protocol (their Section 3.1, Tables 1-2):
#   - Merge ATRW (Amur tiger) + a random subsample of LeopardID 2022.
#   - Each leopard is split into separate left-flank / right-flank "entities"
#     because rosette patterns are asymmetric; front/back views are screened
#     out as "unsuitable for individual identification".
#   - ATRW re-ID identities are already per-flank entities (107 available).
#   - Entities are split ~7:3 into train / test (entity-disjoint). A validation
#     cut is carved from the train entities at the image level.
#   - Evaluation is closed-set with Query = Gallery: every test image is ranked
#     against all OTHER test images (leave-one-out). See eval script.
#   - Metrics: mAP, Rank-1/5/10.
#
# Honest reproduction caveats (to be stated in the report):
#   1. Xu et al. randomly subsampled LeopardID; the exact subsample is not
#      published, so our reconstruction cannot be byte-identical.
#   2. We only have ATRW's public training split (107 entities / ~1,887 imgs);
#      Xu et al. additionally used ATRW's test split (47 entities). Our merged
#      set therefore has fewer tiger entities than theirs.
#   Because of (1)-(2), comparison against Xu's published numbers is a
#   protocol-level reproduction, not a dataset-identical one. For an internally
#   controlled comparison, re-run baselines on THIS reconstruction.
#
# __getitem__ returns (image_tensor, prompt_string, label_int, species_string),
# matching data/atrw_dataset.py and data/leopard_dataset.py so the existing
# model and metrics code work unchanged. Prompts are NEUTRAL (no individual id)
# — the leak-free recipe from Findings 1-2.
#
# Quick check (run from the project root, on the GPU machine):
#   python data/terrestrial_dataset.py

import os
import sys
import json
import random
from collections import defaultdict

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Make `from config import ...` work whether imported as a module or run direct.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ATRW_ROOT, LEOPARD_ROOT, LEOPARD_IMAGE_SUBDIR, IMAGE_SIZE,
    TM_MIN_IMGS_PER_ENTITY, TM_LEOPARD_MAX_ENTITIES,
    TM_TEST_ENTITY_FRAC, TM_VAL_IMG_FRAC, TM_SPLIT_SEED,
)
from data.atrw_dataset import _parse_train_csv


# ── Transforms ────────────────────────────────────────────────────────────────
# CLIP normalization (the backbone is CLIP ViT-B/16).
# Augmentation mirrors Xu et al.: random rotation +/-15 deg + random masking.
# NOTE: no horizontal flip — left/right flank entities are kept distinct
# precisely because rosette/stripe patterns are asymmetric, so flipping a
# left flank would forge a right flank and corrupt the label.
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

_TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(15),                       # Xu: random rotation +/-15 deg
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.20)), # Xu: random masking
])

_EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
])


# ── Neutral, leak-free prompts (Findings 1-2: no individual id in the prompt) ─
NEUTRAL_PROMPTS = {
    "tiger":   ("A photo of a tiger in cool temperature, "
                "with face direction front, captured during the day."),
    "leopard": ("A photo of a leopard in cool temperature, "
                "with face direction front, captured during the day."),
}


# ── Entity gathering ──────────────────────────────────────────────────────────

def _gather_atrw_entities(root):
    """
    ATRW re-ID identities (from reid_list_train.csv) are already per-flank
    entities. Returns { ('tiger', atrw_label) : [(full_path, None), ...] }.
    """
    csv_path = os.path.join(root, "reid_list_train.csv")
    img_dir  = os.path.join(root, "train")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"ATRW csv not found: {csv_path}")

    id_to_files = _parse_train_csv(csv_path)         # {label: [filenames]}
    entities = {}
    for label, fnames in id_to_files.items():
        imgs = [(os.path.join(img_dir, fn), None)
                for fn in fnames
                if os.path.isfile(os.path.join(img_dir, fn))]
        if imgs:
            entities[("tiger", label)] = imgs
    return entities


def _side_from_viewpoint(vp):
    """Map a COCO viewpoint string to 'left' / 'right', or None to screen out."""
    if not vp:
        return None
    v = str(vp).strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if "left" in v:
        return "left"
    if "right" in v:
        return "right"
    return None      # front / back / unknown -> screened out


def _gather_leopard_entities(root):
    """
    Parse the LeopardID COCO json. Each annotation is grouped by
    (uuid, flank-side); front/back/unknown views are screened out.
    Returns ({ ('leopard', uuid, side) : [(full_path, bbox), ...] }, n_screened).
    """
    json_path = os.path.join(root, "annotations", "instances_train2022.json")
    img_dir   = os.path.join(root, "images", LEOPARD_IMAGE_SUBDIR)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Leopard COCO json not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    imgs_by_id = {im["id"]: im["file_name"] for im in coco.get("images", [])}

    entities = defaultdict(list)
    n_screened = 0
    for a in coco.get("annotations", []):
        uuid = a.get("name")
        if not uuid:
            continue
        fname = imgs_by_id.get(a["image_id"])
        if not fname:
            continue
        side = _side_from_viewpoint(a.get("viewpoint"))
        if side is None:
            n_screened += 1
            continue
        full = os.path.join(img_dir, fname)
        if not os.path.isfile(full):
            continue
        bbox = a.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox = tuple(float(v) for v in bbox)
        else:
            bbox = None
        entities[("leopard", uuid, side)].append((full, bbox))
    return dict(entities), n_screened


# ── Split construction ────────────────────────────────────────────────────────

def _build_splits(atrw_entities, leopard_entities):
    """
    Returns (train_samples, val_samples, test_samples,
             n_train_entities, n_test_entities).
    Each sample = (full_path, bbox_or_None, label_int, species_str).
    Train labels are contiguous [0, n_train_entities); test labels are
    independently contiguous [0, n_test_entities).
    """
    rng = random.Random(TM_SPLIT_SEED)

    def _filter_min(d):
        return {k: v for k, v in d.items() if len(v) >= TM_MIN_IMGS_PER_ENTITY}

    atrw_e = _filter_min(atrw_entities)
    leo_e  = _filter_min(leopard_entities)

    # Subsample leopard entities (Xu et al. "randomly subsampled" LeopardID).
    leo_keys = sorted(leo_e.keys())
    if len(leo_keys) > TM_LEOPARD_MAX_ENTITIES:
        rng.shuffle(leo_keys)
        leo_keys = leo_keys[:TM_LEOPARD_MAX_ENTITIES]
    leo_e = {k: leo_e[k] for k in leo_keys}

    # Entity-disjoint train/test split, stratified per species so both species
    # appear in the test set (needed for the Table 5 per-species evaluation).
    train_entities, test_entities = [], []
    for pool in (atrw_e, leo_e):
        keys = sorted(pool.keys())
        rng.shuffle(keys)
        n_test = int(round(len(keys) * TM_TEST_ENTITY_FRAC))
        for k in keys[:n_test]:
            test_entities.append((k, pool[k]))
        for k in keys[n_test:]:
            train_entities.append((k, pool[k]))

    # Contiguous label spaces (train and test are independent).
    train_label = {k: i for i, (k, _) in enumerate(train_entities)}
    test_label  = {k: i for i, (k, _) in enumerate(test_entities)}

    train_s, val_s, test_s = [], [], []

    # Train entities: image-level holdout for validation (val ids subset of
    # train ids, matching Xu's "Val 160" == "Train 160" entity count).
    for key, imgs in train_entities:
        species = key[0]
        label   = train_label[key]
        imgs    = list(imgs)
        rng.shuffle(imgs)
        n_val = int(round(len(imgs) * TM_VAL_IMG_FRAC))
        n_val = min(n_val, len(imgs) - 1)     # keep >= 1 training image
        for path, bbox in imgs[n_val:]:
            train_s.append((path, bbox, label, species))
        for path, bbox in imgs[:n_val]:
            val_s.append((path, bbox, label, species))

    # Test entities: every image goes into the single closed-set Query=Gallery set.
    for key, imgs in test_entities:
        species = key[0]
        label   = test_label[key]
        for path, bbox in imgs:
            test_s.append((path, bbox, label, species))

    return train_s, val_s, test_s, len(train_entities), len(test_entities)


# ── Dataset class ─────────────────────────────────────────────────────────────

class TerrestrialDataset(Dataset):
    """
    samples : list of (full_image_path, bbox_or_None, label_int, species_str)
    split   : "train" -> augmented; "val"/"test" -> deterministic transform.

    The "test" split is the closed-set evaluation set: it is used as BOTH the
    query and the gallery, with each query ranked against all other test
    images (leave-one-out). See the eval script for self-exclusion handling.
    """
    def __init__(self, samples, split="train"):
        self.samples     = samples
        self.split       = split
        self.transform   = _TRAIN_TRANSFORM if split == "train" else _EVAL_TRANSFORM
        self.num_classes = len({lbl for _, _, lbl, _ in samples})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, bbox, label, species = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if bbox is not None:
            x, y, w, h = bbox
            x, y   = max(0, int(x)), max(0, int(y))
            x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
            if x2 > x and y2 > y:
                img = img.crop((x, y, x2, y2))

        img = self.transform(img)
        prompt = NEUTRAL_PROMPTS[species]
        return img, prompt, label, species


# ── Public entry point ────────────────────────────────────────────────────────

def _summarize(tag, samples):
    """Return (n_images, {species: n_images})."""
    by_sp = defaultdict(int)
    for _, _, _, sp in samples:
        by_sp[sp] += 1
    return len(samples), dict(by_sp)


def load_terrestrial(atrw_root=ATRW_ROOT, leopard_root=LEOPARD_ROOT, verbose=True):
    """
    Build the merged Terrestrial Mammal Dataset.
    Returns (train_ds, val_ds, test_ds).
      - train_ds / val_ds share the same entity (label) space.
      - test_ds is the disjoint closed-set Query=Gallery evaluation set.
    """
    atrw_e            = _gather_atrw_entities(atrw_root)
    leo_e, n_screened = _gather_leopard_entities(leopard_root)

    train_s, val_s, test_s, n_train_ent, n_test_ent = _build_splits(atrw_e, leo_e)

    train_ds = TerrestrialDataset(train_s, split="train")
    val_ds   = TerrestrialDataset(val_s,   split="val")
    test_ds  = TerrestrialDataset(test_s,  split="test")

    if verbose:
        tr_n, tr_sp = _summarize("train", train_s)
        va_n, va_sp = _summarize("val",   val_s)
        te_n, te_sp = _summarize("test",  test_s)
        tiger_test_ent = len({lbl for _, _, lbl, sp in test_s if sp == "tiger"})
        leo_test_ent   = len({lbl for _, _, lbl, sp in test_s if sp == "leopard"})

        print("\n" + "=" * 64)
        print(" Terrestrial Mammal Dataset — reconstructed (Xu et al. protocol)")
        print("=" * 64)
        print(f" leopard annotations screened out (front/back/unknown view): {n_screened}")
        print(f" entities kept: {n_train_ent} train  +  {n_test_ent} test "
              f"(>= {TM_MIN_IMGS_PER_ENTITY} imgs each)")
        print(f" test entities by species: {tiger_test_ent} tiger / "
              f"{leo_test_ent} leopard")
        print("-" * 64)
        print(f" {'Split':<16}{'Entities':>10}{'Images':>10}   per-species images")
        print(f" {'Train':<16}{n_train_ent:>10}{tr_n:>10}   {tr_sp}")
        print(f" {'Val':<16}{n_train_ent:>10}{va_n:>10}   {va_sp}")
        print(f" {'Query/Gallery':<16}{n_test_ent:>10}{te_n:>10}   {te_sp}")
        print("-" * 64)
        print(" Xu et al. Table 2 (target):  Train 160/2327  Val 160/561  "
              "Query/Gallery 90/1171")
        print(" NOTE: exact match is impossible — Xu's leopard subsample is")
        print("       unpublished and we lack ATRW's 47-entity test split.")
        print("       Tune TM_LEOPARD_MAX_ENTITIES / TM_TEST_ENTITY_FRAC in")
        print("       config.py if the counts need to be nudged closer.")
        print("=" * 64 + "\n")

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    load_terrestrial(verbose=True)
