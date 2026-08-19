# extract_metadata.py — One-shot preprocessor for Option A.
#
# Walks the dataset's images, computes per-image:
#   - mean grayscale brightness        → day/night categorical
#   - mean RGB                         → temperature categorical (warm/cool/cold)
# For leopard, also extracts:
#   - viewpoint                        → face direction (REAL annotation from COCO)
#
# Saves to <root>/metadata_auto.json keyed by relative image path. Training and
# eval scripts read this cache and use it to build per-image prompts that
# replace the dummy "cool / front / day" we've been using.
#
# Run once before train_*_meta.py / eval_*_meta.py:
#   python scripts/extract_metadata.py --dataset leopard
#   python scripts/extract_metadata.py --dataset atrw

# Allow running this file directly (python scripts/<name>.py) from the repo root.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm

from config import LEOPARD_ROOT, LEOPARD_IMAGE_SUBDIR, ATRW_ROOT, CLIP_MODEL


# ── Heuristics (project_context.txt §11) ──────────────────────────────────────
def _categorize_brightness(brightness_mean: float) -> str:
    """Mean grayscale > 80 → day, else night."""
    return "day" if brightness_mean > 80.0 else "night"


def _categorize_temperature(rgb_mean) -> str:
    """
    Warm/cool/cold from R-vs-B dominance.
      b > r       → cold
      r > b + 20  → warm
      otherwise   → cool
    """
    r, _g, b = float(rgb_mean[0]), float(rgb_mean[1]), float(rgb_mean[2])
    if b > r:        return "cold"
    if r > b + 20:   return "warm"
    return "cool"


def _stats_from_image(img: Image.Image) -> tuple:
    """Return (brightness_mean, [r_mean, g_mean, b_mean]) — small downsample for speed."""
    small = img.convert("RGB").resize((64, 64))
    arr   = np.asarray(small, dtype=np.float32)
    rgb_mean   = arr.reshape(-1, 3).mean(axis=0).tolist()
    brightness = arr.mean()
    return float(brightness), rgb_mean


# ── Zero-shot orientation classifier (CLIP) ───────────────────────────────────
# Used when the dataset has no ground-truth viewpoint annotation (e.g. ATRW).
# Compares each image embedding against four text prompts and picks argmax —
# noisier than a real animal-pose model, but uses the same CLIP we already
# load for MFA, no extra dependency, and gives real per-image variation in the
# prompt instead of a constant "front".

_ORIENTATION_CLASSES = ["front", "back", "left", "right"]


def _orientation_prompt_ensembles(species: str):
    """
    Multiple prompt phrasings per class; we'll average the text embeddings to
    produce one robust class anchor each. Reduces single-prompt CLIP bias
    (in particular the "left" >> "right" skew seen with the naive prompt).
    """
    return {
        "front": [
            f"a {species} facing the camera",
            f"front view of a {species}",
            f"a {species} looking at the camera",
            f"a {species} with its face toward the camera",
            f"head-on view of a {species}",
        ],
        "back": [
            f"a {species} facing away from the camera",
            f"rear view of a {species}",
            f"the back of a {species}",
            f"a {species} walking away from the camera",
            f"a {species} viewed from behind",
        ],
        "left": [
            f"left side profile of a {species}",
            f"a {species} walking from right to left",
            f"a {species} with its left flank visible",
            f"a {species} viewed from its left side",
            f"the left side of a {species}",
        ],
        "right": [
            f"right side profile of a {species}",
            f"a {species} walking from left to right",
            f"a {species} with its right flank visible",
            f"a {species} viewed from its right side",
            f"the right side of a {species}",
        ],
    }


@torch.no_grad()
def _build_class_embeddings(model, prompt_ensembles, device):
    """For each class, encode N prompts → mean → renormalize. Returns (4, D) tensor."""
    class_embs = []
    for cls in _ORIENTATION_CLASSES:
        prompts = prompt_ensembles[cls]
        tokens  = clip.tokenize(prompts).to(device)
        embs    = model.encode_text(tokens).float()
        embs    = F.normalize(embs, dim=-1)
        mean    = embs.mean(dim=0)
        class_embs.append(F.normalize(mean, dim=-1))
    return torch.stack(class_embs, dim=0)   # (4, D)


@torch.no_grad()
def _clip_orientation(items, species: str, batch_size: int = 32) -> dict:
    """
    items : list of (cache_key, full_image_path)
    Returns {cache_key: orientation_str}.
    Loads CLIP once, batches images for speed.
    Uses prompt ensembling (N prompts per class, averaged) to reduce single-
    prompt bias.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  loading CLIP {CLIP_MODEL} on {device} for orientation classification...")
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()

    prompt_ensembles = _orientation_prompt_ensembles(species)
    text_embs        = _build_class_embeddings(model, prompt_ensembles, device)  # (4, D)

    results = {}
    batch_imgs, batch_keys = [], []

    def _flush():
        if not batch_imgs:
            return
        img_t   = torch.stack(batch_imgs).to(device)
        img_emb = model.encode_image(img_t).float()
        img_emb = F.normalize(img_emb, dim=-1)
        sim     = img_emb @ text_embs.T   # (N, 4)
        for i, key in enumerate(batch_keys):
            results[key] = _ORIENTATION_CLASSES[int(sim[i].argmax().item())]
        batch_imgs.clear()
        batch_keys.clear()

    for key, path in tqdm(items, desc=f"clip-orient ({species})"):
        try:
            with Image.open(path) as im:
                t = preprocess(im.convert("RGB"))
            batch_imgs.append(t)
            batch_keys.append(key)
            if len(batch_imgs) >= batch_size:
                _flush()
        except Exception as e:
            print(f"  skip {path}: {e}")
    _flush()
    return results


# ── Leopard ───────────────────────────────────────────────────────────────────
def extract_leopard(root: str = LEOPARD_ROOT) -> dict:
    json_path = os.path.join(root, "annotations", "instances_train2022.json")
    img_dir   = os.path.join(root, "images")
    with open(json_path) as f:
        coco = json.load(f)
    imgs_by_id = {im["id"]: im["file_name"] for im in coco["images"]}

    # First-annotation viewpoint per filename.
    viewpoint_by_file = {}
    for a in coco["annotations"]:
        fn = imgs_by_id.get(a["image_id"])
        if not fn or fn in viewpoint_by_file:
            continue
        vp = (a.get("viewpoint") or "").strip().lower()
        viewpoint_by_file[fn] = vp if vp else "front"

    cache = {}
    for fn, vp in tqdm(viewpoint_by_file.items(), desc="leopard"):
        full = os.path.join(img_dir, LEOPARD_IMAGE_SUBDIR, fn)
        if not os.path.isfile(full):
            continue
        try:
            with Image.open(full) as im:
                bright, rgb = _stats_from_image(im)
        except Exception as e:
            print(f"  skip {fn}: {e}")
            continue
        cache[f"{LEOPARD_IMAGE_SUBDIR}/{fn}"] = {
            "brightness_mean": bright,
            "rgb_mean":        rgb,
            "viewpoint":       vp,
            "circadian":       _categorize_brightness(bright),
            "temperature":     _categorize_temperature(rgb),
            "orientation":     vp,
        }

    out = os.path.join(root, "metadata_auto.json")
    with open(out, "w") as f:
        json.dump(cache, f)
    print(f"\nWrote {out}  ({len(cache)} entries)")

    # Quick distribution print.
    print(f"  circadian: {Counter(v['circadian'] for v in cache.values())}")
    print(f"  temp:      {Counter(v['temperature'] for v in cache.values())}")
    print(f"  viewpoint: {Counter(v['orientation'] for v in cache.values())}")
    return cache


# ── ATRW ──────────────────────────────────────────────────────────────────────
def extract_atrw(root: str = ATRW_ROOT) -> dict:
    """
    ATRW has no viewpoint annotations. Orientation is filled in by zero-shot
    CLIP classification ("front"/"back"/"left"/"right" — noisy but real
    per-image signal, beats a constant default).
    """
    img_dir  = os.path.join(root, "train")
    csv_path = os.path.join(root, "reid_list_train.csv")

    files = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # rows look like "1, 001234.jpg" — take the second field
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                files.append(parts[1])

    files = sorted(set(files))
    cache = {}
    for fn in tqdm(files, desc="atrw stats"):
        full = os.path.join(img_dir, fn)
        if not os.path.isfile(full):
            continue
        try:
            with Image.open(full) as im:
                bright, rgb = _stats_from_image(im)
        except Exception as e:
            print(f"  skip {fn}: {e}")
            continue
        cache[fn] = {
            "brightness_mean": bright,
            "rgb_mean":        rgb,
            "circadian":       _categorize_brightness(bright),
            "temperature":     _categorize_temperature(rgb),
            "orientation":     "front",   # placeholder, overwritten below
        }

    # Fill in orientation via CLIP zero-shot.
    items = [(fn, os.path.join(img_dir, fn)) for fn in cache.keys()]
    orientations = _clip_orientation(items, species="tiger")
    for fn, orient in orientations.items():
        if fn in cache:
            cache[fn]["orientation"] = orient

    out = os.path.join(root, "metadata_auto.json")
    with open(out, "w") as f:
        json.dump(cache, f)
    print(f"\nWrote {out}  ({len(cache)} entries)")
    print(f"  circadian:   {Counter(v['circadian'] for v in cache.values())}")
    print(f"  temp:        {Counter(v['temperature'] for v in cache.values())}")
    print(f"  orientation: {Counter(v['orientation'] for v in cache.values())}")
    return cache


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["leopard", "atrw"], required=True)
    args = parser.parse_args()
    if args.dataset == "leopard":
        extract_leopard()
    else:
        extract_atrw()
