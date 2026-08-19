# extract_orientation_transfer.py — Workaround for mmpose install failures.
#
# Trains a tiny CLIP-feature → orientation classifier on the leopard images
# (whose COCO `viewpoint` field is real human-annotated ground truth), then
# transfers the classifier to ATRW (cross-species, leopard → tiger). Both are
# Panthera felids with similar body geometry, so transfer should be reasonable.
#
# Updates ATRW's metadata_auto.json in place (only the 'orientation' field).
#
# No new heavy dependencies — uses CLIP (already loaded by MFA) + sklearn
# (already in the project's requirements for t-SNE).
#
# Honest caveats:
#   - Cross-species transfer; leopard pose ≠ tiger pose perfectly.
#   - Depends on COCO viewpoint label quality on leopard.
#   - We report held-out leopard validation accuracy before committing to ATRW
#     labels, so you can see if the classifier is trustworthy at all.
#
# Run order (after extract_metadata.py has produced both caches):
#   python scripts/extract_orientation_transfer.py

# Allow running this file directly (python scripts/<name>.py) from the repo root.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
import clip
from PIL import Image
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from config import ATRW_ROOT, LEOPARD_ROOT, CLIP_MODEL


CLASSES = ["front", "back", "left", "right"]


# ── Viewpoint label normalization ─────────────────────────────────────────────
# COCO viewpoint values can be compound ("frontleft", "backright") or other
# variants. Map to the four canonical orientations.

def normalize_viewpoint(raw: str) -> str | None:
    """Returns one of CLASSES or None if not classifiable."""
    if not raw:
        return None
    v = raw.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if v in CLASSES:
        return v
    # Compound — pick the dominant axis
    if v.startswith("front"):
        return "front"
    if v.startswith("back"):
        return "back"
    if v in ("rear",):
        return "back"
    if v.endswith("left"):
        return "left"
    if v.endswith("right"):
        return "right"
    return None


# ── CLIP feature extraction ───────────────────────────────────────────────────
@torch.no_grad()
def extract_clip_features(items, batch_size=32):
    """
    items : list of (key, full_image_path)
    Returns dict {key: np.ndarray (D,)} of L2-normalized CLIP image embeddings.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  loading CLIP {CLIP_MODEL} on {device}...")
    model, preprocess = clip.load(CLIP_MODEL, device=device)
    model.eval()

    results = {}
    batch_imgs, batch_keys = [], []

    def _flush():
        if not batch_imgs:
            return
        img_t   = torch.stack(batch_imgs).to(device)
        img_emb = model.encode_image(img_t).float()
        img_emb = F.normalize(img_emb, dim=-1).cpu().numpy()
        for i, key in enumerate(batch_keys):
            results[key] = img_emb[i]
        batch_imgs.clear()
        batch_keys.clear()

    for key, path in tqdm(items, desc="clip-feat"):
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


# ── Leopard training set ──────────────────────────────────────────────────────
def load_leopard_training_data():
    cache_path = os.path.join(LEOPARD_ROOT, "metadata_auto.json")
    img_dir    = os.path.join(LEOPARD_ROOT, "images")
    with open(cache_path) as f:
        cache = json.load(f)

    # Show raw viewpoint distribution before normalization (useful for debugging).
    raw_counts = Counter(meta.get("viewpoint", "missing") for meta in cache.values())
    print(f"  raw COCO viewpoints: {dict(raw_counts.most_common())}")

    items, labels = [], []
    for rel, meta in cache.items():
        norm = normalize_viewpoint(meta.get("viewpoint", ""))
        if norm is None:
            continue
        items.append((rel, os.path.join(img_dir, rel)))
        labels.append(norm)

    print(f"  normalized → {Counter(labels)}")
    return items, labels


# ── ATRW target set ───────────────────────────────────────────────────────────
def load_atrw_target_data():
    cache_path = os.path.join(ATRW_ROOT, "metadata_auto.json")
    with open(cache_path) as f:
        cache = json.load(f)
    img_dir = os.path.join(ATRW_ROOT, "train")
    items   = [(fn, os.path.join(img_dir, fn)) for fn in cache.keys()]
    return cache_path, cache, items


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Step 1 — Build leopard (training) set
    print("Step 1: loading leopard training data (CLIP features + COCO viewpoints)")
    leo_items, leo_labels = load_leopard_training_data()
    if len(leo_items) < 100:
        raise SystemExit(f"Only {len(leo_items)} leopard samples with valid viewpoint — "
                         "not enough to train a classifier.")
    if len(set(leo_labels)) < 2:
        raise SystemExit(f"Leopard viewpoints collapsed to one class: {set(leo_labels)} — "
                         "the COCO labels in this dataset don't have orientation variation.")

    leo_feats = extract_clip_features(leo_items)
    X = np.stack([leo_feats[key] for key, _ in leo_items if key in leo_feats])
    y = np.array([lbl for (key, _), lbl in zip(leo_items, leo_labels)
                  if key in leo_feats])

    # Step 2 — Train + validate (held-out leopard split)
    print("\nStep 2: training logistic regression classifier")
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if min(Counter(y).values()) >= 2 else None,
    )
    # Note: sklearn ≥1.5 removed `multi_class` arg — multinomial is the default
    # for >2 classes now.
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(X_tr, y_tr)
    train_acc = clf.score(X_tr, y_tr)
    val_acc   = clf.score(X_va, y_va)
    print(f"  train accuracy: {train_acc:.3f}")
    print(f"  val accuracy:   {val_acc:.3f}    (held-out leopard, 20%)")
    print("\n  per-class report on held-out leopard:")
    print(classification_report(y_va, clf.predict(X_va), zero_division=0))

    if val_acc < 0.40:
        print("\n  WARNING: held-out leopard val accuracy is low (<40%). The classifier")
        print("  is barely better than random — labels on ATRW will be unreliable.")
        print("  Recommend dropping orientation from ATRW entirely and using only")
        print("  circadian + temperature in the prompt. Continuing anyway...")

    # Step 3 — Apply to ATRW
    print("\nStep 3: classifying ATRW images")
    atrw_cache_path, atrw_cache, atrw_items = load_atrw_target_data()
    atrw_feats = extract_clip_features(atrw_items)

    keys_test = [fn for fn, _ in atrw_items if fn in atrw_feats]
    X_test    = np.stack([atrw_feats[fn] for fn in keys_test])
    preds     = clf.predict(X_test)

    print(f"  ATRW orientation: {Counter(preds)}")

    # Sanity check: did the classifier just collapse to one class?
    if len(set(preds)) < 3:
        print("  WARNING: predictions collapsed to <3 classes — transfer is weak.")

    for key, pred in zip(keys_test, preds):
        if key in atrw_cache:
            atrw_cache[key]["orientation"] = pred

    with open(atrw_cache_path, "w") as f:
        json.dump(atrw_cache, f)
    print(f"\nUpdated {atrw_cache_path}")


if __name__ == "__main__":
    main()
