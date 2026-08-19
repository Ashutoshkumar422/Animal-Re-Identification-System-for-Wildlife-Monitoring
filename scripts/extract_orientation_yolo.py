# extract_orientation_yolo.py — YOLO-based orientation labels for ATRW.
#
# Same cross-species transfer idea as extract_orientation_transfer.py, but uses
# a fine-tuned YOLOv8-cls CNN as the classifier instead of logistic regression
# on frozen CLIP features. Heavier (~2-3 min training, vs <1 min for LR) but
# the end-to-end CNN may capture orientation cues better.
#
# Setup (much simpler than mmpose):
#   pip install ultralytics
#
# How it works:
#   Step 1 — Use leopard images (which have real COCO viewpoint annotations) to
#            build a 4-class image dataset under data/leopard/_yolo_orient/
#            with train/val/{front,back,left,right}/ folder structure.
#   Step 2 — Fine-tune YOLOv8n-cls on it (downloads ~6 MB pretrained weights).
#            Reports val accuracy at the end of training.
#   Step 3 — Predict orientations on all ATRW images.
#   Step 4 — Update data/atrw/metadata_auto.json's 'orientation' field in place.
#
# Run:
#   python scripts/extract_orientation_yolo.py

# Allow running this file directly (python scripts/<name>.py) from the repo root.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import shutil
import random
from collections import Counter
from pathlib import Path

from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit(
        "ultralytics not installed.\n"
        "Install with:\n  pip install ultralytics"
    )

from config import ATRW_ROOT, LEOPARD_ROOT


CLASSES   = ["front", "back", "left", "right"]
TRAIN_FRAC = 0.8
SEED       = 42
EPOCHS     = 20
IMGSZ      = 224
BATCH      = 64
MIN_PER_CLASS = 20      # minimum samples per class to attempt training


# ── Viewpoint label normalization (matches extract_orientation_transfer.py) ──
def normalize_viewpoint(raw: str):
    if not raw:
        return None
    v = raw.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    if v in CLASSES:
        return v
    if v.startswith("front"):  return "front"
    if v.startswith("back"):   return "back"
    if v in ("rear",):         return "back"
    if v.endswith("left"):     return "left"
    if v.endswith("right"):    return "right"
    return None


# ── Step 1 — Build YOLOv8-cls dataset from leopard ───────────────────────────
def build_yolo_dataset(work_dir: Path) -> Path:
    leopard_cache_path = os.path.join(LEOPARD_ROOT, "metadata_auto.json")
    img_dir            = os.path.join(LEOPARD_ROOT, "images")
    if not os.path.isfile(leopard_cache_path):
        raise SystemExit(
            f"{leopard_cache_path} not found — run "
            "`python scripts/extract_metadata.py --dataset leopard` first."
        )
    with open(leopard_cache_path) as f:
        cache = json.load(f)

    by_class = {c: [] for c in CLASSES}
    raw_dist = Counter()
    for rel, meta in cache.items():
        raw_vp = meta.get("viewpoint", "")
        raw_dist[raw_vp] += 1
        norm = normalize_viewpoint(raw_vp)
        if not norm:
            continue
        full = os.path.join(img_dir, rel)
        if os.path.isfile(full):
            by_class[norm].append(full)

    print(f"  raw COCO viewpoints: {dict(raw_dist.most_common())}")
    print(f"  normalized: { {c: len(by_class[c]) for c in CLASSES} }")

    for c in CLASSES:
        if len(by_class[c]) < MIN_PER_CLASS:
            print(f"  WARNING: class '{c}' has only {len(by_class[c])} samples "
                  f"(< {MIN_PER_CLASS}); fine-tuning may underfit.")

    if work_dir.exists():
        shutil.rmtree(work_dir)
    for split in ("train", "val"):
        for c in CLASSES:
            (work_dir / split / c).mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    for c, paths in by_class.items():
        rng.shuffle(paths)
        n_train = int(len(paths) * TRAIN_FRAC)
        for i, src in enumerate(paths):
            split = "train" if i < n_train else "val"
            dst   = work_dir / split / c / Path(src).name
            try:
                os.symlink(os.path.abspath(src), dst)
            except (OSError, NotImplementedError):
                shutil.copy(src, dst)
    return work_dir


# ── Step 2 — Train classifier ────────────────────────────────────────────────
def train_classifier(dataset_path: Path) -> YOLO:
    print(f"\nStep 2: training YOLOv8n-cls on {dataset_path}")
    print("(downloads ~6 MB pretrained weights on first run)")
    model = YOLO("yolov8n-cls.pt")
    results = model.train(
        data=str(dataset_path),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=5,            # early stop if val acc plateaus
        verbose=True,
        plots=False,
        project=str(dataset_path / "runs"),
        name="orient_cls",
        exist_ok=True,
    )
    # Train loop already logs final top-1/top-5 val accuracy.
    return model


# ── Step 3+4 — Predict on ATRW + update cache ────────────────────────────────
def apply_to_atrw(model: YOLO):
    atrw_cache_path = os.path.join(ATRW_ROOT, "metadata_auto.json")
    img_dir         = os.path.join(ATRW_ROOT, "train")
    if not os.path.isfile(atrw_cache_path):
        raise SystemExit(
            f"{atrw_cache_path} not found — run "
            "`python scripts/extract_metadata.py --dataset atrw` first."
        )
    with open(atrw_cache_path) as f:
        atrw_cache = json.load(f)

    file_keys   = list(atrw_cache.keys())
    image_paths = [os.path.join(img_dir, fn) for fn in file_keys]

    pred_counter = Counter()
    print(f"\nStep 3: predicting orientation for {len(image_paths)} ATRW images")
    # Predict in chunks to avoid loading all results into memory at once.
    chunk = 64
    for start in tqdm(range(0, len(image_paths), chunk), desc="atrw predict"):
        batch_paths = image_paths[start:start + chunk]
        batch_keys  = file_keys[start:start + chunk]
        results = model.predict(batch_paths, imgsz=IMGSZ, verbose=False)
        for fn, res in zip(batch_keys, results):
            top1 = int(res.probs.top1)
            cls  = res.names[top1]
            atrw_cache[fn]["orientation"] = cls
            pred_counter[cls] += 1

    print(f"\n  ATRW orientation: {dict(pred_counter)}")
    if len(set(pred_counter)) < 3:
        print("  WARNING: predictions collapsed to <3 classes — transfer is weak.")

    with open(atrw_cache_path, "w") as f:
        json.dump(atrw_cache, f)
    print(f"\nUpdated {atrw_cache_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    work_dir = Path(LEOPARD_ROOT) / "_yolo_orient_dataset"
    print("Step 1: building YOLO classification dataset from leopard")
    dataset_path = build_yolo_dataset(work_dir)
    model = train_classifier(dataset_path)
    apply_to_atrw(model)

    print("\nDone. The YOLO training artifacts (runs/, weights) live under "
          f"{work_dir}/runs/orient_cls/ and can be deleted if you don't want "
          "to reuse the classifier.")


if __name__ == "__main__":
    main()
