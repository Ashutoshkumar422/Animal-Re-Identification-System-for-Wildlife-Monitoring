# extract_orientation_pose.py — Real animal pose model for ATRW orientation.
#
# Replaces the noisy CLIP-zero-shot orientation labels in the existing
# metadata_auto.json with labels derived from a proper animal pose model
# (HRNet trained on AP-10K, 23 mammal species, 17 keypoints).
#
# How it works:
#   - Run AP-10K HRNet (via mmpose's MMPoseInferencer 'animal' preset)
#   - For each image, get keypoint confidences for left/right eye, nose, tail
#   - Derive orientation from which keypoints are visible:
#       both eyes high conf       → "front"  (face toward camera)
#       only L_eye visible        → "left"   (animal's left side toward camera)
#       only R_eye visible        → "right"
#       no face but tail visible  → "back"
#       (shoulder fallback for ambiguous cases)
#
# Setup (first-time only):
#   pip install -U openmim
#   mim install "mmengine>=0.7.0"
#   mim install "mmcv>=2.0.0,<2.2.0"
#   mim install "mmdet>=3.1.0"
#   mim install "mmpose>=1.1.0"
#
# First run also downloads ~200 MB of model checkpoints to ~/.cache/torch/.
#
# Run order — this OVERLAYS orientation on top of an existing cache, so run
# extract_metadata.py first to get circadian/temperature:
#   python scripts/extract_metadata.py        --dataset atrw      # circadian + temp
#   python scripts/extract_orientation_pose.py --dataset atrw      # real orientation
#
# Also runs on leopard for sanity-checking against the COCO viewpoint labels:
#   python scripts/extract_orientation_pose.py --dataset leopard

# Allow running this file directly (python scripts/<name>.py) from the repo root.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import json
import argparse
from collections import Counter

import numpy as np
from tqdm import tqdm

try:
    from mmpose.apis import MMPoseInferencer
except ImportError as e:
    raise SystemExit(
        "mmpose not installed.\n"
        "Install via:\n"
        "  pip install -U openmim\n"
        '  mim install "mmengine>=0.7.0"\n'
        '  mim install "mmcv>=2.0.0,<2.2.0"\n'
        '  mim install "mmdet>=3.1.0"\n'
        '  mim install "mmpose>=1.1.0"\n'
    )

from config import ATRW_ROOT, LEOPARD_ROOT


# ── AP-10K keypoint indices ──────────────────────────────────────────────────
# Order: [L_Eye, R_Eye, Nose, Neck, Root_of_tail,
#         L_Shoulder, L_Elbow, L_F_Paw,
#         R_Shoulder, R_Elbow, R_F_Paw,
#         L_Hip, L_Knee, L_B_Paw,
#         R_Hip, R_Knee, R_B_Paw]
L_EYE, R_EYE, NOSE, NECK, TAIL = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER         = 5, 8
L_HIP,      R_HIP              = 11, 14

CONF_THRESH = 0.30   # AP-10K-typical confidence threshold for "keypoint visible"


def derive_orientation(scores: np.ndarray) -> str:
    """
    scores : (17,) keypoint confidences in [0, 1]
    Returns 'front' | 'back' | 'left' | 'right'

    Convention: "left" means the ANIMAL's left side is facing the camera, so we
    see its left eye/shoulder/hip (anatomical left, not image-left).
    """
    L_eye = scores[L_EYE] > CONF_THRESH
    R_eye = scores[R_EYE] > CONF_THRESH
    nose  = scores[NOSE]  > CONF_THRESH
    tail  = scores[TAIL]  > CONF_THRESH

    # Primary: eye visibility
    if L_eye and R_eye:
        return "front"
    if L_eye and not R_eye:
        return "left"
    if R_eye and not L_eye:
        return "right"

    # Secondary: tail visible without face → back
    if tail and not nose:
        return "back"

    # Tertiary: shoulder/hip asymmetry
    L_body_score = float(scores[L_SHOULDER] + scores[L_HIP])
    R_body_score = float(scores[R_SHOULDER] + scores[R_HIP])
    if L_body_score > R_body_score + 0.3:
        return "left"
    if R_body_score > L_body_score + 0.3:
        return "right"
    if L_body_score > 0.6 and R_body_score > 0.6:
        return "front"

    # Last resort
    return "back"


def get_dataset_items(dataset: str):
    """Returns (cache_path, cache_dict, items=[(key, full_image_path), ...])."""
    if dataset == "atrw":
        cache_path = os.path.join(ATRW_ROOT, "metadata_auto.json")
        img_dir    = os.path.join(ATRW_ROOT, "train")
        with open(cache_path) as f:
            cache = json.load(f)
        items = [(fn, os.path.join(img_dir, fn)) for fn in cache.keys()]
    elif dataset == "leopard":
        cache_path = os.path.join(LEOPARD_ROOT, "metadata_auto.json")
        img_dir    = os.path.join(LEOPARD_ROOT, "images")
        with open(cache_path) as f:
            cache = json.load(f)
        items = [(rel, os.path.join(img_dir, rel)) for rel in cache.keys()]
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    return cache_path, cache, items


def main(dataset: str):
    cache_path, cache, items = get_dataset_items(dataset)
    print(f"Loaded cache: {len(cache)} entries from {cache_path}")
    print(f"Running AP-10K pose on {len(items)} images...")
    print("(first run downloads ~200 MB of model weights)")

    inferencer = MMPoseInferencer('animal')

    new_orient = {}
    n_skip = 0

    for key, img_path in tqdm(items, desc=f"ap10k ({dataset})"):
        try:
            result_gen = inferencer(img_path, show=False, return_vis=False)
            result    = next(result_gen)
            preds     = result.get("predictions", [[]])[0]
            if not preds:
                n_skip += 1
                continue
            # Pick the detection with the highest mean keypoint confidence
            # (in case the detector finds multiple animals).
            best = max(preds, key=lambda p: float(np.mean(p.get("keypoint_scores", [0]))))
            scores = np.asarray(best.get("keypoint_scores", []), dtype=np.float32)
            if scores.shape[0] < 17:
                n_skip += 1
                continue
            new_orient[key] = derive_orientation(scores)
        except Exception as e:
            print(f"  skip {key}: {e}")
            n_skip += 1

    print(f"\nProcessed:   {len(new_orient)}/{len(items)} (skipped {n_skip})")
    print(f"orientation: {Counter(new_orient.values())}")

    # Sanity check: for leopard, compare against the existing COCO viewpoint labels.
    if dataset == "leopard":
        agree, total = 0, 0
        confusion = Counter()
        for key, ap10k_orient in new_orient.items():
            coco_orient = (cache[key].get("viewpoint") or "").strip().lower()
            if coco_orient and coco_orient in ("front", "back", "left", "right"):
                total += 1
                confusion[(coco_orient, ap10k_orient)] += 1
                if coco_orient == ap10k_orient:
                    agree += 1
        if total > 0:
            print(f"\n[leopard sanity check] AP-10K agrees with COCO viewpoint on "
                  f"{agree}/{total} ({100*agree/total:.1f}%)")
            print("Confusion (rows=COCO truth, cols=AP-10K prediction):")
            classes = ["front", "back", "left", "right"]
            print(f"            {'front':>8} {'back':>8} {'left':>8} {'right':>8}")
            for r in classes:
                row = f"  {r:<8}"
                for c in classes:
                    row += f" {confusion.get((r,c), 0):>8}"
                print(row)

    # Overlay onto cache (preserves circadian/temperature/etc.)
    for key, orient in new_orient.items():
        if key in cache:
            cache[key]["orientation"] = orient

    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"\nUpdated {cache_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["atrw", "leopard"], required=True)
    args = parser.parse_args()
    main(args.dataset)
