# config.py — All hyperparameters and paths for MetaWild implementation

import os

# Absolute path to this repository — the paths below resolve correctly no
# matter which directory a script is launched from.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Dataset ─────────────────────────────────────────────────────────────────
DATASET_NAME = "atrw"       # <── change this line only to switch datasets ("atrw" | "leopard")
ATRW_ROOT    = os.path.join(ROOT_DIR, "data", "atrw")
LEOPARD_ROOT = os.path.join(ROOT_DIR, "data", "leopard")   # see docs/DATASETS.md for expected layout
LEOPARD_IMAGE_SUBDIR = "train2022"        # subfolder under <root>/images/ holding all images
SPECIES_LIST = ["Deer", "Hare", "Penguin", "Pukeko", "Stoat", "Wallaby"]
SPECIES_CONFIGS = ["deer", "hare", "penguin", "pukeko", "stoat", "wallaby"]

# ── Leopard split protocol (mirrors Xu et al. 2026 disjoint-identity setup) ──
LEOPARD_MIN_IMGS_PER_ID  = 4       # drop identities with fewer than this many images
LEOPARD_TRAIN_ID_FRAC    = 0.70    # 70% of identities → train, 30% → test (disjoint)
LEOPARD_GALLERY_FRAC     = 0.65    # within each test identity: 65% gallery / 35% query
LEOPARD_SPLIT_SEED       = 42      # deterministic split for reproducibility

# ── Terrestrial Mammal Dataset (Xu et al. 2026 merged-protocol reproduction) ──
# Reproduces the merged ATRW + LeopardID dataset used for the journal paper's
# Tables 3-5. See data/terrestrial_dataset.py for the full protocol notes.
TM_MIN_IMGS_PER_ENTITY  = 4        # drop entities with fewer images (both species)
TM_LEOPARD_MAX_ENTITIES = 143      # subsample leopard (uuid,side) entities to this many
TM_TEST_ENTITY_FRAC     = 0.36     # fraction of merged entities held out as disjoint test
TM_VAL_IMG_FRAC         = 0.19     # within train entities, image fraction held out as val
TM_SPLIT_SEED           = 42       # deterministic split for reproducibility

# ── Partial backbone fine-tuning ──────────────────────────────────────────────
# Unfreeze the last N CLIP visual transformer blocks so the backbone adapts to
# camera-trap imagery. 0 = fully frozen (original MFA behaviour).
UNFREEZE_LAST_N_BLOCKS = 4         # ViT-B/16 has 12 visual blocks
BACKBONE_LR_MULT       = 0.1       # unfrozen blocks train at LR * this multiplier
IMAGE_SIZE     = 224
SPLIT_TRAIN    = "train"
SPLIT_GALLERY  = "gallery"
SPLIT_QUERY    = "query"

# ── Model ────────────────────────────────────────────────────────────────────
CLIP_MODEL     = "ViT-B/16"        # CLIP backbone (frozen)
EMBED_DIM      = 512               # ViT-B/16 output dimension
NUM_HEADS_ATTN = 8                 # heads in cross-attention
MLP_HIDDEN_RATIO = 4               # hidden = embed_dim // ratio

# ── Training ─────────────────────────────────────────────────────────────────
DEVICE         = "cuda"            # "cuda" or "cpu"
BATCH_SIZE     = 64
NUM_EPOCHS     = 60
LR             = 3e-4
WEIGHT_DECAY   = 1e-2
GRAD_CLIP      = 1.0
TRIPLET_MARGIN = 0.3
LABEL_SMOOTHING = 0.1

# ── Loss weights ─────────────────────────────────────────────────────────────
LAMBDA_ID      = 1.0               # identity classification loss weight
LAMBDA_TRI     = 1.0               # triplet loss weight
LAMBDA_IA      = 0.5               # cross-modal contrastive loss weight

# ── Temperature buckets (paper Section 4.1) ──────────────────────────────────
TEMP_BUCKETS = {
    "freezing": (-100,  0),
    "cold":     (  0,   8),
    "chilly":   (  8,  14),
    "cool":     ( 14,  20),
    "warm":     ( 20,  26),
    "hot":      ( 26, 100),
}

# ── Prompt template (paper Section 4.1) ──────────────────────────────────────
PROMPT_TEMPLATE = (
    "A photo of a {species} individual in {temp_cat} temperature, "
    "with face direction {face_orientation}, captured during the {circadian}."
)

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(ROOT_DIR, "results")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)
