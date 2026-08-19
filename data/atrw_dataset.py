import os, random, sys
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_SIZE

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def _parse_train_csv(csv_path):
    id_to_files = defaultdict(list)
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                identity_id = int(parts[0])
                filename    = parts[1].strip()
            except ValueError:
                continue
            id_to_files[identity_id].append(filename)

    # Remap raw IDs to 0-indexed labels (0 ... num_classes-1)
    sorted_ids = sorted(id_to_files.keys())
    id_remap   = {orig: new for new, orig in enumerate(sorted_ids)}
    remapped   = defaultdict(list)
    for orig_id, files in id_to_files.items():
        remapped[id_remap[orig_id]] = files
    return remapped

def _make_splits(id_to_files, train_ratio=0.6, gallery_ratio=0.25, seed=42):
    rng = random.Random(seed)
    train, gallery, query = [], [], []
    for identity_id, files in sorted(id_to_files.items()):
        rng.shuffle(files)
        n         = len(files)
        n_train   = max(1, int(n * train_ratio))
        n_gallery = max(1, int(n * gallery_ratio))
        for f in files[:n_train]:
            train.append((f, identity_id))
        for f in files[n_train:n_train + n_gallery]:
            gallery.append((f, identity_id))
        for f in files[n_train + n_gallery:]:
            query.append((f, identity_id))
        if not files[n_train + n_gallery:] and n_gallery > 1:
            moved = gallery.pop()
            query.append(moved)
    return train, gallery, query


class ATRWDataset(Dataset):
    def __init__(self, samples, img_dir, split="train"):
        self.transform   = TRAIN_TRANSFORM if split == "train" else TEST_TRANSFORM
        self.img_dir     = img_dir
        self.split       = split
        self.num_classes = len(set(s[1] for s in samples))

        self.samples = []
        missing = 0
        for fname, label in samples:
            path = os.path.join(img_dir, fname)
            if os.path.exists(path):
                self.samples.append((path, label))
            else:
                missing += 1
        if missing:
            print(f"[ATRW {split}] WARNING: {missing} missing images skipped")
        print(f"[ATRW {split}] {len(self.samples)} images | "
              f"{self.num_classes} identities")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        prompt = (
            f"A photo of a tiger individual {label} in cool temperature, "
            f"with face direction front, captured during the day."
        )
        return img, prompt, label, "tiger"


def load_atrw(root="./data/atrw", transform_train=None, transform_test=None):
    csv_path = os.path.join(root, "reid_list_train.csv")
    img_dir  = os.path.join(root, "train")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Cannot find: {csv_path}")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Cannot find image dir: {img_dir}")

    print(f"[ATRW] CSV:       {csv_path}")
    print(f"[ATRW] Image dir: {img_dir}")

    id_to_files = _parse_train_csv(csv_path)
    print(f"[ATRW] Identities: {len(id_to_files)} | "
          f"Images: {sum(len(v) for v in id_to_files.values())}")

    train_s, gallery_s, query_s = _make_splits(id_to_files)

    train_ds   = ATRWDataset(train_s,   img_dir, split="train")
    gallery_ds = ATRWDataset(gallery_s, img_dir, split="gallery")
    query_ds   = ATRWDataset(query_s,   img_dir, split="query")

    return train_ds, gallery_ds, query_ds