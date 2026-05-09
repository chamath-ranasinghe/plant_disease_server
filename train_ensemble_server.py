"""
train_ensemble_linux.py
=======================
Full Deep Ensemble training pipeline for plant disease uncertainty estimation.
Linux-compatible, self-contained — downloads all data via Kaggle automatically.

Usage:
    # First time — downloads data and trains all 5 members:
    python train_ensemble_linux.py

    # If training was interrupted — resumes from last checkpoint:
    python train_ensemble_linux.py

Outputs (saved to checkpoints/):
    ensemble_member_0.pth ... ensemble_member_4.pth
    ensemble_outputs.npz   (upload this to Colab for Day 5/6)
"""

import os
import re
import sys
import json
import math
import time
import random
import shutil
import zipfile
import subprocess
import warnings
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.stats import entropy as scipy_entropy
from scipy.special import softmax as scipy_softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ══════════════════════════════════════════════════════════════════
# 1.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
BASE_DIR         = Path(__file__).parent
PLANTVILLAGE_DIR = BASE_DIR / "plantvillage_raw"
PLANTDOC_DIR     = BASE_DIR / "plantdoc"
CHECKPOINT_DIR   = BASE_DIR / "checkpoints"
LOG_DIR          = BASE_DIR / "logs"

for d in [PLANTVILLAGE_DIR, PLANTDOC_DIR, CHECKPOINT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

NUM_CLASSES    = 38
ENSEMBLE_SIZE  = 5
ENSEMBLE_SEEDS = [42, 7, 123, 2024, 99]
EPOCHS         = 20
BATCH_SIZE     = 32
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
SPLIT_SEED     = 42
BUCKET_SIZE    = 10

# Linux can use multiple workers safely
NUM_WORKERS = min(4, os.cpu_count() or 1)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ══════════════════════════════════════════════════════════════════
# 2.  DEVICE CHECK
# ══════════════════════════════════════════════════════════════════
def check_device():
    print("\n" + "="*60)
    print("  DEVICE CHECK")
    print("="*60)
    print(f"  PyTorch version : {torch.__version__}")
    print(f"  CUDA available  : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  GPU             : {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  VRAM            : {props.total_memory / 1e9:.1f} GB")
        print(f"  CUDA version    : {torch.version.cuda}")
    else:
        print("  *** Running on CPU — training will be very slow ***")
        print("  If you have an NVIDIA GPU reinstall PyTorch with:")
        print("  pip install torch torchvision "
              "--index-url https://download.pytorch.org/whl/cu121")

    print(f"  Active device   : {device}")
    print(f"  DataLoader workers: {NUM_WORKERS}")
    print("="*60 + "\n")


# ══════════════════════════════════════════════════════════════════
# 3.  DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════
PLANTDOC_URL = (
    "https://github.com/pratikkayal/PlantDoc-Dataset/archive/"
    "refs/heads/master.zip"
)


def verify_kaggle_credentials():
    """Checks kaggle.json exists and is readable before downloading."""
    kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"

    if not kaggle_creds.exists():
        print("  ✗ kaggle.json not found at:")
        print(f"    {kaggle_creds}")
        print("\n  Setup instructions:")
        print("  1. Go to https://www.kaggle.com → Account")
        print("  2. Click 'Create New Token' → downloads kaggle.json")
        print(f"  3. Run:")
        print(f"       mkdir -p ~/.kaggle")
        print(f"       cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json")
        print(f"       chmod 600 ~/.kaggle/kaggle.json")
        print("  4. Re-run this script.")
        sys.exit(1)

    # Kaggle requires 600 permissions on Linux
    current_mode = oct(kaggle_creds.stat().st_mode)[-3:]
    if current_mode != '600':
        print(f"  Fixing kaggle.json permissions "
              f"({current_mode} → 600)...")
        kaggle_creds.chmod(0o600)

    print("  ✓ Kaggle credentials found.")


def download_plantvillage():
    """
    Downloads PlantVillage via tensorflow-datasets — the exact same
    source used in Colab for baseline and MC Dropout training.
    Guarantees identical label ordering and image content.
    """
    existing_dirs = (
        [d for d in PLANTVILLAGE_DIR.iterdir() if d.is_dir()]
        if PLANTVILLAGE_DIR.exists() else []
    )
    if len(existing_dirs) >= 38:
        total_imgs = sum(
            len(list(d.glob('*.jpg'))) for d in existing_dirs
        )
        print(f"  PlantVillage already downloaded: "
              f"{total_imgs:,} images in "
              f"{len(existing_dirs)} classes.")
        return

    print("\n" + "="*60)
    print("  DOWNLOADING PLANTVILLAGE (via tensorflow-datasets)")
    print("  This guarantees the same label ordering as Colab.")
    print("="*60)

    import tensorflow_datasets as tfds

    print("  Loading dataset (downloads ~300MB on first run)...")
    ds_raw, info = tfds.load(
        'plant_village',
        split='train',
        with_info=True,
        as_supervised=True,
        shuffle_files=False,   # critical — keep order stable
    )

    label_names = info.features['label'].names
    n_classes   = info.features['label'].num_classes
    print(f"  Classes: {n_classes}")
    print(f"  First 5 labels: {label_names[:5]}")

    # Save label map — same format as before so rest of
    # script works unchanged
    label_map = {name: idx for idx, name in enumerate(label_names)}
    label_map_path = PLANTVILLAGE_DIR / "label_map.json"
    with open(label_map_path, 'w') as f:
        json.dump(label_map, f, indent=2)

    print("  Writing images to disk...")
    counts = defaultdict(int)

    for i, (img_tensor, label_tensor) in enumerate(
            tqdm(ds_raw, desc="  Saving", unit="img")):
        label    = int(label_tensor.numpy())
        dest_dir = PLANTVILLAGE_DIR / str(label)
        dest_dir.mkdir(exist_ok=True)

        img_path = dest_dir / f"{i:05d}.jpg"
        if not img_path.exists():
            img = Image.fromarray(img_tensor.numpy()).convert('RGB')
            img.save(img_path, 'JPEG', quality=95)

        counts[label] += 1

    total = sum(counts.values())
    print(f"  Saved {total:,} images across "
          f"{len(counts)} classes.")
    print("  ✓ Label ordering matches Colab exactly.")
    print("  PlantVillage ready.\n")


def download_plantdoc():
    """Downloads PlantDoc from GitHub."""
    if (PLANTDOC_DIR / "train").exists():
        n = len(list((PLANTDOC_DIR / "train").rglob("*.jpg")))
        print(f"  PlantDoc already downloaded: {n:,} train images.")
        return

    print("\n" + "="*60)
    print("  DOWNLOADING PLANTDOC (via GitHub)")
    print("="*60)

    zip_path    = BASE_DIR / "plantdoc.zip"
    extract_dir = BASE_DIR / "plantdoc_extracted"

    if not zip_path.exists():
        print(f"  Downloading PlantDoc (~500 MB)...")
        result = subprocess.run(
            ["wget", "-q", "--show-progress",
             "-O", str(zip_path), PLANTDOC_URL],
            check=False,
        )
        # Fall back to requests if wget is unavailable
        if result.returncode != 0 or not zip_path.exists():
            print("  wget failed — falling back to requests...")
            import requests
            response = requests.get(PLANTDOC_URL, stream=True,
                                    timeout=60)
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(
                total=total, unit='B',
                unit_scale=True, desc="  plantdoc.zip"
            ) as bar:
                for chunk in response.iter_content(8192):
                    f.write(chunk)
                    bar.update(len(chunk))

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            n_files = len(z.namelist())
        print(f"  Valid zip: {n_files:,} files inside.")
    except zipfile.BadZipFile:
        print("  ✗ PlantDoc zip is corrupt — deleting and re-run.")
        zip_path.unlink(missing_ok=True)
        sys.exit(1)

    extract_dir.mkdir(exist_ok=True)
    _extract_zip(zip_path, extract_dir)

    # GitHub extracts as PlantDoc-Dataset-master/
    extracted_root = extract_dir / "PlantDoc-Dataset-master"
    if not extracted_root.exists():
        candidates = [d for d in extract_dir.iterdir() if d.is_dir()]
        if candidates:
            extracted_root = candidates[0]

    shutil.copytree(str(extracted_root), str(PLANTDOC_DIR),
                    dirs_exist_ok=True)
    shutil.rmtree(extract_dir, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    print("  PlantDoc ready.\n")


def _extract_zip(zip_path, extract_to):
    print(f"  Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        members = z.namelist()
        for member in tqdm(members, desc="  Extracting",
                           unit="file"):
            z.extract(member, extract_to)
    print(f"  Extracted {len(members):,} files.")


# ══════════════════════════════════════════════════════════════════
# 4.  DATASET BUILDING
# ══════════════════════════════════════════════════════════════════
def build_plantvillage_index():
    """
    Builds (path_str, label_int) index.
    Sorted explicitly by folder name then filename for
    cross-platform reproducibility.
    """
    index = []
    label_dirs = sorted(
        [d for d in PLANTVILLAGE_DIR.iterdir() if d.is_dir()],
        key=lambda p: int(p.name)
    )
    for label_dir in label_dirs:
        label = int(label_dir.name)
        for img_path in sorted(label_dir.glob('*.jpg'),
                                key=lambda p: p.name):
            index.append((str(img_path), label))

    n_classes = len(set(l for _, l in index))
    print(f"  Index: {len(index):,} images, {n_classes} classes")
    return index


def instance_level_split(index):
    """
    Reproduces the exact 80/10/10 instance-level split from Colab.
    Uses SPLIT_SEED=42 and BUCKET_SIZE=10 — do not change these
    or your splits will differ from the Colab baseline/MC models.
    """
    random.seed(SPLIT_SEED)

    groups = defaultdict(list)
    for idx, (_, label) in enumerate(index):
        bucket_id = idx // BUCKET_SIZE
        groups[(label, bucket_id)].append(idx)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    n          = len(group_keys)
    train_keys = group_keys[:int(0.80 * n)]
    val_keys   = group_keys[int(0.80 * n):int(0.90 * n)]
    test_keys  = group_keys[int(0.90 * n):]

    train_samples = [index[i] for k in train_keys for i in groups[k]]
    val_samples   = [index[i] for k in val_keys   for i in groups[k]]
    test_samples  = [index[i] for k in test_keys  for i in groups[k]]

    # Verify no leakage
    train_paths = {p for p, _ in train_samples}
    val_paths   = {p for p, _ in val_samples}
    test_paths  = {p for p, _ in test_samples}
    assert not train_paths & val_paths,  "Leakage: train/val"
    assert not train_paths & test_paths, "Leakage: train/test"
    assert not val_paths   & test_paths, "Leakage: val/test"

    print(f"  Split — "
          f"train: {len(train_samples):,} | "
          f"val: {len(val_samples):,} | "
          f"test: {len(test_samples):,}")
    print("  ✓ No data leakage.")
    return train_samples, val_samples, test_samples


def build_plantdoc_index():
    """
    Builds PlantDoc OOD test index mapped to PlantVillage labels.
    """
    label_map_path = PLANTVILLAGE_DIR / "label_map.json"
    if not label_map_path.exists():
        print("  Warning: label_map.json missing — "
              "skipping PlantDoc.")
        return []

    with open(label_map_path) as f:
        pv_label_map = json.load(f)

    def normalise(s):
        return set(re.sub(r'[^a-z0-9]', ' ', s.lower()).split())

    pv_lookup = {
        frozenset(normalise(name)): idx
        for name, idx in pv_label_map.items()
    }

    test_dir = PLANTDOC_DIR / "test"
    if not test_dir.exists():
        print("  Warning: PlantDoc test dir missing.")
        return []

    index     = []
    matched   = 0
    unmatched = []

    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        pd_tokens = frozenset(normalise(class_dir.name))

        best_label, best_overlap = None, 0
        for pv_tokens, pv_idx in pv_lookup.items():
            overlap = len(pd_tokens & pv_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_label   = pv_idx

        if best_label is not None and best_overlap >= 2:
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                for img_path in class_dir.glob(ext):
                    index.append((str(img_path), best_label))
            matched += 1
        else:
            unmatched.append(class_dir.name)

    print(f"  PlantDoc: {len(index):,} images, "
          f"{matched} matched classes")
    if unmatched:
        print(f"  Unmatched (skipped): {unmatched}")
    return index


# ══════════════════════════════════════════════════════════════════
# 5.  PYTORCH DATASET AND TRANSFORMS
# ══════════════════════════════════════════════════════════════════
class LeafDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples   = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


train_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def make_loader(samples, transform, shuffle, seed=None):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        LeafDataset(samples, transform=transform),
        batch_size         = BATCH_SIZE,
        shuffle            = shuffle,
        num_workers        = NUM_WORKERS,
        pin_memory         = (device.type == 'cuda'),
        generator          = generator,
        persistent_workers = (NUM_WORKERS > 0),  # safe on Linux
        prefetch_factor    = 2 if NUM_WORKERS > 0 else None,
    )


# ══════════════════════════════════════════════════════════════════
# 6.  SANITY CHECKS
# ══════════════════════════════════════════════════════════════════
def run_sanity_checks(train_samples, val_samples, train_loader):
    print("\n" + "="*60)
    print("  SANITY CHECKS")
    print("="*60)

    train_labels = [l for _, l in train_samples]
    val_labels   = [l for _, l in val_samples]
    train_unique = len(set(train_labels))
    val_unique   = len(set(val_labels))

    # Get actual class count from the data rather than hardcoding 38
    actual_classes = len(set(train_labels) | set(val_labels))
    print(f"  Total unique classes: {actual_classes}")
    print(f"  Train unique labels : {train_unique}")
    print(f"  Val   unique labels : {val_unique}")

    # Update NUM_CLASSES globally to match actual data
    global NUM_CLASSES
    if actual_classes != NUM_CLASSES:
        print(f"  Updating NUM_CLASSES: {NUM_CLASSES} → "
              f"{actual_classes}")
        NUM_CLASSES = actual_classes

    if train_unique < NUM_CLASSES * 0.9:
        print(f"  ✗ Too few classes in training set "
              f"({train_unique}/{NUM_CLASSES})")
        print("    Check plantvillage_raw/ folder contents.")
        sys.exit(1)

    # Rest of checks unchanged from before
    expected_loss = math.log(NUM_CLASSES)
    probe         = make_resnet18(NUM_CLASSES, seed=0)
    probe.eval()
    imgs, labels  = next(iter(train_loader))
    with torch.no_grad():
        logits = probe(imgs.to(device))
        loss   = nn.CrossEntropyLoss()(logits, labels.to(device))
    print(f"  Initial loss        : {loss.item():.4f}  "
          f"(expect ~{expected_loss:.4f} = log({NUM_CLASSES}))")
    del probe

    if not (0.5 * expected_loss < loss.item() < 2 * expected_loss):
        print("  ✗ Initial loss is abnormal.")
        sys.exit(1)

    print(f"  Pixel range (normed): "
          f"{imgs.min():.2f} to {imgs.max():.2f}")
    print(f"  Label range in batch: "
          f"{labels.min().item()} – {labels.max().item()}")
    print("  ✓ All checks passed.")
    print("="*60 + "\n")


# ══════════════════════════════════════════════════════════════════
# 7.  MODEL
# ══════════════════════════════════════════════════════════════════
def make_resnet18(num_classes, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    base    = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )
    base.fc = nn.Linear(base.fc.in_features, num_classes)
    return base.to(device)


# ══════════════════════════════════════════════════════════════════
# 8.  LOGGING
# ══════════════════════════════════════════════════════════════════
class TrainingLogger:
    def __init__(self, member_id):
        self.path = LOG_DIR / f"member_{member_id}_log.csv"
        with open(self.path, 'w') as f:
            f.write("epoch,train_loss,train_acc,val_acc,lr\n")

    def log(self, epoch, train_loss, train_acc, val_acc, lr):
        with open(self.path, 'a') as f:
            f.write(f"{epoch},"
                    f"{train_loss:.6f},"
                    f"{train_acc:.6f},"
                    f"{val_acc:.6f},"
                    f"{lr:.8f}\n")


# ══════════════════════════════════════════════════════════════════
# 9.  TRAIN ONE MEMBER
# ══════════════════════════════════════════════════════════════════
def train_one_member(member_id, seed, train_samples, val_samples):
    print(f"\n{'='*60}")
    print(f"  Member {member_id+1}/{ENSEMBLE_SIZE}  (seed={seed})")
    print(f"{'='*60}")

    ckpt_path = CHECKPOINT_DIR / f"ensemble_member_{member_id}.pth"

    if ckpt_path.exists():
        saved = torch.load(ckpt_path, map_location='cpu')
        print(f"  Checkpoint found — "
              f"epoch {saved['epoch']}, "
              f"val acc {saved['val_acc']:.4f}")
        print(f"  Skipping. Delete {ckpt_path.name} to retrain.")
        return str(ckpt_path)

    train_loader = make_loader(train_samples, train_transforms,
                               shuffle=True, seed=seed)
    val_loader   = make_loader(val_samples,   eval_transforms,
                               shuffle=False)

    member    = make_resnet18(NUM_CLASSES, seed)
    optimizer = AdamW(member.parameters(),
                      lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer,
                                  T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler(enabled=(device.type == 'cuda'))
    logger    = TrainingLogger(member_id)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────
        member.train()
        train_loss = train_correct = train_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(
                    enabled=(device.type == 'cuda')):
                logits = member(imgs)
                loss   = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss    += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels
                              ).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc   = train_correct / train_total

        # ── Validate ──────────────────────────────────────────────
        member.eval()
        val_correct = val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(
                        enabled=(device.type == 'cuda')):
                    logits = member(imgs)
                val_correct += (logits.argmax(1) == labels
                                ).sum().item()
                val_total   += imgs.size(0)

        val_acc = val_correct / val_total
        lr      = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        # ── Checkpoint ────────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'member_id':  member_id,
                'seed':       seed,
                'epoch':      epoch,
                'state_dict': member.state_dict(),
                'val_acc':    val_acc,
            }, ckpt_path)
            saved_marker = " ← saved"
        else:
            saved_marker = ""

        logger.log(epoch, train_loss, train_acc, val_acc, lr)

        print(f"  [{member_id+1}] "
              f"Epoch {epoch:02d}/{EPOCHS} | "
              f"loss {train_loss:.4f} | "
              f"train acc {train_acc:.4f} | "
              f"val acc {val_acc:.4f} | "
              f"lr {lr:.2e} | "
              f"{elapsed:.0f}s"
              f"{saved_marker}")

    print(f"  Member {member_id+1} done. "
          f"Best val acc: {best_val_acc:.4f}")
    return str(ckpt_path)


# Need GradScaler at module level for train_one_member
from torch.cuda.amp import GradScaler


# ══════════════════════════════════════════════════════════════════
# 10. INFERENCE
# ══════════════════════════════════════════════════════════════════
def load_ensemble(checkpoint_paths):
    members = []
    for path in checkpoint_paths:
        ckpt   = torch.load(path, map_location=device)
        member = make_resnet18(NUM_CLASSES, seed=ckpt['seed'])
        member.load_state_dict(ckpt['state_dict'])
        member.eval()
        members.append(member)
        print(f"  Loaded member {ckpt['member_id']+1} | "
              f"seed {ckpt['seed']} | "
              f"epoch {ckpt['epoch']} | "
              f"val acc {ckpt['val_acc']:.4f}")
    return members


def ensemble_predict(members, loader):
    for m in members:
        m.eval()

    all_mean_probs   = []
    all_member_probs = []
    all_labels       = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)

            member_probs = torch.stack([
                F.softmax(member(imgs), dim=1)
                for member in members
            ], dim=0)                              # (M, batch, C)

            mean_probs = member_probs.mean(dim=0)  # (batch, C)

            all_mean_probs.append(mean_probs.cpu())
            all_member_probs.append(member_probs.cpu())
            all_labels.append(labels)

    mean_probs_all   = torch.cat(
        all_mean_probs, dim=0).numpy()             # (N, C)
    member_probs_all = torch.cat(
        all_member_probs, dim=1).numpy()           # (M, N, C)
    labels_all       = torch.cat(
        all_labels, dim=0).numpy()                 # (N,)

    total_entropy    = scipy_entropy(
        mean_probs_all, axis=1)

    member_entropies = np.stack([
        scipy_entropy(member_probs_all[m], axis=1)
        for m in range(len(members))
    ], axis=0)
    expected_entropy = member_entropies.mean(axis=0)
    epistemic        = total_entropy - expected_entropy
    preds            = mean_probs_all.argmax(axis=1)

    return (mean_probs_all, total_entropy,
            expected_entropy, epistemic,
            preds, labels_all)


def run_inference(members, test_samples, plantdoc_samples):
    print("\n" + "="*60)
    print("  RUNNING INFERENCE")
    print("="*60)

    test_loader = make_loader(
        test_samples, eval_transforms, shuffle=False
    )
    print("  PlantVillage test set...")
    (pv_probs, pv_total_ent, pv_expected_ent,
     pv_epistemic, pv_preds, pv_labels) = \
        ensemble_predict(members, test_loader)

    pv_acc = (pv_preds == pv_labels).mean()
    print(f"  PV accuracy         : {pv_acc:.4f}")
    print(f"  PV mean confidence  : "
          f"{pv_probs.max(axis=1).mean():.4f}")
    print(f"  PV mean epistemic   : {pv_epistemic.mean():.4f}")

    pd_results = (None,) * 6

    if plantdoc_samples:
        pd_loader = make_loader(
            plantdoc_samples, eval_transforms, shuffle=False
        )
        print("  PlantDoc (OOD)...")
        pd_results = ensemble_predict(members, pd_loader)
        (pd_probs, pd_total_ent, pd_expected_ent,
         pd_epistemic, pd_preds, pd_labels) = pd_results

        pd_acc = (pd_preds == pd_labels).mean()
        print(f"  PD accuracy         : {pd_acc:.4f}  "
              f"(expect ~0.30)")
        print(f"  PD mean confidence  : "
              f"{pd_probs.max(axis=1).mean():.4f}  "
              f"(expect >0.85)")
        print(f"  PD mean epistemic   : "
              f"{pd_epistemic.mean():.4f}  "
              f"(expect > PV)")

    return (pv_probs, pv_total_ent, pv_expected_ent,
            pv_epistemic, pv_preds, pv_labels,
            *pd_results)


# ══════════════════════════════════════════════════════════════════
# 11. SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════
def save_outputs(pv_probs, pv_total_ent, pv_expected_ent,
                 pv_epistemic, pv_preds, pv_labels,
                 pd_probs, pd_total_ent, pd_expected_ent,
                 pd_epistemic, pd_preds, pd_labels):

    out_path  = CHECKPOINT_DIR / "ensemble_outputs.npz"
    save_dict = dict(
        pv_probs        = pv_probs,
        pv_total_ent    = pv_total_ent,
        pv_expected_ent = pv_expected_ent,
        pv_epistemic    = pv_epistemic,
        pv_preds        = pv_preds,
        pv_labels       = pv_labels,
    )
    if pd_probs is not None:
        save_dict.update(dict(
            pd_probs        = pd_probs,
            pd_total_ent    = pd_total_ent,
            pd_expected_ent = pd_expected_ent,
            pd_epistemic    = pd_epistemic,
            pd_preds        = pd_preds,
            pd_labels       = pd_labels,
        ))

    np.savez(out_path, **save_dict)
    print(f"\n  Outputs saved: {out_path}")


# ══════════════════════════════════════════════════════════════════
# 12. MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    check_device()

    # ── Download ──────────────────────────────────────────────────
    download_plantvillage()
    download_plantdoc()

    # ── Build index ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("  BUILDING DATA INDEX")
    print("="*60)
    index = build_plantvillage_index()
    train_samples, val_samples, test_samples = \
        instance_level_split(index)
    plantdoc_samples = build_plantdoc_index()

    # ── Sanity checks ─────────────────────────────────────────────
    check_loader = make_loader(
        train_samples, train_transforms, shuffle=True
    )
    run_sanity_checks(train_samples, val_samples, check_loader)
    del check_loader

    # ── Train all members ─────────────────────────────────────────
    checkpoint_paths = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        ckpt = train_one_member(
            member_id     = i,
            seed          = seed,
            train_samples = train_samples,
            val_samples   = val_samples,
        )
        checkpoint_paths.append(ckpt)
        torch.cuda.empty_cache()

    # ── Inference ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  LOADING ENSEMBLE FOR INFERENCE")
    print("="*60)
    members = load_ensemble(checkpoint_paths)

    outputs = run_inference(
        members, test_samples, plantdoc_samples
    )
    save_outputs(*outputs)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ALL DONE — files to upload to Colab:")
    print("="*60)
    for i in range(ENSEMBLE_SIZE):
        p = CHECKPOINT_DIR / f"ensemble_member_{i}.pth"
        mb = p.stat().st_size / 1e6 if p.exists() else 0
        print(f"  checkpoints/ensemble_member_{i}.pth  "
              f"({mb:.0f} MB)")
    npz = CHECKPOINT_DIR / "ensemble_outputs.npz"
    if npz.exists():
        print(f"  checkpoints/ensemble_outputs.npz  "
              f"({npz.stat().st_size / 1e6:.0f} MB)")
    print("="*60)


if __name__ == '__main__':
    main()