"""Shared dataset helpers for the lab.

Breed labels are taken from the filename, not from the parent folder.
In this copy of PetFaces two folders glue together different Oxford-IIIT
breeds (dog_american, dog_english), so folder names are the wrong classes.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ROOT = Path(__file__).resolve().parent
PETFACES = ROOT / "petfaces"
OXFORD = ROOT / "images1"
SPLITS_DIR = ROOT / "splits"
CKPT_DIR = ROOT / "checkpoints"
RUNS_DIR = ROOT / "runs"

SEED = 13
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pretty_label(raw: str) -> str:
    return re.sub(r"[_]+", " ", raw).strip()


def _breed_from_name(path: Path) -> str:
    return path.stem.rsplit("_", 1)[0]


def _species_from_name(breed: str) -> str:
    cats = {
        "Abyssinian",
        "Bengal",
        "Birman",
        "Bombay",
        "British_Shorthair",
        "Egyptian_Mau",
        "Maine_Coon",
        "Persian",
        "Ragdoll",
        "Russian_Blue",
        "Siamese",
        "Sphynx",
    }
    return "cat" if breed in cats else "dog"


def scan_petfaces(root: Path = PETFACES) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        breed = _breed_from_name(p)
        species = _species_from_name(breed)
        rows.append(
            {
                "path": str(p),
                "breed": breed,
                "species": species,
                "pretty": pretty_label(breed),
                "folder": p.parent.name,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"no images under {root}")
    return df


def scan_oxford(root: Path = OXFORD) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        breed = _breed_from_name(p)
        parent = p.parent.name.lower()
        if parent in {"cats", "cat"}:
            species = "cat"
        elif parent in {"dogs", "dog"}:
            species = "dog"
        else:
            species = _species_from_name(breed)
        rows.append(
            {
                "path": str(p),
                "breed": breed,
                "species": species,
                "pretty": pretty_label(breed),
                "folder": p.parent.name,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"no images under {root}")
    return df


def add_label_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    df = df.copy()
    breeds = sorted(df["breed"].unique())
    breed2id = {b: i for i, b in enumerate(breeds)}
    species2id = {"cat": 0, "dog": 1}
    df["breed_id"] = df["breed"].map(breed2id)
    df["species_id"] = df["species"].map(species2id)
    return df, breed2id, species2id


def stratified_splits(
    df: pd.DataFrame, seed: int = SEED, val_size: float = 0.15, test_size: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hold = val_size + test_size
    train_df, rest = train_test_split(
        df, test_size=hold, random_state=seed, stratify=df["breed_id"]
    )
    rel_test = test_size / hold
    val_df, test_df = train_test_split(
        rest, test_size=rel_test, random_state=seed, stratify=rest["breed_id"]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_splits(train_df, val_df, test_df, breed2id, out_dir: Path = SPLITS_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    (out_dir / "breed2id.json").write_text(json.dumps(breed2id, indent=2, ensure_ascii=False))


def load_splits(out_dir: Path = SPLITS_DIR):
    train_df = pd.read_csv(out_dir / "train.csv")
    val_df = pd.read_csv(out_dir / "val.csv")
    test_df = pd.read_csv(out_dir / "test.csv")
    breed2id = json.loads((out_dir / "breed2id.json").read_text())
    return train_df, val_df, test_df, breed2id


def ensure_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    marker = SPLITS_DIR / "train.csv"
    if marker.exists():
        return load_splits()
    df = scan_petfaces()
    df, breed2id, _ = add_label_ids(df)
    train_df, val_df, test_df = stratified_splits(df)
    save_splits(train_df, val_df, test_df, breed2id)
    return train_df, val_df, test_df, breed2id


@dataclass
class Tfms:
    train: object
    eval: object


def make_tfms(size: int, imagenet: bool = False, erase: bool = False) -> Tfms:
    if imagenet:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    train_ops = [
        transforms.Resize(int(size * 1.15)),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=12, translate=(0.04, 0.04), scale=(0.9, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    if erase:
        train_ops.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.12)))

    eval_ops = [
        transforms.Resize(int(size * 1.15)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    return Tfms(transforms.Compose(train_ops), transforms.Compose(eval_ops))


class PetSet(Dataset):
    def __init__(self, frame: pd.DataFrame, tfm=None):
        self.frame = frame.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        if self.tfm is not None:
            img = self.tfm(img)
        return {
            "image": img,
            "breed": int(row["breed_id"]),
            "species": int(row["species_id"]),
            "path": row["path"],
        }


def collate(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "breed": torch.tensor([b["breed"] for b in batch], dtype=torch.long),
        "species": torch.tensor([b["species"] for b in batch], dtype=torch.long),
        "path": [b["path"] for b in batch],
    }


def make_loaders(train_df, val_df, test_df, tfms: Tfms, batch: int = 32, workers: int = 0):
    train_ds = PetSet(train_df, tfms.train)
    val_ds = PetSet(val_df, tfms.eval)
    test_ds = PetSet(test_df, tfms.eval)
    kw = dict(num_workers=workers, collate_fn=collate, pin_memory=False)
    return (
        DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True, **kw),
        DataLoader(val_ds, batch_size=batch, shuffle=False, **kw),
        DataLoader(test_ds, batch_size=batch, shuffle=False, **kw),
    )


def class_weights(train_df: pd.DataFrame, n_classes: int) -> torch.Tensor:
    counts = (
        train_df["breed_id"].value_counts().reindex(range(n_classes), fill_value=1).astype(float)
    )
    w = 1.0 / torch.tensor(counts.values, dtype=torch.float32)
    return w / w.mean()


def denorm(x: torch.Tensor, imagenet: bool = False) -> torch.Tensor:
    if imagenet:
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[:, None, None]
    else:
        mean = 0.5
        std = 0.5
    return (x * std + mean).clamp(0, 1)
