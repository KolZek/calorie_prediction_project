import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_dataframe(data_root: str) -> pd.DataFrame:
    """Загружает dish.csv и ingredients.csv, расшифровывает ID в названия."""

    dish = pd.read_csv(os.path.join(data_root, "dish.csv"))
    ingr = pd.read_csv(os.path.join(data_root, "ingredients.csv"))
    ingr_map = {f"ingr_{int(i):010d}": name for i, name in zip(ingr["id"], ingr["ingr"])}

    def parse(s):
        if not isinstance(s, str) or not s.strip():
            return []

        ids = [x.strip() for x in s.split(";") if x.strip()]
        return [ingr_map.get(x, x) for x in ids]

    dish["ingredients_list"] = dish["ingredients"].apply(parse)
    dish["ingredients_text"] = dish["ingredients_list"].apply(lambda x: ", ".join(x))

    return dish


def get_transforms(img_size: int, train: bool):

    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class DishDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: str, tokenizer, transform, max_len: int = 128):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, "images", str(row["dish_id"]), "rgb.png")
        image = np.array(Image.open(img_path).convert("RGB"))
        image = self.transform(image=image)["image"]

        tokens = self.tokenizer(
            row["ingredients_text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "target": torch.tensor(row["total_calories"], dtype=torch.float32),
            "dish_id": row["dish_id"],
        }


def build_dataloaders(df, cfg, tokenizer):
    """Возвращает (train, val, test) loaders."""

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    val_idx = train_df.sample(frac=cfg.VAL_FRAC, random_state=cfg.SEED).index
    val_df = train_df.loc[val_idx].reset_index(drop=True)
    train_df = train_df.drop(val_idx).reset_index(drop=True)

    train_ds = DishDataset(train_df, cfg.DATA_ROOT, tokenizer, get_transforms(cfg.IMG_SIZE, True), cfg.MAX_LEN)
    val_ds = DishDataset(val_df, cfg.DATA_ROOT, tokenizer, get_transforms(cfg.IMG_SIZE, False), cfg.MAX_LEN)
    test_ds = DishDataset(test_df, cfg.DATA_ROOT, tokenizer, get_transforms(cfg.IMG_SIZE, False), cfg.MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    print(f"Размеры: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    
    return train_loader, val_loader, test_loader
