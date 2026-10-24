import os
import random
import numpy as np
import pandas as pd
from functools import partial

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

import torchmetrics
import timm
from transformers import AutoModel, AutoTokenizer

from dataset import load_dataframe, build_dataloaders


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MultiModalRegressor(nn.Module):
    def __init__(self, image_model: str, text_model: str, hidden_dim: int, dropout: float):
        super().__init__()

        image_factory = partial(timm.create_model, pretrained=True, num_classes=0)
        self.image_encoder = image_factory(image_model)
        image_dim = self.image_encoder.num_features

        text_factory = partial(AutoModel.from_pretrained)
        self.text_encoder = text_factory(text_model)
        text_dim = self.text_encoder.config.hidden_size

        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.image_encoder(image)
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_feat = text_out.last_hidden_state[:, 0]

        img_proj = self.image_projection(img_feat)
        text_proj = self.text_projection(text_feat)
        x = torch.cat([img_proj, text_proj], dim=1)
        return self.classifier(x).squeeze(-1)


def build_optimizer(model, cfg):
    return AdamW([
        {"params": model.text_encoder.parameters(), "lr": cfg.TEXT_LR},
        {"params": model.image_encoder.parameters(), "lr": cfg.IMAGE_LR},
        {"params": model.text_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.image_projection.parameters(), "lr": cfg.CLASSIFIER_LR},
        {"params": model.classifier.parameters(), "lr": cfg.CLASSIFIER_LR},
    ])


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    mae = torchmetrics.MeanAbsoluteError().to(device)
    total_loss, total_n = 0.0, 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(image, input_ids, attention_mask)
        loss = criterion(pred, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        total_n += target.size(0)
        mae.update(pred.detach(), target)

    return total_loss / max(total_n, 1), mae.compute().item()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    mae = torchmetrics.MeanAbsoluteError().to(device)
    total_loss, total_n = 0.0, 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        pred = model(image, input_ids, attention_mask)
        loss = criterion(pred, target)
        total_loss += loss.item() * target.size(0)
        total_n += target.size(0)
        mae.update(pred, target)

    return total_loss / max(total_n, 1), mae.compute().item()


def train(cfg):
    set_seed(cfg.SEED)
    os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    df = load_dataframe(cfg.DATA_ROOT)
    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL)
    train_loader, val_loader, _ = build_dataloaders(df, cfg, tokenizer)

    model = MultiModalRegressor(cfg.IMAGE_MODEL, cfg.TEXT_MODEL,
                                cfg.HIDDEN_DIM, cfg.DROPOUT).to(device)
    optimizer = build_optimizer(model, cfg)
    criterion = nn.L1Loss()

    best_val_mae = float("inf")
    history = []
    
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_mae": train_mae, "val_loss": val_loss, "val_mae": val_mae})
        
        print(f"[Epoch {epoch}/{cfg.EPOCHS}] train_MAE={train_mae:.2f}  val_MAE={val_mae:.2f}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "val_mae": val_mae,
                "epoch": epoch,
            }, cfg.MODEL_SAVE_PATH)

            print(f"  → сохранён чекпоинт: {cfg.MODEL_SAVE_PATH} (val_MAE={val_mae:.2f})")

    print(f"\nЛучший val_MAE: {best_val_mae:.2f}")
    return history


@torch.no_grad()
def predict(model, loader, device, clip_min: float = 0.0):
    model.eval()
    preds, targets, dish_ids = [], [], []

    for batch in loader:
        image = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(image, input_ids, attention_mask)

        if clip_min is not None:
            out = out.clamp(min=clip_min)

        preds.extend(out.cpu().numpy().tolist())
        targets.extend(batch["target"].numpy().tolist())
        dish_ids.extend(batch["dish_id"])

    return np.array(preds), np.array(targets), dish_ids


def load_model_from_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg_dict = ckpt["config"]
    model = MultiModalRegressor(cfg_dict["IMAGE_MODEL"], cfg_dict["TEXT_MODEL"], cfg_dict["HIDDEN_DIM"], cfg_dict["DROPOUT"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    return model, cfg_dict, ckpt["val_mae"]
