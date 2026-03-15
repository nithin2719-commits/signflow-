"""
Train a focused 20-word ASL model for demo.
Uses existing landmark data from landmark_data_combined/.
Should take ~10-15 minutes on RTX 3070 Ti.
"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------
# 20 demo words
# ---------------------------------------------------------------
DEMO_WORDS = [
    # Greetings & basics
    "hello", "thankyou", "help", "sorry", "please", "yes", "no",
    "what_is_your_name", "i_love_you",
    # Pronouns
    "i", "hesheit",
    # Questions
    "what", "how", "when", "where",
    # Actions
    "stop", "want", "love", "need", "eat", "drink", "make",
    "see", "talk", "wake", "open", "cry", "stay", "buy", "cook",
    # Descriptions
    "sad", "hungry", "hot", "old", "sick", "thirsty", "mad",
    "cute", "better", "fine",
    # Time
    "day", "night", "yesterday", "tomorrow", "later", "time",
    # People
    "dad", "mom", "brother", "man", "boy", "grandma", "grandpa",
    # Food & things
    "water", "food", "milk", "apple", "home",
]

# Display names — clean text for UI
DISPLAY_NAMES = {
    "thankyou": "thank you",
    "what_is_your_name": "what is your name",
    "hesheit": "he/she/it",
    "i_love_you": "I love you",
}

MAX_FRAMES = 64
NUM_LANDMARKS = 92
NUM_COORDS = 3

DATA_DIR = Path(r"c:\Users\Asus\project\New Msasl\landmark_data_combined")
OUT_DIR = Path(r"c:\Users\Asus\project\New Msasl\outputs\demo_20_words")

# ---------------------------------------------------------------
# Model (identical architecture)
# ---------------------------------------------------------------
class LandmarkEmbedding(nn.Module):
    def __init__(self, in_features, units):
        super().__init__()
        self.empty_embedding = nn.Parameter(torch.zeros(units))
        self.proj = nn.Sequential(
            nn.Linear(in_features, units, bias=False), nn.GELU(),
            nn.Linear(units, units, bias=False))
    def forward(self, x):
        out = self.proj(x)
        mask = (x.abs().sum(dim=-1, keepdim=True) == 0)
        return torch.where(mask, self.empty_embedding, out)

class LandmarkTransformerEmbedding(nn.Module):
    def __init__(self, max_frames, units, lips_units, hands_units, pose_units):
        super().__init__()
        self.positional_embedding = nn.Embedding(max_frames + 1, units)
        nn.init.zeros_(self.positional_embedding.weight)
        self.lips_embedding = LandmarkEmbedding(40 * 3, lips_units)
        self.lh_embedding = LandmarkEmbedding(21 * 3, hands_units)
        self.rh_embedding = LandmarkEmbedding(21 * 3, hands_units)
        self.pose_embedding = LandmarkEmbedding(10 * 3, pose_units)
        self.landmark_weights = nn.Parameter(torch.zeros(4))
        self.fc = nn.Sequential(
            nn.Linear(max(lips_units, hands_units, pose_units), units, bias=False),
            nn.GELU(), nn.Linear(units, units, bias=False))
        self.max_frames = max_frames

    def forward(self, frames, non_empty_frame_idxs):
        x = frames
        lips = x[:,:,0:40,:].reshape(x.shape[0], x.shape[1], 40*3)
        lh = x[:,:,40:61,:].reshape(x.shape[0], x.shape[1], 21*3)
        rh = x[:,:,61:82,:].reshape(x.shape[0], x.shape[1], 21*3)
        pose = x[:,:,82:92,:].reshape(x.shape[0], x.shape[1], 10*3)
        lips_emb = self.lips_embedding(lips)
        lh_emb = self.lh_embedding(lh)
        rh_emb = self.rh_embedding(rh)
        pose_emb = self.pose_embedding(pose)
        mu = max(lips_emb.shape[-1], lh_emb.shape[-1], rh_emb.shape[-1], pose_emb.shape[-1])
        if lips_emb.shape[-1] < mu: lips_emb = F.pad(lips_emb, (0, mu - lips_emb.shape[-1]))
        if lh_emb.shape[-1] < mu: lh_emb = F.pad(lh_emb, (0, mu - lh_emb.shape[-1]))
        if rh_emb.shape[-1] < mu: rh_emb = F.pad(rh_emb, (0, mu - rh_emb.shape[-1]))
        if pose_emb.shape[-1] < mu: pose_emb = F.pad(pose_emb, (0, mu - pose_emb.shape[-1]))
        stacked = torch.stack([lips_emb, lh_emb, rh_emb, pose_emb], dim=-1)
        weights = torch.softmax(self.landmark_weights, dim=0)
        fused = (stacked * weights).sum(dim=-1)
        fused = self.fc(fused)
        max_idx = non_empty_frame_idxs.max(dim=1, keepdim=True).values.clamp(min=1)
        pos_indices = torch.where(
            non_empty_frame_idxs == -1.0,
            torch.tensor(self.max_frames, device=frames.device, dtype=torch.long),
            (non_empty_frame_idxs / max_idx * self.max_frames).long().clamp(0, self.max_frames - 1))
        return fused + self.positional_embedding(pos_indices)

class TransformerBlock(nn.Module):
    def __init__(self, units, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(units)
        self.attn = nn.MultiheadAttention(units, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(units)
        self.mlp = nn.Sequential(
            nn.Linear(units, units*mlp_ratio), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(units*mlp_ratio, units), nn.Dropout(dropout))
    def forward(self, x, key_padding_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        return x + self.mlp(self.norm2(x))

class LandmarkTransformer(nn.Module):
    def __init__(self, num_classes, max_frames=64, units=512, num_blocks=8,
                 num_heads=8, mlp_ratio=4, dropout=0.3, emb_dropout=0.1,
                 lips_units=384, hands_units=384, pose_units=256):
        super().__init__()
        self.embedding = LandmarkTransformerEmbedding(max_frames, units, lips_units, hands_units, pose_units)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([TransformerBlock(units, num_heads, mlp_ratio, dropout) for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(units)
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(units, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, frames, non_empty_frame_idxs):
        x = self.embedding(frames, non_empty_frame_idxs)
        x = self.emb_dropout(x)
        kpm = (non_empty_frame_idxs == -1.0)
        for block in self.blocks:
            x = block(x, key_padding_mask=kpm)
        x = self.norm(x)
        mask = (~kpm).unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1e-6)
        x = (x * mask).sum(dim=1) / denom
        x = self.head_dropout(x)
        return self.classifier(x)


# ---------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------
class LandmarkDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples  # list of (path, label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = np.load(path).astype(np.float32)
        if arr.shape != (MAX_FRAMES, NUM_LANDMARKS, NUM_COORDS):
            arr = np.zeros((MAX_FRAMES, NUM_LANDMARKS, NUM_COORDS), dtype=np.float32)
        non_empty = np.any(arr != 0, axis=(1, 2))
        ne_idxs = np.where(non_empty, np.arange(MAX_FRAMES, dtype=np.float32), -1.0)
        return torch.from_numpy(arr), torch.from_numpy(ne_idxs), label


# ---------------------------------------------------------------
# Training
# ---------------------------------------------------------------
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build class map
    class_map = {i: w for i, w in enumerate(DEMO_WORDS)}
    name_to_idx = {w: i for i, w in class_map.items()}
    num_classes = len(DEMO_WORDS)
    print(f"\nTraining {num_classes} classes: {', '.join(DEMO_WORDS)}")

    # Load data
    train_samples = []
    val_samples = []

    for word in DEMO_WORDS:
        label = name_to_idx[word]
        train_dir = DATA_DIR / "train" / word
        val_dir = DATA_DIR / "val" / word

        if train_dir.exists():
            for f in train_dir.glob("*.npy"):
                train_samples.append((str(f), label))
        if val_dir.exists():
            for f in val_dir.glob("*.npy"):
                val_samples.append((str(f), label))

    random.shuffle(train_samples)
    print(f"Train: {len(train_samples)} samples")
    print(f"Val:   {len(val_samples)} samples")

    if len(train_samples) == 0:
        print("ERROR: No training data found!")
        sys.exit(1)

    # Per-class counts
    train_counts = {}
    for _, lbl in train_samples:
        w = class_map[lbl]
        train_counts[w] = train_counts.get(w, 0) + 1
    for w in DEMO_WORDS:
        print(f"  {w:<20s} train={train_counts.get(w, 0):>5d}")

    # Dataloaders
    train_ds = LandmarkDataset(train_samples)
    val_ds = LandmarkDataset(val_samples)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4,
                          pin_memory=True, drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2,
                        pin_memory=True, persistent_workers=True)

    # ---- Transfer learning from 621-class model ----
    pretrained_path = Path(r"c:\Users\Asus\project\New Msasl\outputs\landmark_transformer_common_v1\best_model.pth")

    # First build model with 621 classes to load pretrained weights
    pretrained_ckpt = torch.load(str(pretrained_path), map_location=device, weights_only=False)
    pretrained_config = pretrained_ckpt.get("config", {})
    pretrained_classes = pretrained_ckpt.get("num_classes", 621)

    model = LandmarkTransformer(
        num_classes=pretrained_classes,  # 621 first
        max_frames=MAX_FRAMES,
        units=pretrained_config.get("units", 512),
        num_blocks=pretrained_config.get("num_blocks", 8),
        num_heads=pretrained_config.get("num_heads", 8),
        dropout=0.3,
    ).to(device)
    model.load_state_dict(pretrained_ckpt["model_state_dict"])
    print(f"Loaded pretrained 621-class model (79.7% val acc)")

    # Replace classifier head: 621 → num_classes (104)
    model.classifier = nn.Linear(512, num_classes).to(device)
    nn.init.trunc_normal_(model.classifier.weight, std=0.02)
    nn.init.zeros_(model.classifier.bias)
    print(f"Replaced classifier: 621 → {num_classes} classes")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {total_params:,}")

    # ---- Training setup (tuned for 95%+ val) ----
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = [p for n, p in model.named_parameters() if "classifier" in n]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 3e-5},
        {"params": head_params, "lr": 8e-4},
    ], weight_decay=0.05)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    MIXUP_ALPHA = 0.1

    # 40 epochs — proven best
    num_epochs = 40
    warmup_epochs = 3
    steps_per_epoch = len(train_dl)
    total_steps = num_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_epochs * steps_per_epoch:
            return step / (warmup_epochs * steps_per_epoch)
        progress = (step - warmup_epochs * steps_per_epoch) / (total_steps - warmup_epochs * steps_per_epoch)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Output dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    print(f"\n{'='*60}")
    print(f"Training for {num_epochs} epochs (transfer learning)")
    print(f"  Backbone LR: 3e-5 | Head LR: 8e-4")
    print(f"  Mixup: {MIXUP_ALPHA} | Label smoothing: 0.05")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        t0 = time.time()

        for frames, idxs, labels in train_dl:
            frames = frames.to(device, non_blocking=True)
            idxs = idxs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # Mixup augmentation
            do_mixup = MIXUP_ALPHA > 0 and random.random() < 0.5
            if do_mixup:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                perm = torch.randperm(frames.size(0), device=device)
                frames = lam * frames + (1 - lam) * frames[perm]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(frames, idxs)
                if do_mixup:
                    loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[perm])
                else:
                    loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_acc = train_correct / train_total
        train_loss /= train_total

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for frames, idxs, labels in val_dl:
                frames = frames.to(device, non_blocking=True)
                idxs = idxs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(frames, idxs)

                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:>2}/{num_epochs}  "
              f"loss={train_loss:.4f}  train={train_acc:.1%}  "
              f"val={val_acc:.1%}  lr={lr:.2e}  ({elapsed:.0f}s)"
              f"{'  ★ BEST' if val_acc > best_val_acc else ''}")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "best_val_acc": best_val_acc,
                "epoch": epoch + 1,
                "class_names": DEMO_WORDS,
                "config": {
                    "max_frames": MAX_FRAMES,
                    "units": 512,
                    "num_blocks": 8,
                    "num_heads": 8,
                    "dropout": 0.3,
                },
            }, str(OUT_DIR / "best_model.pth"))

            # Save class map (with display names)
            display_map = {i: DISPLAY_NAMES.get(w, w) for i, w in class_map.items()}
            with open(str(OUT_DIR / "class_map.json"), "w") as f:
                json.dump(display_map, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! Best val accuracy: {best_val_acc:.1%}")
    print(f"Model saved to: {OUT_DIR / 'best_model.pth'}")
    print(f"Class map: {OUT_DIR / 'class_map.json'}")
    print(f"{'='*60}")
    print(f"\nTo use: python sign_inference.py --model \"{OUT_DIR / 'best_model.pth'}\"")


if __name__ == "__main__":
    main()
