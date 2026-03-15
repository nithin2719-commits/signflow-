"""
Landmark-based Transformer for 1000-class ASL recognition.
Designed for >80% validation accuracy on MS-ASL + WLASL landmark data.

Key design decisions:
- Larger Transformer (4 blocks, 512 dim, 8 heads) for 1000-class capacity
- Both hands + lips + pose landmarks (92 landmarks x 3 coords)
- Heavy augmentation: temporal crop/stretch, spatial jitter, dropout, mixup
- Cosine LR with warmup, label smoothing, AdamW
- Mixed precision (AMP) for RTX 3070 Ti
"""
import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════
MAX_FRAMES = 64
NUM_LANDMARKS = 92   # 40 lip + 21 LH + 21 RH + 10 pose
NUM_COORDS = 3       # x, y, z


# ═══════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════
class LandmarkDataset(Dataset):
    def __init__(self, data_dir, split="train", max_frames=MAX_FRAMES, augment=True,
                 class_to_idx=None):
        """
        Args:
            class_to_idx: shared mapping from class name -> label index.
                          If None, builds from the union of train+val directories.
        """
        self.max_frames = max_frames
        self.augment = augment and (split == "train")
        self.samples = []   # (npy_path, label_idx)

        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        # Build a SHARED class_to_idx from ALL splits so labels are consistent
        if class_to_idx is None:
            all_classes = set()
            for s in ["train", "val"]:
                s_dir = os.path.join(data_dir, s)
                if os.path.exists(s_dir):
                    all_classes.update(
                        d for d in os.listdir(s_dir)
                        if os.path.isdir(os.path.join(s_dir, d))
                    )
            self.class_names = sorted(all_classes)
            self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        else:
            self.class_to_idx = class_to_idx
            self.class_names = sorted(class_to_idx, key=class_to_idx.get)

        self.num_classes = len(self.class_names)

        # Load samples for THIS split only
        split_classes = sorted([d for d in os.listdir(split_dir)
                                if os.path.isdir(os.path.join(split_dir, d))])
        for cls_name in split_classes:
            if cls_name not in self.class_to_idx:
                continue
            cls_dir = os.path.join(split_dir, cls_name)
            label_idx = self.class_to_idx[cls_name]
            for f in os.listdir(cls_dir):
                if f.endswith(".npy"):
                    self.samples.append((os.path.join(cls_dir, f), label_idx))

        # Compute class weights for balanced sampling
        label_counts = {}
        for _, label in self.samples:
            label_counts[label] = label_counts.get(label, 0) + 1
        self.label_counts = label_counts

        # Sample weights for WeightedRandomSampler
        total = len(self.samples)
        self.sample_weights = []
        for _, label in self.samples:
            w = total / (self.num_classes * label_counts[label])
            self.sample_weights.append(w)

    def __len__(self):
        return len(self.samples)

    def _augment(self, landmarks):
        """Apply data augmentation to landmark sequence [T, V, C]."""
        T, V, C = landmarks.shape

        # 1. Temporal augmentation: random crop and speed perturbation
        non_empty = np.any(landmarks != 0, axis=(1, 2))
        valid_frames = np.where(non_empty)[0]
        if len(valid_frames) >= 8:
            start = valid_frames[0]
            end = valid_frames[-1] + 1
            # Random temporal crop (keep 70-100% of valid frames)
            crop_ratio = random.uniform(0.7, 1.0)
            crop_len = max(8, int((end - start) * crop_ratio))
            max_start = end - crop_len
            crop_start = random.randint(start, max(start, max_start))
            landmarks = landmarks[crop_start:crop_start + crop_len]

            # Random speed perturbation via interpolation
            if random.random() < 0.5:
                speed = random.uniform(0.8, 1.2)
                new_len = max(8, int(landmarks.shape[0] * speed))
                new_len = min(new_len, self.max_frames)
                indices = np.linspace(0, landmarks.shape[0] - 1, new_len)
                int_indices = indices.astype(int)
                frac = indices - int_indices
                frac = frac[:, None, None]
                int_indices = np.clip(int_indices, 0, landmarks.shape[0] - 2)
                next_indices = np.clip(int_indices + 1, 0, landmarks.shape[0] - 1)
                landmarks = landmarks[int_indices] * (1 - frac) + landmarks[next_indices] * frac
                landmarks = landmarks.astype(np.float32)

        # Pad/truncate to max_frames
        landmarks = self._pad_truncate(landmarks)

        # 2. Spatial augmentation
        # Random horizontal flip (mirror x-coordinates, swap hands)
        if random.random() < 0.5:
            landmarks[:, :, 0] = 1.0 - landmarks[:, :, 0]
            # Swap left and right hands (indices 40-60 and 61-81)
            lh = landmarks[:, 40:61, :].copy()
            rh = landmarks[:, 61:82, :].copy()
            landmarks[:, 40:61, :] = rh
            landmarks[:, 61:82, :] = lh

        # Random spatial jitter
        if random.random() < 0.7:
            noise = np.random.normal(0, 0.005, landmarks.shape).astype(np.float32)
            # Only add noise to non-zero landmarks
            mask = (landmarks != 0).astype(np.float32)
            landmarks = landmarks + noise * mask

        # Random scale
        if random.random() < 0.5:
            scale = random.uniform(0.9, 1.1)
            nonzero_x = landmarks[:, :, 0][landmarks[:, :, 0] != 0]
            nonzero_y = landmarks[:, :, 1][landmarks[:, :, 1] != 0]
            center_x = float(np.mean(nonzero_x)) if len(nonzero_x) > 0 else 0.5
            center_y = float(np.mean(nonzero_y)) if len(nonzero_y) > 0 else 0.5
            mask = (landmarks != 0).astype(np.float32)
            landmarks[:, :, 0] = (landmarks[:, :, 0] - center_x) * scale + center_x
            landmarks[:, :, 1] = (landmarks[:, :, 1] - center_y) * scale + center_y
            landmarks = landmarks * mask

        # Random rotation (small angle)
        if random.random() < 0.3:
            angle = random.uniform(-0.1, 0.1)  # radians
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            x = landmarks[:, :, 0].copy()
            y = landmarks[:, :, 1].copy()
            mask = ((x != 0) | (y != 0)).astype(np.float32)
            nonzero_x = x[x != 0]
            nonzero_y = y[y != 0]
            cx = float(np.mean(nonzero_x)) if len(nonzero_x) > 0 else 0.5
            cy = float(np.mean(nonzero_y)) if len(nonzero_y) > 0 else 0.5
            landmarks[:, :, 0] = ((x - cx) * cos_a - (y - cy) * sin_a + cx) * mask
            landmarks[:, :, 1] = ((x - cx) * sin_a + (y - cy) * cos_a + cy) * mask

        # Random frame dropout
        if random.random() < 0.3:
            num_drop = random.randint(1, max(1, self.max_frames // 10))
            drop_indices = random.sample(range(self.max_frames), num_drop)
            landmarks[drop_indices] = 0.0

        # Random body part dropout (webcam robustness)
        # Reduced lips dropout: lips carry critical sign info, 50% was too aggressive
        non_empty_mask = np.any(landmarks != 0, axis=(1, 2))
        if non_empty_mask.any():
            drop_lips = random.random() < 0.15
            drop_lh = random.random() < 0.05
            drop_rh = random.random() < 0.05
            drop_pose = random.random() < 0.08
            # Never drop everything - always keep at least hands or pose
            if drop_lips and drop_lh and drop_rh and drop_pose:
                drop_pose = False  # keep pose as fallback
            ne_idx = np.where(non_empty_mask)[0]
            if drop_lips:
                landmarks[np.ix_(ne_idx, range(0, 40))] = 0.0
            if drop_lh:
                landmarks[np.ix_(ne_idx, range(40, 61))] = 0.0
            if drop_rh:
                landmarks[np.ix_(ne_idx, range(61, 82))] = 0.0
            if drop_pose:
                landmarks[np.ix_(ne_idx, range(82, 92))] = 0.0

        return landmarks

    def _pad_truncate(self, landmarks):
        T = landmarks.shape[0]
        if T >= self.max_frames:
            return landmarks[:self.max_frames]
        pad = np.zeros((self.max_frames - T, landmarks.shape[1], landmarks.shape[2]), dtype=np.float32)
        return np.concatenate([landmarks, pad], axis=0)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        landmarks = np.load(npy_path).astype(np.float32)  # [T, 92, 3]

        # Ensure correct landmark count
        if landmarks.shape[1] != NUM_LANDMARKS:
            # Pad or truncate landmark dimension
            if landmarks.shape[1] < NUM_LANDMARKS:
                pad = np.zeros((landmarks.shape[0], NUM_LANDMARKS - landmarks.shape[1], landmarks.shape[2]), dtype=np.float32)
                landmarks = np.concatenate([landmarks, pad], axis=1)
            else:
                landmarks = landmarks[:, :NUM_LANDMARKS, :]

        if self.augment:
            landmarks = self._augment(landmarks)
        else:
            landmarks = self._pad_truncate(landmarks)

        # Safety: if augmentation produced an all-empty sequence, reload without augmentation
        if not np.any(landmarks != 0):
            landmarks = np.load(npy_path).astype(np.float32)
            if landmarks.shape[1] != NUM_LANDMARKS:
                if landmarks.shape[1] < NUM_LANDMARKS:
                    pad = np.zeros((landmarks.shape[0], NUM_LANDMARKS - landmarks.shape[1], landmarks.shape[2]), dtype=np.float32)
                    landmarks = np.concatenate([landmarks, pad], axis=1)
                else:
                    landmarks = landmarks[:, :NUM_LANDMARKS, :]
            landmarks = self._pad_truncate(landmarks)

        # Compute non-empty frame indices for positional encoding
        non_empty = np.any(landmarks != 0, axis=(1, 2))
        # Ensure at least one frame is non-empty to prevent NaN in attention
        if not np.any(non_empty):
            non_empty[0] = True
            landmarks[0, 0, 0] = 1e-7  # tiny value to avoid all-zero
        non_empty_idxs = np.where(non_empty, np.arange(self.max_frames, dtype=np.float32), -1.0)

        return (
            torch.from_numpy(landmarks),             # [T, 92, 3]
            torch.from_numpy(non_empty_idxs),         # [T]
            torch.tensor(label, dtype=torch.long),
        )


# ═══════════════════════════════════════════════════════════
# Model: Landmark Transformer
# ═══════════════════════════════════════════════════════════
class LandmarkEmbedding(nn.Module):
    def __init__(self, in_features, units):
        super().__init__()
        self.empty_embedding = nn.Parameter(torch.zeros(units))
        self.proj = nn.Sequential(
            nn.Linear(in_features, units, bias=False),
            nn.GELU(),
            nn.Linear(units, units, bias=False),
        )

    def forward(self, x):
        # x: [B, T, landmarks * coords]
        out = self.proj(x)
        # Replace all-zero frames with learned empty embedding
        mask = (x.abs().sum(dim=-1, keepdim=True) == 0)
        out = torch.where(mask, self.empty_embedding, out)
        return out


class LandmarkTransformerEmbedding(nn.Module):
    def __init__(self, max_frames, units, lips_units, hands_units, pose_units):
        super().__init__()
        self.positional_embedding = nn.Embedding(max_frames + 1, units)
        nn.init.zeros_(self.positional_embedding.weight)

        # Separate embeddings for each body part (use all 3 coords: x, y, z)
        self.lips_embedding = LandmarkEmbedding(40 * 3, lips_units)    # 40 lip landmarks x 3D
        self.lh_embedding = LandmarkEmbedding(21 * 3, hands_units)     # 21 left hand x 3D
        self.rh_embedding = LandmarkEmbedding(21 * 3, hands_units)     # 21 right hand x 3D
        self.pose_embedding = LandmarkEmbedding(10 * 3, pose_units)    # 10 pose landmarks x 3D

        # Learned fusion weights
        self.landmark_weights = nn.Parameter(torch.zeros(4))

        # Final projection
        self.fc = nn.Sequential(
            nn.Linear(max(lips_units, hands_units, pose_units), units, bias=False),
            nn.GELU(),
            nn.Linear(units, units, bias=False),
        )

        self.max_frames = max_frames

    def forward(self, frames, non_empty_frame_idxs):
        # frames: [B, T, 92, 3] -> use all 3 coords (x, y, z)
        x = frames  # [B, T, 92, 3]

        lips = x[:, :, 0:40, :].reshape(x.shape[0], x.shape[1], 40 * 3)
        lh = x[:, :, 40:61, :].reshape(x.shape[0], x.shape[1], 21 * 3)
        rh = x[:, :, 61:82, :].reshape(x.shape[0], x.shape[1], 21 * 3)
        pose = x[:, :, 82:92, :].reshape(x.shape[0], x.shape[1], 10 * 3)

        lips_emb = self.lips_embedding(lips)     # [B, T, lips_units]
        lh_emb = self.lh_embedding(lh)           # [B, T, hands_units]
        rh_emb = self.rh_embedding(rh)           # [B, T, hands_units]
        pose_emb = self.pose_embedding(pose)     # [B, T, pose_units]

        # Pad to same size for weighted fusion
        max_units = max(lips_emb.shape[-1], lh_emb.shape[-1], rh_emb.shape[-1], pose_emb.shape[-1])
        if lips_emb.shape[-1] < max_units:
            lips_emb = F.pad(lips_emb, (0, max_units - lips_emb.shape[-1]))
        if lh_emb.shape[-1] < max_units:
            lh_emb = F.pad(lh_emb, (0, max_units - lh_emb.shape[-1]))
        if rh_emb.shape[-1] < max_units:
            rh_emb = F.pad(rh_emb, (0, max_units - rh_emb.shape[-1]))
        if pose_emb.shape[-1] < max_units:
            pose_emb = F.pad(pose_emb, (0, max_units - pose_emb.shape[-1]))

        stacked = torch.stack([lips_emb, lh_emb, rh_emb, pose_emb], dim=-1)  # [B, T, U, 4]
        weights = torch.softmax(self.landmark_weights, dim=0)
        fused = (stacked * weights).sum(dim=-1)  # [B, T, U]
        fused = self.fc(fused)  # [B, T, units]

        # Positional embedding
        max_idx = non_empty_frame_idxs.max(dim=1, keepdim=True).values.clamp(min=1)
        pos_indices = torch.where(
            non_empty_frame_idxs == -1.0,
            torch.tensor(self.max_frames, device=frames.device, dtype=torch.long),
            (non_empty_frame_idxs / max_idx * self.max_frames).long().clamp(0, self.max_frames - 1),
        )
        fused = fused + self.positional_embedding(pos_indices)

        return fused


class TransformerBlock(nn.Module):
    def __init__(self, units, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(units)
        self.attn = nn.MultiheadAttention(units, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(units)
        self.mlp = nn.Sequential(
            nn.Linear(units, units * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(units * mlp_ratio, units),
            nn.Dropout(dropout),
        )

    def forward(self, x, key_padding_mask=None):
        # Pre-norm architecture
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class LandmarkTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        max_frames=64,
        units=512,
        num_blocks=4,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.2,
        emb_dropout=0.1,
        lips_units=384,
        hands_units=384,
        pose_units=256,
    ):
        super().__init__()
        self.embedding = LandmarkTransformerEmbedding(
            max_frames, units, lips_units, hands_units, pose_units
        )
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(units, num_heads, mlp_ratio, dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(units)
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(units, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, frames, non_empty_frame_idxs):
        # frames: [B, T, 92, 3]
        # non_empty_frame_idxs: [B, T]

        x = self.embedding(frames, non_empty_frame_idxs)  # [B, T, units]
        x = self.emb_dropout(x)

        # Create attention mask (True = padding)
        key_padding_mask = (non_empty_frame_idxs == -1.0)  # [B, T]

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)

        # Masked average pooling
        mask = (~key_padding_mask).unsqueeze(-1).float()  # [B, T, 1]
        denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
        x = (x * mask).sum(dim=1) / denom  # [B, units]

        x = self.head_dropout(x)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════
# Mixup
# ═══════════════════════════════════════════════════════════
def mixup_data(frames, idxs, labels, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = frames.size(0)
    index = torch.randperm(batch_size, device=frames.device)

    mixed_frames = lam * frames + (1 - lam) * frames[index]
    mixed_idxs = idxs  # Keep original indices (approximation)
    labels_a, labels_b = labels, labels[index]
    return mixed_frames, mixed_idxs, labels_a, labels_b, lam


def mixup_criterion(criterion, pred, labels_a, labels_b, lam):
    return lam * criterion(pred, labels_a) + (1 - lam) * criterion(pred, labels_b)


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, epoch, use_mixup=True, mixup_alpha=0.2, label_smoothing=0.1):
    model.train()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    total_loss = 0
    correct = 0
    total = 0
    start = time.time()

    for batch_idx, (frames, idxs, labels) in enumerate(loader):
        frames = frames.to(device, non_blocking=True)
        idxs = idxs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            if use_mixup and random.random() < 0.5:
                frames_m, idxs_m, labels_a, labels_b, lam = mixup_data(frames, idxs, labels, mixup_alpha)
                logits = model(frames_m, idxs_m)
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                # For accuracy, use original predictions
                with torch.no_grad():
                    logits_orig = model(frames, idxs)
                    preds = logits_orig.argmax(dim=1)
            else:
                logits = model(frames, idxs)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {total_loss/total:.4f} | "
                  f"Acc: {100*correct/total:.2f}% | "
                  f"Time: {elapsed:.1f}s")

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, device, use_tta=False):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0
    correct = 0
    top5_correct = 0
    total = 0

    for frames, idxs, labels in loader:
        frames = frames.to(device, non_blocking=True)
        idxs = idxs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda"):
            logits = model(frames, idxs)

            # TTA: average with horizontally-flipped prediction
            if use_tta:
                frames_flip = frames.clone()
                frames_flip[:, :, :, 0] = 1.0 - frames_flip[:, :, :, 0]
                # Swap left/right hands
                lh = frames_flip[:, :, 40:61, :].clone()
                rh = frames_flip[:, :, 61:82, :].clone()
                frames_flip[:, :, 40:61, :] = rh
                frames_flip[:, :, 61:82, :] = lh
                logits_flip = model(frames_flip, idxs)
                logits = (logits + logits_flip) / 2.0

            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        # Top-5 accuracy
        _, top5_preds = logits.topk(5, dim=1)
        top5_correct += (top5_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

    return total_loss / total, correct / total, top5_correct / total


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def main():
    parser = argparse.ArgumentParser(description="Train Landmark Transformer for 1000-class ASL")
    parser.add_argument("--data-dir", type=str, default="c:/Users/Asus/project/New Msasl/landmark_data")
    parser.add_argument("--output-dir", type=str, default="c:/Users/Asus/project/New Msasl/outputs/landmark_transformer_1000")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--units", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune: load only model weights, reset optimizer/scheduler/epoch")
    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    # Datasets - use SHARED class_to_idx so labels are consistent
    print("Loading datasets...")
    train_dataset = LandmarkDataset(args.data_dir, split="train", augment=True)
    val_dataset = LandmarkDataset(args.data_dir, split="val", augment=False,
                                  class_to_idx=train_dataset.class_to_idx)

    print(f"Train: {len(train_dataset)} samples, {train_dataset.num_classes} classes")
    print(f"Val: {len(val_dataset)} samples, {val_dataset.num_classes} classes")

    num_classes = train_dataset.num_classes

    # Use weighted random sampler for class balance
    sampler = WeightedRandomSampler(
        weights=train_dataset.sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Model
    model = LandmarkTransformer(
        num_classes=num_classes,
        max_frames=MAX_FRAMES,
        units=args.units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        mlp_ratio=4,
        dropout=args.dropout,
        emb_dropout=0.1,
        lips_units=384,
        hands_units=384,
        pose_units=256,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # Scheduler: cosine with warmup
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler("cuda")

    # Resume
    start_epoch = 0
    best_val_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if args.finetune:
            # Partial load: skip mismatched keys (different num_blocks or num_classes)
            model_dict = model.state_dict()
            pretrained = ckpt["model_state_dict"]
            loaded = 0
            skipped_keys = []
            for k, v in pretrained.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    model_dict[k] = v
                    loaded += 1
                else:
                    skipped_keys.append(k)
            model.load_state_dict(model_dict)
            print(f"  Fine-tune: loaded {loaded}/{len(pretrained)} tensors "
                  f"(skipped {len(skipped_keys)} mismatched)")
            print(f"  Starting fresh from epoch 0 with new optimizer/scheduler")
        else:
            model.load_state_dict(ckpt["model_state_dict"])
            # Full resume: restore optimizer, scheduler, scaler, epoch
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_acc = ckpt.get("best_val_acc", 0.0)
            print(f"  Resumed at epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "training_log.csv")

    # Write CSV header if starting fresh
    if start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_top5_acc", "lr", "time_sec"])

    # Save class mapping
    class_map = {i: name for i, name in enumerate(train_dataset.class_names)}
    with open(os.path.join(args.output_dir, "class_map.json"), "w") as f:
        json.dump(class_map, f, indent=2)

    # Training loop
    patience_counter = 0

    # SWA: start averaging weights after 60% of epochs for better generalization
    swa_model = AveragedModel(model)
    swa_start_epoch = int(args.epochs * 0.6)
    swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=5)
    swa_active = False

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"  LR: {args.lr}, WD: {args.weight_decay}")
    print(f"  Model: {args.num_blocks} blocks, {args.units} dim, {args.num_heads} heads")
    print(f"  Augmentation: mixup={args.mixup_alpha}, label_smoothing={args.label_smoothing}")
    print(f"  Patience: {args.patience} epochs")
    print(f"  SWA starts at epoch {swa_start_epoch}")
    print()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch+1}/{args.epochs} [lr={current_lr:.6f}]")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, epoch,
            use_mixup=True, mixup_alpha=args.mixup_alpha, label_smoothing=args.label_smoothing,
        )

        # Validate with TTA (Test-Time Augmentation)
        val_loss, val_acc, val_top5 = validate(model, val_loader, device, use_tta=True)

        # SWA: update averaged model after swa_start_epoch
        if epoch >= swa_start_epoch:
            if not swa_active:
                print(f"  [SWA] Starting Stochastic Weight Averaging")
                swa_active = True
            swa_model.update_parameters(model)

        epoch_time = time.time() - epoch_start

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {100*val_acc:.2f}% | Val Top-5: {100*val_top5:.2f}%")
        print(f"  Time: {epoch_time:.1f}s")

        # Log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1, f"{train_loss:.4f}", f"{100*train_acc:.2f}",
                             f"{val_loss:.4f}", f"{100*val_acc:.2f}", f"{100*val_top5:.2f}",
                             f"{current_lr:.6f}", f"{epoch_time:.1f}"])

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "class_names": train_dataset.class_names,
                "config": {
                    "units": args.units,
                    "num_blocks": args.num_blocks,
                    "num_heads": args.num_heads,
                    "dropout": args.dropout,
                    "max_frames": MAX_FRAMES,
                    "num_landmarks": NUM_LANDMARKS,
                },
            }, save_path)
            print(f"  *** New best model saved: {100*val_acc:.2f}% ***")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")

        # Save latest checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.output_dir, "latest_checkpoint.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "class_names": train_dataset.class_names,
                "config": {
                    "units": args.units,
                    "num_blocks": args.num_blocks,
                    "num_heads": args.num_heads,
                    "dropout": args.dropout,
                    "max_frames": MAX_FRAMES,
                    "num_landmarks": NUM_LANDMARKS,
                },
            }, ckpt_path)

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping after {args.patience} epochs without improvement.")
            break

        print()

    # SWA: final batch norm update and evaluation
    if swa_active:
        print("\n[SWA] Updating batch norm statistics...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_loss, swa_acc, swa_top5 = validate(swa_model, val_loader, device, use_tta=True)
        print(f"[SWA] Val Acc: {100*swa_acc:.2f}% | Val Top-5: {100*swa_top5:.2f}%")

        if swa_acc > best_val_acc:
            best_val_acc = swa_acc
            swa_path = os.path.join(args.output_dir, "best_model.pth")
            torch.save({
                "epoch": args.epochs,
                "model_state_dict": swa_model.module.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "class_names": train_dataset.class_names,
                "config": {
                    "units": args.units,
                    "num_blocks": args.num_blocks,
                    "num_heads": args.num_heads,
                    "dropout": args.dropout,
                    "max_frames": MAX_FRAMES,
                    "num_landmarks": NUM_LANDMARKS,
                },
            }, swa_path)
            print(f"  *** SWA model is better! Saved: {100*swa_acc:.2f}% ***")
        else:
            print(f"  SWA model ({100*swa_acc:.2f}%) did not beat best ({100*best_val_acc:.2f}%)")

    print(f"\nTraining complete. Best validation accuracy: {100*best_val_acc:.2f}%")
    print(f"Best model saved at: {os.path.join(args.output_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
