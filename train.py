"""
train.py  –  Training loop, validation, checkpointing, and early stopping.

Features:
- Adam optimizer with ReduceLROnPlateau scheduler.
- Cross-entropy loss ignoring <pad> tokens.
- Gradient clipping (1.0).
- Dynamic teacher forcing decay per epoch.
- Model checkpointing based on best validation loss.
- Early stopping with configurable patience.
- Perplexity logging for train and validation.
- Saves history to logs/train_history.json.
"""

import sys
import math
import time
import json

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import config as cfg
from data import get_dataloaders
from model import build_model, Seq2Seq


# ══════════════════════════════════════════════════════════════
# Training & Evaluation Step Functions
# ══════════════════════════════════════════════════════════════

def train_epoch(
    model: Seq2Seq,
    iterator,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    clip: float,
    teacher_forcing_ratio: float,
) -> float:
    """Train for a single epoch and return average loss."""
    model.train()
    epoch_loss = 0.0

    for i, (src, tgt, src_lengths, src_mask) in enumerate(iterator):
        src = src.to(cfg.DEVICE)
        tgt = tgt.to(cfg.DEVICE)
        src_mask = src_mask.to(cfg.DEVICE)

        optimizer.zero_grad()

        # output: (tgt_len, batch_size, tgt_vocab_size)
        output = model(
            src=src,
            tgt=tgt,
            src_lengths=src_lengths,
            src_mask=src_mask,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        output_dim = output.shape[-1]

        # Ignore first token (<sos>) when calculating loss
        # Flatten: output -> ((tgt_len - 1) * batch, tgt_vocab_size)
        output_flat = output[1:].view(-1, output_dim)
        tgt_flat = tgt[1:].view(-1)

        loss = criterion(output_flat, tgt_flat)
        loss.backward()

        # Gradient clipping to prevent exploding gradients in RNNs
        nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate_epoch(
    model: Seq2Seq,
    iterator,
    criterion: nn.CrossEntropyLoss,
) -> float:
    """Evaluate on validation set and return average loss."""
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for i, (src, tgt, src_lengths, src_mask) in enumerate(iterator):
            src = src.to(cfg.DEVICE)
            tgt = tgt.to(cfg.DEVICE)
            src_mask = src_mask.to(cfg.DEVICE)

            # Turn off teacher forcing during validation
            output = model(
                src=src,
                tgt=tgt,
                src_lengths=src_lengths,
                src_mask=src_mask,
                teacher_forcing_ratio=0.0,
            )

            output_dim = output.shape[-1]
            output_flat = output[1:].view(-1, output_dim)
            tgt_flat = tgt[1:].view(-1)

            loss = criterion(output_flat, tgt_flat)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: float, end_time: float):
    """Helper to convert time delta to minutes and seconds."""
    elapsed = end_time - start_time
    mins = int(elapsed / 60)
    secs = int(elapsed - (mins * 60))
    return mins, secs


# ══════════════════════════════════════════════════════════════
# Main Training Loop
# ══════════════════════════════════════════════════════════════

def train():
    print(f"[train] Using device: {cfg.DEVICE}")

    # 1. Load data
    train_loader, valid_loader, test_loader, src_vocab, tgt_vocab = get_dataloaders()

    # 2. Build model
    model = build_model(len(src_vocab), len(tgt_vocab))

    # 3. Setup optimizer, scheduler, criterion
    optimizer = Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.LR_FACTOR,
        patience=cfg.LR_PATIENCE,
        verbose=True,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=cfg.PAD_IDX)

    best_valid_loss = float("inf")
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_ppl": [],
        "val_loss": [],
        "val_ppl": [],
        "lr": [],
        "teacher_forcing": [],
    }

    tf_ratio = cfg.TEACHER_FORCING_INIT

    print("\n" + "=" * 60)
    print("Starting Training".center(60))
    print("=" * 60)

    for epoch in range(cfg.N_EPOCHS):
        start_time = time.time()

        # Train & Validate
        train_loss = train_epoch(
            model=model,
            iterator=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            clip=cfg.CLIP_GRAD,
            teacher_forcing_ratio=tf_ratio,
        )
        val_loss = evaluate_epoch(
            model=model,
            iterator=valid_loader,
            criterion=criterion,
        )

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # LR step based on validation loss
        scheduler.step(val_loss)

        # Perplexity = exp(loss)
        train_ppl = math.exp(min(train_loss, 100))
        val_ppl   = math.exp(min(val_loss, 100))

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_ppl"].append(train_ppl)
        history["val_loss"].append(val_loss)
        history["val_ppl"].append(val_ppl)
        history["lr"].append(current_lr)
        history["teacher_forcing"].append(tf_ratio)

        print(
            f"Epoch: {epoch + 1:02d} / {cfg.N_EPOCHS:02d} | "
            f"Time: {epoch_mins}m {epoch_secs}s | "
            f"TF: {tf_ratio:.2f} | "
            f"LR: {current_lr:.6f}"
        )
        print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:7.3f}")
        print(f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {val_ppl:7.3f}")

        # Checkpointing
        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "src_vocab_len": len(src_vocab),
                    "tgt_vocab_len": len(tgt_vocab),
                },
                cfg.BEST_MODEL_PATH,
            )
            print(f"\t[*] Saved new best model to {cfg.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            print(f"\t[-] Patience counter: {patience_counter}/{cfg.PATIENCE}")

        # Decay teacher forcing ratio
        tf_ratio = max(cfg.TEACHER_FORCING_MIN, tf_ratio - cfg.TEACHER_FORCING_DECAY)

        # Early stopping check
        if patience_counter >= cfg.PATIENCE:
            print("\n[Early Stopping] Validation loss did not improve. Stopping training.")
            break

    # Save training history log
    with open(cfg.TRAIN_HISTORY, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\n[train] Training history written to {cfg.TRAIN_HISTORY}")
    print("[train] Training completed.")


if __name__ == "__main__":
    train()
