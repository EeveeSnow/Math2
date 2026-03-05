import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import csv
import os
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader

# Assuming these are your local imports
from data import Vocabulary, encode_batch
from dataloaders import build_dataloaders
from model import Im2LatexModel
from interface import greedy_decode, beam_search
from wraper import MathWritingDataset

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
EPOCHS = 40
LR = 3e-4
MAX_LEN = 512
SAVE_EVERY = 5
CHECKPOINT_PATH = "checkpoints/last_checkpoint.pt" # Path for resuming
BEST_MODEL_PATH = "checkpoints/best_model.pt"
LOG_FILE = "training_log.csv"

PAD = 0
SOS = 1
EOS = 2

# ======================
# UTILS
# ======================

def save_checkpoint(state, is_best, filename=CHECKPOINT_PATH):
    """Saves all training metadata"""
    torch.save(state, filename)
    if is_best:
        torch.save(state, BEST_MODEL_PATH)

# (Keeping your existing sampling_prob, mix_inputs, metrics functions...)
def sampling_prob(epoch):
    if epoch < 5: return 0.0
    return min(0.3, 0.02 * (epoch - 5))

def mix_inputs(gt_tokens, pred_tokens, p):
    if p == 0.0: return gt_tokens
    mask = (torch.rand_like(gt_tokens.float()) < p)
    return torch.where(mask, pred_tokens, gt_tokens)

# ======================
# TRAIN
# ======================
def train():
    os.makedirs("checkpoints", exist_ok=True)
    
    # 1. Prepare Data
    ds = load_dataset("deepcopy/MathWriting-Human")
    train_dataset = MathWritingDataset(ds["train"], image_size=(384, 384))
    val_dataset = MathWritingDataset(ds["val"], image_size=(384, 384))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

    # 2. Initialize Vocab and Model
    # We check if checkpoint exists FIRST to load the vocab from there
    start_epoch = 1
    best_val_loss = float('inf')
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"--- Loading checkpoint: {CHECKPOINT_PATH} ---")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        
        vocab = checkpoint['vocab']
        model = Im2LatexModel(len(vocab)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
        
        # Load states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"--- Resuming from Epoch {start_epoch} ---")
    else:
        print("--- Starting training from scratch ---")
        vocab = Vocabulary(freq_threshold=2)
        corpus = [sample["latex"] for sample in ds["train"]]
        vocab.build_vocab(corpus)
        
        model = Im2LatexModel(len(vocab)).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
        
        # Initialize log file
        with open(LOG_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])

    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    # 3. Training Loop
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        p = sampling_prob(epoch)

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            images = batch["image"].to(DEVICE)
            captions = encode_batch(batch["latex"], vocab).to(DEVICE)

            inp = captions[:, :-1]
            target = captions[:, 1:]

            logits, _ = model(images, inp)

            if p > 0:
                preds = logits.argmax(-1)
                inp = mix_inputs(inp, preds.detach(), p)
                logits, _ = model(images, inp)

            loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # 4. Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                captions = encode_batch(batch["latex"], vocab).to(DEVICE)
                logits, _ = model(images, captions[:, :-1])
                loss = criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))
                val_losses.append(loss.item())

        val_loss = sum(val_losses) / len(val_losses) if val_losses else -1
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # 5. Logging
        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss])

        # 6. Save State
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'vocab': vocab, # Storing the whole object
            'best_val_loss': best_val_loss,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        # Save "Last" checkpoint for resuming
        save_checkpoint(checkpoint_state, is_best)
        
        # Optional: Save periodic snapshots
        if epoch % SAVE_EVERY == 0:
            torch.save(checkpoint_state, f"checkpoints/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()