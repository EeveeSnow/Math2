import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import csv
import os
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader

from model_mamba import Im2LatexModel
from wraper import MathWritingDataset, Vocabulary, encode_batch

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./torch_compile_cache"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
ACCUMULATION_STEPS = 4 
EPOCHS = 40
LR = 3e-4
MAX_LEN = 256
SAVE_EVERY = 1
CHECKPOINT_PATH = "checkpoints/last_checkpoint.pt"
BEST_MODEL_PATH = "checkpoints/best_model.pt"
LOG_FILE_STEP = "training_step_log.csv"
LOG_FILE_EPOCH = "training_epoch_log.csv"
LOG_STEP_INTERVAL = 100

PAD = 0
SOS = 1
EOS = 2

# ======================
# UTILS
# ======================
def save_checkpoint(state, is_best, filename=CHECKPOINT_PATH):
    torch.save(state, filename)
    if is_best:
        torch.save(state, BEST_MODEL_PATH)

def log_to_csv(filepath, data):
    """Utility to append a row to a CSV file"""
    file_exists = os.path.isfile(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(data.keys())
        writer.writerow(data.values())

def sampling_prob(epoch):
    if epoch < 5: return 0.0
    return min(0.3, 0.02 * (epoch - 5))

def mix_inputs(gt_tokens, pred_tokens, p):
    if p == 0.0: return gt_tokens
    mask = (torch.rand_like(gt_tokens.float()) < p)
    return torch.where(mask, pred_tokens, gt_tokens)

# def collate_fn(batch):
#     images = torch.stack([item["image"] for item in batch])
#     latexs = [item["latex"] for item in batch]
#     captions = pad_sequence([item["captions"] for item in batch], batch_first=True, padding_value=0)
    
#     return {
#         "image": images,
#         "latex": latexs,
#         "captions": captions
#     }

# ======================
# TRAIN
# ======================
def train():
    os.makedirs("checkpoints", exist_ok=True)
    
    ds = load_dataset("deepcopy/MathWriting-Human")
    train_dataset = MathWritingDataset(ds["train"], image_size=(384, 384))
    val_dataset = MathWritingDataset(ds["val"], image_size=(384, 384))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=8)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True, persistent_workers=True, prefetch_factor=8)

    start_epoch = 1
    best_val_loss = float('inf')
    global_step = 0
    total_updates = (EPOCHS * len(train_loader)) // ACCUMULATION_STEPS
    if os.path.exists(CHECKPOINT_PATH):
        print(f"--- Loading checkpoint ---")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        vocab = checkpoint['vocab']
        model = Im2LatexModel(len(vocab)).to(DEVICE)
        model = torch.compile(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        global_step = checkpoint.get('global_step', (start_epoch-1) * (len(train_loader)//ACCUMULATION_STEPS))
        print(f"--- Resuming from Epoch {start_epoch} ---")
    else:
        print("--- Starting training from scratch ---")
        vocab = Vocabulary(freq_threshold=2)
        vocab.build_vocab([sample["latex"] for sample in ds["train"]])
        model = Im2LatexModel(len(vocab)).to(DEVICE)
        model = torch.compile(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates)

        
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    scaler = GradScaler()

    print(len(vocab))
    print(len(train_loader))
    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        running_step_loss = 0.0

        optimizer.zero_grad()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, batch in enumerate(loop):
            images = batch["image"].to(DEVICE, non_blocking=True)
            captions = encode_batch(batch["latex"], vocab).to(DEVICE, non_blocking=True)
            inp = captions[:, :-1]

            with autocast('cuda'):
                logits, _ = model(images, inp)
                loss = criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))
                loss = loss / ACCUMULATION_STEPS
            
            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                # Tracking
                global_step += 1
                current_loss = loss.item() * ACCUMULATION_STEPS
                epoch_loss += current_loss
                running_step_loss += current_loss
                
                if global_step % LOG_STEP_INTERVAL == 0:
                    avg_step_loss = running_step_loss / LOG_STEP_INTERVAL
                    current_lr = optimizer.param_groups[0]['lr']
                    
                    log_to_csv(LOG_FILE_STEP, {
                        "epoch": epoch,
                        "step": global_step,
                        "batch_idx": i,
                        "loss": round(avg_step_loss, 5),
                        "lr": f"{current_lr:.8f}"
                    })
                    running_step_loss = 0.0

                
                loop.set_postfix(loss=current_loss)

        train_loss = epoch_loss / (len(train_loader) / ACCUMULATION_STEPS)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                captions = encode_batch(batch["latex"], vocab).to(DEVICE)

                inp = captions[:, :-1]

                with autocast('cuda'):
                    logits, _ = model(images, inp)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), captions[:, 1:].reshape(-1))
                    loss = loss / ACCUMULATION_STEPS
                val_losses.append(loss.item())

        val_loss = sum(val_losses) / len(val_losses) if val_losses else -1
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        log_to_csv(LOG_FILE_EPOCH, {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_loss": round(val_loss, 5),
            "lr": optimizer.param_groups[0]['lr']
        })

        is_best = val_loss < best_val_loss
        if is_best: best_val_loss = val_loss

        checkpoint_state = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'vocab': vocab,
            'best_val_loss': best_val_loss,
            'train_loss': train_loss,
            'val_loss': val_loss
        }
        
        save_checkpoint(checkpoint_state, is_best)
        
        if epoch % SAVE_EVERY == 0:
            torch.save(checkpoint_state, f"checkpoints/model_epoch_{epoch}.pt")

if __name__ == "__main__":
    train()