"""
Training script for SwinMambaTex.

Fixed bugs vs original:
  [F1] ZeroDivisionError on odd epochs (samples/total_tokens were 0)
  [F2] Checkpoint was loaded twice — now loaded once, vocab & weights extracted
  [F3] Last batch is always stepped (partial accumulation window handled)
  [F4] LR warmup added via SequentialLR (LinearLR → CosineAnnealingLR)
  [F5] forward() returns 3 values; coverage used as optional regulariser
  [F6] raw_model kept alongside compiled model for generate() in validation
  [F7] model.train() called at end of every epoch (matters after generate())
  [F8] initial zero_grad uses set_to_none=True for consistency
"""

import csv
import os
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from old.model_mamba_2layer import SwinMambaTex as Im2LatexModel
from wraper import MathWritingDataset, Vocabulary, encode_batch
from metrics import levenshtein

os.environ["TORCHINDUCTOR_CACHE_DIR"] = "./torch_compile_cache"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
# TODO: update to different LR for encoder and decoder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 12
ACCUMULATION_STEPS = 4
EPOCHS = 20
LR = 2e-4
MIN_LR = 1e-6
WARMUP_RATIO = 0.05
MAX_LEN = 256
CHECKPOINT_PATH = "checkpoints/last_checkpoint.pt"
BEST_MODEL_PATH = "checkpoints/best_model.pt"
LOG_DIR = "logs"
LOG_FILE_STEP = f"{LOG_DIR}/step_log.csv"
LOG_FILE_EPOCH = f"{LOG_DIR}/epoch_log.csv"
LOG_STEP_INTERVAL = 100
METRIC_EVERY_N = 2
SAVE_EVERY_N = 1

PAD, SOS, EOS = 0, 1, 2

DRAFT_LOSS_W    = 0.4
REFINE_LOSS_W   = 0.6
COVERAGE_LOSS_W = 0.05

USE_BFLOAT16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
AMP_DTYPE    = torch.bfloat16 if USE_BFLOAT16 else torch.float16


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def decode_tokens(token_ids: List[int], vocab) -> List[str]:
    out = []
    for t in token_ids:
        if t in (PAD, SOS):
            continue
        if t == EOS:
            break
        out.append(vocab.itos[t])
    return out


def coverage_regularisation(coverage: torch.Tensor) -> torch.Tensor:
    coverage = coverage.float().clamp(min=0.0)
    total    = coverage.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    cov_norm = coverage / total
    entr     = -(cov_norm * torch.log(cov_norm.clamp(min=1e-8))).sum(dim=-1)
    return entr.mean()   # minimise or maximise entropy ???


def clean_state_dict(sd: dict) -> dict:
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def log_to_csv(path: str, row: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(row.keys())
        w.writerow(row.values())


def save_checkpoint(state: dict, is_best: bool):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    torch.save(state, CHECKPOINT_PATH)
    if is_best:
        torch.save(state, BEST_MODEL_PATH)


def build_scheduler(optimizer, total_steps: int, warmup_steps: int):
    warm = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1e-9 / LR,
        end_factor   = 1.0,
        total_iters  = warmup_steps,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = max(total_steps - warmup_steps, 1),
        eta_min = MIN_LR,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [warm, cosine],
        milestones = [warmup_steps],
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    ds = load_dataset("deepcopy/MathWriting-Human")
    checkpoint = None
    if os.path.exists(CHECKPOINT_PATH):
        print("Checkpoint found — restoring …")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        vocab = checkpoint["vocab"]
    else:
        print("No checkpoint — building vocab from scratch …")
        vocab = Vocabulary(freq_threshold=2)
        vocab.build_vocab([s["latex"] for s in ds["train"]])


    def collate_fn(batch):
        images   = torch.stack([item["image"] for item in batch])
        captions = encode_batch([item["latex"] for item in batch], vocab)
        return {"image": images, "captions": captions}

    train_loader = DataLoader(
        MathWritingDataset(ds["train"], image_size=(384, 384)),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
        prefetch_factor=4, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        MathWritingDataset(ds["val"], image_size=(384, 384)),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
        prefetch_factor=4, collate_fn=collate_fn,
    )

    raw_model = Im2LatexModel(len(vocab)).to(DEVICE)

    steps_per_epoch   = len(train_loader)
    updates_per_epoch = (steps_per_epoch + ACCUMULATION_STEPS - 1) // ACCUMULATION_STEPS
    total_updates     = EPOCHS * updates_per_epoch
    warmup_updates    = max(1, int(total_updates * WARMUP_RATIO))

    start_epoch   = 1
    best_val_loss = float("inf")
    global_step   = 0

    if checkpoint is not None:
        raw_model.load_state_dict(clean_state_dict(checkpoint["model_state_dict"]))
        model     = torch.compile(raw_model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
        scheduler = build_scheduler(optimizer, total_updates, warmup_updates)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch   = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
        global_step   = checkpoint.get("global_step", (start_epoch - 1) * updates_per_epoch)
        del checkpoint          # free CPU RAM
        print(f"Resumed: epoch {start_epoch}, step {global_step}")
    else:
        model     = torch.compile(raw_model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
        scheduler = build_scheduler(optimizer, total_updates, warmup_updates)
        print("Starting from scratch.")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    scaler    = GradScaler(enabled=not USE_BFLOAT16)

    print(f"Vocab size    : {len(vocab)}")
    print(f"Steps / epoch : {steps_per_epoch}")
    print(f"Total updates : {total_updates}  (warmup {warmup_updates})")
    print(f"AMP dtype     : {AMP_DTYPE}")
    print(f"Device        : {DEVICE}")

    for epoch in range(start_epoch, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        running_step_loss = 0.0
        running_draft_loss = 0.0
        running_refine_loss = 0.0
        running_cov_loss = 0.0
        n_updates = 0

        # [F8] Consistent set_to_none=True everywhere
        optimizer.zero_grad(set_to_none=True)
        loop = tqdm(train_loader, desc=f"Train {epoch}/{EPOCHS}")

        for i, batch in enumerate(loop):
            images   = batch["image"].to(DEVICE, non_blocking=True)
            captions = batch["captions"].to(DEVICE, non_blocking=True)
            inp      = captions[:, :-1]
            targets  = captions[:, 1:]

            with autocast("cuda", dtype=AMP_DTYPE):
                draft_logits, refined_logits, coverage = model(images, inp)

                draft_loss  = criterion(
                    draft_logits.reshape(-1, draft_logits.size(-1)),
                    targets.reshape(-1),
                )
                refine_loss = criterion(
                    refined_logits.reshape(-1, refined_logits.size(-1)),
                    targets.reshape(-1),
                )
                loss = DRAFT_LOSS_W * draft_loss + REFINE_LOSS_W * refine_loss

                if COVERAGE_LOSS_W > 0.0:
                    cov_loss = coverage_regularisation(coverage)
                    loss     = loss + COVERAGE_LOSS_W * cov_loss

                draft_loss = draft_loss / ACCUMULATION_STEPS
                refine_loss = refine_loss / ACCUMULATION_STEPS
                cov_loss = cov_loss / ACCUMULATION_STEPS
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            is_last_batch = (i + 1) == len(train_loader)
            if (i + 1) % ACCUMULATION_STEPS == 0 or is_last_batch:
                scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                global_step += 1
                n_updates += 1
                step_loss = loss.item() * ACCUMULATION_STEPS
                epoch_loss += step_loss
                running_step_loss += step_loss
                running_draft_loss += draft_loss.item() * ACCUMULATION_STEPS
                running_refine_loss += refine_loss.item() * ACCUMULATION_STEPS
                running_cov_loss += cov_loss.item() * ACCUMULATION_STEPS


                if global_step % LOG_STEP_INTERVAL == 0:
                    avg_sl = running_step_loss / LOG_STEP_INTERVAL
                    log_to_csv(LOG_FILE_STEP, {
                        "epoch": epoch,
                        "step": global_step,
                        "batch_idx": i,
                        "loss": round(avg_sl, 5),
                        "draft_loss": round(running_draft_loss / LOG_STEP_INTERVAL, 5),
                        "refine_loss": round(running_refine_loss / LOG_STEP_INTERVAL, 5),
                        "cov_loss": round(running_cov_loss / LOG_STEP_INTERVAL, 5),
                        "lr": f"{optimizer.param_groups[0]['lr']:.8f}",
                    })
                    running_step_loss = 0.0
                    running_draft_loss = 0.0
                    running_refine_loss = 0.0
                    running_cov_loss = 0.0

                loop.set_postfix(loss=f"{step_loss:.4f}")

        train_loss = epoch_loss / max(n_updates, 1)




        model.eval()

        val_losses = []
        run_metrics = (epoch % METRIC_EVERY_N == 0)
        total_ed = 0
        total_ref_len = 0
        exact_matches = 0
        n_samples = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  Val  {epoch}", leave=False):
                images = batch["image"].to(DEVICE, non_blocking=True)
                captions = batch["captions"].to(DEVICE, non_blocking=True)
                inp = captions[:, :-1]
                targets = captions[:, 1:]

                with autocast("cuda", dtype=AMP_DTYPE):
                    _, refined_logits, _ = model(images, inp)
                    v_loss = criterion(
                        refined_logits.reshape(-1, refined_logits.size(-1)),
                        targets.reshape(-1),
                    )

                val_losses.append(v_loss.item())

                if run_metrics:
                    # TODO: generation on bf16 will be faster ??? 
                    pred_tokens = raw_model.generate(
                        images,
                        start_token_id = SOS,
                        max_new_tokens = MAX_LEN,
                        eos_token_id   = EOS,
                    ).cpu()
                    captions_cpu = captions.cpu()

                    for pred, tgt in zip(pred_tokens, captions_cpu):
                        pred_seq = decode_tokens(pred.tolist(), vocab)
                        tgt_seq  = decode_tokens(tgt.tolist(), vocab)
                        ed = levenshtein(pred_seq, tgt_seq)
                        total_ed      += ed
                        total_ref_len += max(len(tgt_seq), 1)
                        exact_matches += int(pred_seq == tgt_seq)
                        n_samples     += 1

        val_loss = sum(val_losses) / len(val_losses)
        cur_lr   = optimizer.param_groups[0]["lr"]

        if run_metrics and n_samples > 0:
            avg_ed = total_ed / n_samples
            norm_ed = total_ed / total_ref_len
            seq_acc = exact_matches / n_samples

            print(
                f"Ep {epoch:3d} | Train {train_loss:.4f} | Val {val_loss:.4f} | "
                f"ED {avg_ed:.3f} | NED {norm_ed:.3f} | Acc {seq_acc:.3f} | "
                f"LR {cur_lr:.2e}"
            )
            log_to_csv(LOG_FILE_EPOCH, {
                "epoch":              epoch,
                "train_loss":         round(train_loss, 5),
                "val_loss":           round(val_loss, 5),
                "edit_distance":      round(avg_ed, 5),
                "norm_edit_distance": round(norm_ed, 5),
                "sequence_accuracy":  round(seq_acc, 5),
                "lr":                 cur_lr,
            })
        else:
            print(
                f"Ep {epoch:3d} | Train {train_loss:.4f} | Val {val_loss:.4f} | "
                f"LR {cur_lr:.2e}"
            )
            log_to_csv(LOG_FILE_EPOCH, {
                "epoch":              epoch,
                "train_loss":         round(train_loss, 5),
                "val_loss":           round(val_loss, 5),
                "edit_distance":      None,
                "norm_edit_distance": None,
                "sequence_accuracy":  None,
                "lr":                 cur_lr,
            })

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        state = {
            "epoch":                epoch,
            "global_step":          global_step,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "vocab":                vocab,
            "best_val_loss":        best_val_loss,
            "train_loss":           train_loss,
            "val_loss":             val_loss,
        }
        save_checkpoint(state, is_best)

        if epoch % SAVE_EVERY_N == 0:
            torch.save(state, f"checkpoints/epoch_{epoch:03d}.pt")

        model.train()


if __name__ == "__main__":
    train()