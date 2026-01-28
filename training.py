import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import csv
import os
from pathlib import Path

from data import Vocabulary, encode_batch
from dataloaders import build_dataloaders
from model import Im2LatexModel

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
EPOCHS = 40
LR = 3e-4
MAX_LEN = 512
SAVE_EVERY = 5

PAD = 0
SOS = 1
EOS = 2

LOG_FILE = "training_log.csv"

# ======================
# SCHEDULED SAMPLING
# ======================
def sampling_prob(epoch):
    if epoch < 5:
        return 0.0
    return min(0.3, 0.02 * (epoch - 5))


def mix_inputs(gt_tokens, pred_tokens, p):
    """
    gt_tokens: (B, T)
    pred_tokens: (B, T)
    """
    if p == 0.0:
        return gt_tokens
    mask = (torch.rand_like(gt_tokens.float()) < p)
    return torch.where(mask, pred_tokens, gt_tokens)

# ======================
# METRICS
# ======================
def strip_eos(seq):
    seq = seq.tolist()
    return seq[:seq.index(EOS)] if EOS in seq else seq


def token_accuracy(pred, tgt):
    device = tgt.device
    
    if pred.size(1) < tgt.size(1):
        diff = tgt.size(1) - pred.size(1)
        pad = torch.full((pred.size(0), diff), PAD, device=device)
        pred = torch.cat([pred, pad], dim=1)
    
    if tgt.size(1) < pred.size(1):
        diff = pred.size(1) - tgt.size(1)
        pad = torch.full((tgt.size(0), diff), PAD, device=device)
        tgt = torch.cat([tgt, pad], dim=1)

    mask = (tgt != PAD)
    correct = (pred == tgt) & mask
    
    total_correct = correct.sum().item()
    total_tokens = mask.sum().item()
    
    if total_tokens == 0:
        return 0.0
        
    return total_correct / total_tokens


def exact_match(pred, tgt): 
    matches = 0 
    for p, t in zip(pred, tgt): 
        p = p.tolist() 
        t = t.tolist() 
        if EOS in t: t = t[:t.index(EOS)] 
        if EOS in p: p = p[:p.index(EOS)] 
        matches += int(p == t) 
    return matches / len(pred)


def levenshtein(a, b):
    dp = [[i + j if i * j == 0 else 0 for j in range(len(b)+1)] for i in range(len(a)+1)]
    for i in range(1, len(a)+1):
        for j in range(1, len(b)+1):
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + (a[i-1] != b[j-1])
            )
    return dp[-1][-1]


def avg_edit_distance(pred, tgt):
    return sum(
        levenshtein(strip_eos(p), strip_eos(t))
        for p, t in zip(pred, tgt)
    ) / len(pred)

# ======================
# DECODING
# ======================
@torch.no_grad()
def greedy_decode(model, images, max_len=MAX_LEN):
    model.eval()
    device = images.device

    memory, (H, W) = model.encoder(images)
    pos = model.pos_enc(H, W, device)
    memory = memory + pos.unsqueeze(0)

    out = torch.full((images.size(0), 1), SOS, device=device)

    for _ in range(max_len):
        logits = model.decoder(memory, out)
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, next_tok], dim=1)
        if (next_tok == EOS).all():
            break

    return out[:, 1:]


@torch.no_grad()
def beam_search(model, images, beam_size=5, max_len=MAX_LEN, length_penalty=0.7):
    model.eval()
    device = images.device
    B = images.size(0)

    memory, (H, W) = model.encoder(images)
    pos = model.pos_enc(H, W, device)
    memory = memory + pos.unsqueeze(0)

    beams = [[(torch.tensor([SOS], device=device), 0.0)] for _ in range(B)]
    finished = [[] for _ in range(B)]

    for _ in range(max_len):
        new_beams = [[] for _ in range(B)]

        for b in range(B):
            for seq, score in beams[b]:
                if seq[-1] == EOS:
                    finished[b].append((seq, score))
                    continue

                logits = model.decoder(memory[b:b+1], seq.unsqueeze(0))
                log_probs = torch.log_softmax(logits[:, -1], dim=-1)
                vals, idx = log_probs.topk(beam_size)

                for k in range(beam_size):
                    new_seq = torch.cat([seq, idx[0, k:k+1]])
                    new_score = score + vals[0, k].item()
                    new_beams[b].append((new_seq, new_score))

            new_beams[b] = sorted(
                new_beams[b],
                key=lambda x: x[1] / (len(x[0]) ** length_penalty),
                reverse=True
            )[:beam_size]

        beams = new_beams
        if all(len(finished[b]) >= beam_size for b in range(B)):
            break

    results = []
    for b in range(B):
        candidates = finished[b] if finished[b] else beams[b]
        best = max(
            candidates,
            key=lambda x: x[1] / (len(x[0]) ** length_penalty)
        )[0]
        results.append(best[1:])

    return pad_sequence(results, batch_first=True, padding_value=PAD)

# ======================
# TRAIN
# ======================
def train():
    vocab = Vocabulary(freq_threshold=2)
    corpus = []

    for txt in Path("CROHME_processed").glob("*.txt"):
        corpus.append(txt.read_text().strip())

    vocab.build_vocab(corpus)

    model = Im2LatexModel(len(vocab)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)

    os.makedirs("checkpoints", exist_ok=True)

    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "token_acc", "EM", "edit_dist"]
        )

    best_em = 0.458
    train_loader, val_loader = build_dataloaders(
            image_size=(224, 224),
            root_dir="CROHME_processed",
            batch_size=BATCH_SIZE,
            epoch=1
        )
    for epoch in range(1, EPOCHS + 1):
        

        # ===== TRAIN =====
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

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)

        # ===== VALIDATION =====
        model.eval()
        val_loss = 0.0
        val_losses = []
        val_accs = []
        val_ems = []
        val_eds = []

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(DEVICE)
                captions = encode_batch(batch["latex"], vocab).to(DEVICE)

                logits, _ = model(images, captions[:, :-1])
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    captions[:, 1:].reshape(-1)
                )
                val_losses.append(loss.item())

                preds = beam_search(model, images) if epoch >= 10 and epoch % 5 == 0 else greedy_decode(model, images)
                
                val_accs.append(token_accuracy(preds, captions[:, 1:]))
                val_ems.append(exact_match(preds, captions[:, 1:]))
                val_eds.append(avg_edit_distance(preds, captions[:, 1:]))


        if len(val_accs) > 0:
            acc = sum(val_accs) / len(val_accs)
            em = sum(val_ems) / len(val_ems)
            ed = sum(val_eds) / len(val_eds)
            val_loss = sum(val_losses) / len(val_losses)
        else:
            acc, em, ed, val_loss = 0, 0, 0, 0

        print(
            f"\nEpoch {epoch} | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | "
            f"Acc {acc:.3f} | EM {em:.3f} | ED {ed:.2f}"
        )

        with open(LOG_FILE, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, acc, em, ed])

        if em > best_em:
            best_em = em
            torch.save(model.state_dict(), "checkpoints/best_model.pt")

        if epoch % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f"checkpoints/model_{epoch}.pt")


if __name__ == "__main__":
    train()
