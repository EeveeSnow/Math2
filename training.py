import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import os
import math

from data import MathDataset, Vocabulary
from model import Im2LatexModel

# ======================
# CONFIG
# ======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 50
LR = 3e-4
SAVE_EVERY = 5
MAX_LEN = 512

PAD = 0
SOS = 1
EOS = 2

# ======================
# COLLATE FUNCTION
# ======================
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)

    captions = pad_sequence(
        captions, batch_first=True, padding_value=PAD
    )

    return images, captions

# ======================
# METRICS
# ======================
def token_accuracy(pred, tgt):
    mask = tgt != PAD
    correct = (pred == tgt) & mask
    return correct.sum().item() / mask.sum().item()

def exact_match(pred, tgt):
    matches = 0
    for p, t in zip(pred, tgt):
        p = p.tolist()
        t = t.tolist()
        if EOS in t:
            t = t[:t.index(EOS)]
        if EOS in p:
            p = p[:p.index(EOS)]
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
    dist = 0
    for p, t in zip(pred, tgt):
        p = p.tolist()
        t = t.tolist()
        if EOS in p: p = p[:p.index(EOS)]
        if EOS in t: t = t[:t.index(EOS)]
        dist += levenshtein(p, t)
    return dist / len(pred)

# ======================
# TRAIN
# ======================
def train():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    vocab = Vocabulary(freq_threshold=2)
    dataset = MathDataset("data/inkml", vocab, transform)

    vocab.build_vocab([latex for _, latex in dataset.samples])

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    model = Im2LatexModel(len(vocab)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    scaler = torch.cuda.amp.GradScaler()

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        loop = tqdm(loader, desc=f"Epoch {epoch}")

        for images, captions in loop:
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                logits, _ = model(images, captions[:, :-1])
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    captions[:, 1:].reshape(-1)
                )

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # ======================
        # EVALUATION (light)
        # ======================
        model.eval()
        with torch.no_grad():
            images, captions = next(iter(loader))
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            preds = generate(model, images)
            acc = token_accuracy(preds, captions[:, 1:])
            em = exact_match(preds, captions[:, 1:])
            ed = avg_edit_distance(preds, captions[:, 1:])

        print(
            f"\nEpoch {epoch} | "
            f"Loss: {total_loss/len(loader):.4f} | "
            f"TokenAcc: {acc:.3f} | "
            f"EM: {em:.3f} | "
            f"EditDist: {ed:.2f}"
        )

        # ======================
        # SAVE
        # ======================
        if epoch % SAVE_EVERY == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "vocab": vocab.stoi
                },
                f"checkpoints/model_epoch_{epoch}.pt"
            )

# ======================
# GENERATION
# ======================
def generate(model, images, max_len=MAX_LEN):
    model.eval()
    memory, _ = model.encoder(images)
    out = torch.full((images.size(0), 1), SOS, device=images.device)

    for _ in range(max_len):
        logits = model.decoder(memory, out)
        next_tok = logits[:, -1].argmax(-1, keepdim=True)
        out = torch.cat([out, next_tok], dim=1)
        if (next_tok == EOS).all():
            break
    return out[:, 1:]

if __name__ == "__main__":
    train()
