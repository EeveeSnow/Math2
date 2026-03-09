import torch


PAD = 0
SOS = 1
EOS = 2

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