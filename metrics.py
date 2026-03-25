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


def levenshtein(seq1, seq2):
    m = len(seq1)
    n = len(seq2)

    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1):
        dp[i][0] = i
    for j in range(n+1):
        dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):

            if seq1[i-1] == seq2[j-1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )

    return dp[m][n]


def avg_edit_distance(pred, tgt):
    return sum(
        levenshtein(strip_eos(p), strip_eos(t))
        for p, t in zip(pred, tgt)
    ) / len(pred)