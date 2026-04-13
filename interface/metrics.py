import torch
import nltk

import interface.configs as conf


try:
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    print("Внимание: SymPy или antlr4 не установлены. AST-метрика будет использовать fallback.")

PAD = 0
SOS = 1
EOS = 2

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
        if EOS in t: 
            t = t[:t.index(EOS)] 
        if EOS in p: 
            p = p[:p.index(EOS)] 
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

def get_edit_operations(ref_tokens, hyp_tokens):
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): 
        dp[i][0] = i
    for j in range(m + 1): 
        dp[0][j] = j
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i-1] == hyp_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                
    ins, dl, sub = 0, 0, 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_tokens[i-1] == hyp_tokens[j-1]:
            i, j = i-1, j-1
        else:
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                sub += 1
                i, j = i-1, j-1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                dl += 1
                i -= 1
            else:
                ins += 1
                j -= 1
    return ins, dl, sub

def calc_structural_score(ref_tokens, hyp_tokens):
    ref_struct = [t for t in ref_tokens if t in conf.STRUCTURAL_TOKENS]
    hyp_struct = [t for t in hyp_tokens if t in conf.STRUCTURAL_TOKENS]
    if not ref_struct and not hyp_struct: 
        return 1.0
    if not ref_struct: 
        return 0.0
    ed = nltk.edit_distance(ref_struct, hyp_struct)
    return max(0.0, 1.0 - (ed / len(ref_struct)))

def ast_match_score(gt, pred, gt_tokens, pred_tokens):
    if not SYMPY_AVAILABLE:
        return calc_structural_score(gt_tokens, pred_tokens)
    try:
        ast_ref = parse_latex(gt)
        ast_hyp = parse_latex(pred)
        return 1.0 if str(ast_ref) == str(ast_hyp) else 0.0
    except Exception:
        return calc_structural_score(gt_tokens, pred_tokens)