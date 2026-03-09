import torch
from torch.nn.utils.rnn import pad_sequence
from image_processing import RandomWidth, ResizePadHW
from torchvision import transforms


PAD = 0
SOS = 1
EOS = 2
MAX_LEN = 150    

image_transform = transforms.Compose([
            transforms.Grayscale(),
            RandomWidth(),
            ResizePadHW(*(384, 384)),
            transforms.ToTensor()
        ])


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
    pos = model.pos_encoder(H, W, device)
    memory = memory + pos

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
        best = max(candidates, key=lambda x: x[1] / (len(x[0]) ** length_penalty))[0]
        results.append(best[1:])

    return pad_sequence(results, batch_first=True, padding_value=PAD)


def decode_tokens(vocab_obj, token_ids):
    itos = vocab_obj
    latex = []
    for tid in token_ids:
        tid = tid.item() if hasattr(tid, 'item') else int(tid)
        
        if tid == 2:
            break
            
        if tid not in [0, 1, 3]: 
            token_str = itos.get(tid, "")
            latex.append(str(token_str))
            
    return " ".join(latex)

def predict_latex(image, model, DEVICE, vocab):
    if image is None:
        return "", ""
    
    img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)
    prediction = beam_search(model, img_tensor, beam_size=3)
    print(prediction)
    latex_str = decode_tokens(vocab, prediction[0])
    print(latex_str)
    rendered_math = f"$$ {latex_str} $$"
    
    return latex_str, rendered_math