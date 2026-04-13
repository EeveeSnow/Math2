import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import re

from image_processing import RandomWidth, ResizePadHW


PAD = 0
SOS = 1
EOS = 2


class MathWritingDataset(Dataset):
    def __init__(self, hf_dataset, vocab, image_size=(384,512), dataset_part=1):
        limit = int(len(hf_dataset) * dataset_part)
        self.ds = hf_dataset.select(range(limit))
        self.vocab = vocab
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            RandomWidth(),
            ResizePadHW(*image_size),
            transforms.ToTensor()
        ])
        
        raw_latexs = [item["latex"] for item in self.ds]
        self.tokenized_captions = encode_batch(raw_latexs, self.vocab) 


    def __len__(self):
        return len(self.ds)


    def __getitem__(self, idx):
        sample = self.ds[idx]
        image = self.transform(sample["image"])
        caption = self.tokenized_captions[idx] 
        return {
            "image": image,
            "caption": caption
        }
    
def encode_batch(latex_list, vocab, device="cpu"):
    sequences = []

    for latex in latex_list:
        tokens = vocab.tokenize(latex)
        ids = [vocab.stoi.get(t, vocab.stoi["<UNK>"]) for t in tokens]
        seq = [vocab.stoi["<SOS>"]] + ids + [vocab.stoi["<EOS>"]]
        sequences.append(torch.tensor(seq, dtype=torch.long))

    return sequences 


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
        # self._COMMAND_RE = re.compile(r'\\(mathbb{[a-zA-Z]}|begin{[a-z]+}|end{[a-z]+}|operatorname\*|[a-zA-Z]+|.)')
        self._COMMAND_RE = re.compile(
        r'\\mathbb{[a-zA-Z]}'       
        r'|\\mathcal{[a-zA-Z]}'
        r'|\\begin{[a-zA-Z*]+}'
        r'|\\end{[a-zA-Z*]+}'
        r'|\\operatorname\*?'
        r'|\\[a-zA-Z]+'
        r'|\\.'
        # r'|\d+(?:\.\d+)?'
        r'|[^\s]')
    

    def __len__(self) -> int:
        return len(self.itos)

    def tokenize(self, text: str) -> list[str]:
        return self._COMMAND_RE.findall(text)

    def build_vocab(self, sentence_list: list[str]) -> dict[int, str]:
        frequencies = {}
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text: str) -> list[int]:
        tokenized_text = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]
    

image_transform = transforms.Compose([
            transforms.Grayscale(),
            RandomWidth(),
            ResizePadHW(*(384, 384)),
            transforms.ToTensor()
        ])

def decode_tokens(vocab_obj, token_ids):
    itos = vocab_obj
    latex = ""
    
    for tid in token_ids:
        tid = tid.item() if hasattr(tid, 'item') else int(tid)
        
        if tid == 2:
            break
            
        if tid not in [0, 1, 3]:
            token_str = str(itos.get(tid, ""))
            

            if token_str.startswith("\\"):
                latex += token_str + " "
            else:
                latex += token_str
    
    latex = re.sub(r'(\\[a-zA-Z]+)\s+([^a-zA-Z])', r'\1\2', latex)
    
    latex = re.sub(r'(\\[a-zA-Z]+)\s+([^a-zA-Z])', r'\1\2', latex)
    
    return latex.strip()

def predict_latex(image, model, DEVICE, vocab):
    if image is None:
        return "", ""
    
    
    if DEVICE == torch.device("cuda"):
        img_tensor = image_transform(image).unsqueeze(0).to(torch.bfloat16).cuda()
    else:
        img_tensor = image_transform(image).unsqueeze(0).to(DEVICE)

    predictions = model.generate_beam_search(
        images=img_tensor, 
        start_token_id=1,
        eos_token_id=2,
        beam_size=5,
        max_new_tokens=256)
    latex_str = [decode_tokens(vocab, predictions[0][0]), decode_tokens(vocab, predictions[0][1]), decode_tokens(vocab, predictions[0][2])]
    print(latex_str)
    rendered_math = f"$$ {latex_str[0]} $$\n$$ {latex_str[1]} $$\n$$ {latex_str[2]} $$"
    latex_str = '\n'.join(latex_str)
    return latex_str, rendered_math