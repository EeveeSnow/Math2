import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import re

from image_processing import RandomWidth, ResizePadHW


class MathWritingDataset(Dataset):
    def __init__(self, hf_dataset, image_size=(384,512), dataset_part=1):
        limit = int(len(hf_dataset) * dataset_part)
        self.ds = hf_dataset.select(range(limit))

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            RandomWidth(),
            ResizePadHW(*image_size),
            transforms.ToTensor()
        ])


    def __len__(self):
        return len(self.ds)


    def __getitem__(self, idx):
        sample = self.ds[idx]
        image = self.transform(sample["image"])
        latex = sample["latex"]
        return {
            "image": image,
            "latex": latex
        }
    
def encode_batch(latex_list, vocab, device="cpu"):
    sequences = []

    for latex in latex_list:
        tokens = vocab.tokenize(latex)
        ids = [vocab.stoi.get(t, vocab.stoi["<UNK>"]) for t in tokens]
        seq = [vocab.stoi["<SOS>"]] + ids + [vocab.stoi["<EOS>"]]
        sequences.append(torch.tensor(seq, dtype=torch.long))

    batch_tensor = pad_sequence(sequences, batch_first=True, padding_value=vocab.stoi["<PAD>"])
    return batch_tensor.to(device)


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        # Better tokenization for LaTeX: split on spaces, but keep commands together
        tokens = re.findall(r'\\[a-zA-Z]+|[a-zA-Z]+|\d+|[{}()=+\-*/^_]', text)
        return tokens

    def build_vocab(self, sentence_list):
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

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]