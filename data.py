import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, ImageEnhance
import os
import xml.etree.ElementTree as ET
import re
import random

# Function to extract LaTeX from INKML
def extract_latex_from_inkml(inkml_file):
    try:
        tree = ET.parse(inkml_file)
        root = tree.getroot()
        ns = {'inkml': 'http://www.w3.org/2003/InkML'}
        annotation = root.find('.//inkml:annotation[@type="truth"]', ns)
        if annotation is not None:
            return annotation.text.strip()
        else:
            return None
    except Exception as e:
        print(f"Error parsing {inkml_file}: {e}")
        return None

# Function to render INKML to image
def inkml_to_image(inkml_file):
    try:
        tree = ET.parse(inkml_file)
        root = tree.getroot()
        ns = {'inkml': 'http://www.w3.org/2003/InkML'}
        all_points = []

        for trace in root.findall('.//inkml:trace', ns):
            coords = trace.text.strip().split(',')
            for coord in coords:
                parts = coord.strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    all_points.append((x, y))

        if not all_points:
            return Image.new('RGB', (224, 224), color='white')

        # Find bounds
        xs, ys = zip(*all_points)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Scale to 224x224 with margin
        width, height = 224, 224
        margin = 10
        scale_x = (width - 2 * margin) / (max_x - min_x) if max_x > min_x else 1
        scale_y = (height - 2 * margin) / (max_y - min_y) if max_y > min_y else 1
        scale = min(scale_x, scale_y)

        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)

        for trace in root.findall('.//inkml:trace', ns):
            points = []
            coords = trace.text.strip().split(',')
            for coord in coords:
                parts = coord.strip().split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    x = margin + (x - min_x) * scale
                    y = margin + (y - min_y) * scale
                    points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill='black', width=2)

        # Data augmentation
        if random.random() < 0.5:
            img = img.rotate(random.uniform(-10, 10))
        if random.random() < 0.3:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))
        if random.random() < 0.3:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(0.8, 1.2))

        return img
    except Exception as e:
        print(f"Error rendering {inkml_file}: {e}")
        return Image.new('RGB', (224, 224), color='white')

class MathDataset(Dataset):
    def __init__(self, root_dir, vocab, transform=None):
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform
        self.samples = []

        # Scan for .inkml files, extract LaTeX from annotation, skip if no truth
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.inkml'):
                    inkml_file = os.path.join(root, file)
                    latex = extract_latex_from_inkml(inkml_file)
                    if latex is not None:
                        self.samples.append((inkml_file, latex))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inkml_file, latex = self.samples[idx]

        # Load image
        image = inkml_to_image(inkml_file)
        if self.transform:
            image = self.transform(image)

        # Use LaTeX directly
        tokens = self.vocab.tokenize(latex)
        numericalized_tokens = [self.vocab.stoi.get(token, self.vocab.stoi["<UNK>"]) for token in tokens]
        caption = [self.vocab.stoi['<SOS>']] + numericalized_tokens + [self.vocab.stoi['<EOS>']]
        caption = torch.tensor(caption)

        return image, caption

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

import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CROHMEDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        image_size=(256, 1024),  # (H, W)
        samples: list[str] | None = None,
    ):
        self.root_dir = Path(root_dir)

        if samples is None:
            self.samples = sorted([
                p.stem.replace(".inkml", "")
                for p in self.root_dir.glob("*.inkml.png")
            ])
        else:
            self.samples = samples

        self.transform = T.Compose([
            T.Resize(image_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])  # grayscale-friendly
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name = self.samples[idx]

        img_path = self.root_dir / f"{name}.inkml.png"
        txt_path = self.root_dir / f"{name}.txt"

        # image
        img = Image.open(img_path).convert("L")  # grayscale
        img = self.transform(img)

        # latex
        with open(txt_path, "r", encoding="utf-8") as f:
            latex = f.readline().strip()

        return {
            "image": img,          # Tensor [1, H, W]
            "latex": latex,        # str
            "id": name
        }
        
        
import torch
from torch.nn.utils.rnn import pad_sequence

def encode_batch(latex_list, vocab, device="cpu"):
    """
    Превращает список LaTeX-строк в тензор с padding:
    [batch, max_len]
    """
    sequences = []

    for latex in latex_list:
        tokens = vocab.tokenize(latex)
        ids = [vocab.stoi.get(t, vocab.stoi["<UNK>"]) for t in tokens]
        seq = [vocab.stoi["<SOS>"]] + ids + [vocab.stoi["<EOS>"]]
        sequences.append(torch.tensor(seq, dtype=torch.long))

    # Padding
    batch_tensor = pad_sequence(sequences, batch_first=True, padding_value=vocab.stoi["<PAD>"])
    return batch_tensor.to(device)



if __name__ == "__main__":
    image = inkml_to_image("image\\105_em_88.inkml")
    image.save("sample_image.png")