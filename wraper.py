from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import torch
import random

class ResizePadHW:
    def __init__(self, target_h=384, target_w=512):
        self.target_h = target_h
        self.target_w = target_w

    def __call__(self, img):

        w, h = img.size

        scale = min(
            self.target_w / w,
            self.target_h / h
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        img = img.resize((new_w, new_h), Image.BILINEAR)

        pad_w = self.target_w - new_w
        pad_h = self.target_h - new_h

        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2
        )

        img = TF.pad(img, padding, fill=255)

        return img
    
class RandomWidth:
    def __call__(self, img):
        scale = random.uniform(0.9, 1.1)
        w, h = img.size
        return img.resize((int(w*scale), h))
    
class MathWritingDataset(Dataset):

    def __init__(self, hf_dataset, image_size=(384,512)):
        self.ds = hf_dataset

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