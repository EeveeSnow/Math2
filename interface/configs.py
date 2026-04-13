import torch
from torchvision import transforms
from image_processing import RandomWidth, ResizePadHW

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_obj = None
vocab = None
VOCAB_SIZE = 0
ARCH = None
model = None


TOP_K = 2
NUM_EXPERTS = 8
routing_history = {}


STRUCTURAL_TOKENS = {
    '^', '_', '{', '}', '\\frac', '\\sqrt', '\\sum', '\\int', 
    '\\left', '\\right', '(', ')', '[', ']', '\\begin', '\\end'
}

BUCKET_KEYS = ["<10", "10-19", "20-29", "30-39", "40-49", "50-59", ">60"]

image_transform = transforms.Compose([
    transforms.Grayscale(),
    RandomWidth(),
    ResizePadHW(*(384, 384)),
    transforms.ToTensor()
])