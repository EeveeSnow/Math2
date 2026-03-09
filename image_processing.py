from PIL import Image
import random
import torchvision.transforms.functional as TF


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