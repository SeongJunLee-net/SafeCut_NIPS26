import os
import logging
from PIL import Image
from torch.utils.data import Dataset
from typing import Sequence, Callable, Optional
from torchvision import transforms
import numpy as np
logger = logging.getLogger(__name__)
class ImageList(Dataset):
    def __init__(
        self,
        image_root: str,
        label_files: Sequence[str],
        transform: Optional[Callable] = None
    ):
        self.image_root = image_root
        self.label_files = label_files
        self.transform = transform
        self.samples = []
        for file in label_files:
            self.samples += self.build_index(label_file=file)
    def build_index(self, label_file):
        with open(label_file, "r") as fd:
            lines = fd.readlines()
        lines = [line.strip() for line in lines if line]
        item_list = []
        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(self.image_root, img_file)
            domain = img_file.split(os.sep)[0]
            item_list.append((img_path, int(label), domain))
        return item_list
    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label,idx
    def __len__(self):
        return len(self.samples)
class ImageList_idx_aug_fix(Dataset):
    def __init__(
        self,
        image_root: str,
        label_files: Sequence[str],
        transform: Optional[Callable] = None
    ):
        self.image_root = image_root
        self.label_files = label_files
        self.transform = transform
        resize_size = 256 
        crop_size = 224 
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                   std=[0.26862954, 0.26130258, 0.27577711])
        self.rf_1 = transforms.Compose([
            transforms.Resize(224, interpolation=Image.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
                normalize
            ])
        self.samples = []
        for file in label_files:
            self.samples += self.build_index(label_file=file)
    def build_index(self, label_file):
        with open(label_file, "r") as fd:
            lines = fd.readlines()
        lines = [line.strip() for line in lines if line]
        item_list = []
        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(self.image_root, img_file)
            domain = img_file.split(os.sep)[0]
            item_list.append((img_path, int(label), domain))
        return item_list
    def __getitem__(self, idx):
        img_path, label, domain = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img_1 = self.transform(img)
            img_2 = self.rf_1(img)
        return img_1,img_2, label,idx
    def __len__(self):
        return len(self.samples)