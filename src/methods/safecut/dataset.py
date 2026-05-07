import os.path as osp
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
DATASET_CONFIG = {
    "office-home": {
        "data_folder": "office-home",                    
        "pretrained_folder": "office-home",              
        "class_num": 65,
        "res_name": "resnet50",
        "domain_files": {
            "a": "Art_list",
            "c": "Clipart_list",
            "p": "Product_list",
            "r": "RealWorld_list",
        },
        "domain_dirs": {"a": "A", "c": "C", "p": "P", "r": "R"},
        "tasks": [
            "a2c", "a2p", "a2r",
            "c2a", "c2p", "c2r",
            "p2a", "p2c", "p2r",
            "r2a", "r2c", "r2p",
        ],
    },
    "office-31": {
        "data_folder": "office",
        "pretrained_folder": "office",
        "class_num": 31,
        "res_name": "resnet50",
        "domain_files": {
            "a": "amazon_list",
            "d": "dslr_list",
            "w": "webcam_list",
        },
        "domain_dirs": {"a": "A", "d": "D", "w": "W"},
        "tasks": ["a2d", "a2w", "d2a", "d2w", "w2a", "w2d"],
    },
    "domainnet126": {
        "data_folder": "domainnet126",
        "pretrained_folder": "domainnet126",
        "class_num": 126,
        "res_name": "resnet50",
        "domain_files": {
            "c": "clipart_list",
            "p": "painting_list",
            "r": "real_list",
            "s": "sketch_list",
        },
        "domain_dirs": {"c": "C", "p": "P", "r": "R", "s": "S"},
        "tasks": [
            "c2p", "c2r", "c2s",
            "p2c", "p2r", "p2s",
            "r2c", "r2p", "r2s",
            "s2c", "s2p", "s2r",
        ],
    },
    "visda-c": {
        "data_folder": "VISDA-C",
        "pretrained_folder": "VISDA-C",
        "class_num": 12,
        "res_name": "resnet101",
        "domain_files": {
            "t": "train_list",
            "v": "validation_list",
        },
        "domain_dirs": {"t": "T"},
        "tasks": ["t2v"],
    },
}
def image_target(resize_size: int = 256, crop_size: int = 224):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
def image_test(resize_size: int = 256, crop_size: int = 224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ),
    ])
def make_dataset(image_list, labels):
    if labels:
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len(image_list))]
    else:
        if len(image_list[0].split()) > 2:
            images = [
                (val.split()[0], np.array([int(la) for la in val.split()[1:]]))
                for val in image_list
            ]
        else:
            images = [
                (val.split()[0], int(val.split()[1]))
                for val in image_list
            ]
    return images
def rgb_loader(path: str):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
class ImageListTri(Dataset):
    def __init__(self, image_list, transform_weak=None, transform_clip=None):
        self.imgs           = make_dataset(image_list, None)
        self.transform_weak = transform_weak
        self.transform_clip = transform_clip
        self.loader         = rgb_loader
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img          = self.loader(path)
        img_weak     = self.transform_weak(img) if self.transform_weak else img
        img_clip     = self.transform_clip(img) if self.transform_clip else img
        return img_weak, img_clip, target, index
    def __len__(self):
        return len(self.imgs)
def get_dataloaders(args, clip_preprocess):
    cfg = DATASET_CONFIG[args.dataset]
    _, tt = args.dset.split('2')                    
    data_dir = f"./data/{cfg['data_folder']}"
    t_fname  = cfg["domain_files"][tt]
    t_file   = osp.join(data_dir, f"{t_fname}.txt")
    with open(t_file) as f:
        image_list = f.readlines()
    train_dataset = ImageListTri(
        image_list,
        transform_weak=image_target(),
        transform_clip=clip_preprocess,
    )
    test_dataset = ImageListTri(
        image_list,
        transform_weak=image_test(),
        transform_clip=clip_preprocess,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.worker,
    )
    return train_dataset, test_loader
def get_weight_dir(args) -> str:
    cfg    = DATASET_CONFIG[args.dataset]
    ss, _  = args.dset.split('2')                  
    src_dir = cfg["domain_dirs"].get(ss, ss.upper())
    return osp.join("pretrained_model", cfg["pretrained_folder"], src_dir)