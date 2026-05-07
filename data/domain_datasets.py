import math
import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL
from PIL import Image
import re
class BaseJsonDataset(Dataset):
    def __init__(self, json_path, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.split_json = json_path
        self.mode = mode
        self.image_list = []
        self.label_list = []
        txt_tar = open(json_path).readlines()
        samples = []
        for line in txt_tar:
            line=line.rstrip("\n")
            line_split = re.split(' ',line)
            samples.append(line_split)
        for s in samples: 
            self.image_list.append(s[0])
            s[1] = int(s[1])
            self.label_list.append(s[1])
        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        image_path = self.image_list[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label).long()
domain_datasets = ['VISDA-C', 'office', 'office-home','Domain-Net','cifar10_c','cifar100_c','imagenet_c','imagenet_a','imagenet_r','imagenet_k','domainnet126','imagenet_v']
path_dict = {
    "flower102": ["jpg", "data/data_splits/split_zhou_OxfordFlowers.json"],
    "food101": ["images", "data/data_splits/split_zhou_Food101.json"],
    "dtd": ["images", "data/data_splits/split_zhou_DescribableTextures.json"],
    "pets": ["", "data/data_splits/split_zhou_OxfordPets.json"],
    "sun397": ["", "data/data_splits/split_zhou_SUN397.json"],
    "caltech101": ["", "data/data_splits/split_zhou_Caltech101.json"],
    "ucf101": ["", "data/data_splits/split_zhou_UCF101.json"],
    "cars": ["", "data/data_splits/split_zhou_StanfordCars.json"],
    "eurosat": ["", "data/data_splits/split_zhou_EuroSAT.json"]
}
def build_domain_dataset(set_id, root, transform,domain_name, mode='train', n_shot=None):
    if set_id.lower() == 'aircraft':
        return Aircraft(root, mode, n_shot, transform)
    folder = './data/'
    dset_path = folder + set_id + '/' + domain_name + '_list.txt'
    return BaseJsonDataset(dset_path,mode, n_shot, transform)
class Aircraft(Dataset):
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode
        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]
        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))
        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image = Image.open(image_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label).long()