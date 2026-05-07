import os
import logging
import random
import numpy as np
import time
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
import pickle
from .Dataset_Idx import Dataset_Idx
from conf import complete_data_dir_path
from src.data.datasets.imagelist_dataset import ImageList, ImageList_idx_aug_fix
from src.data.datasets.imagenet_subsets import create_imagenet_subset, class_mapping_164_to_109
from src.data.datasets.corruptions_datasets import create_cifarc_dataset, create_imagenetc_dataset
from src.data.datasets.imagenet_d_utils import create_symlinks_and_get_imagenet_visda_mapping
from src.data.datasets.imagenet_dict import map_dict
from src.data.augmentations.transforms_adacontrast import get_augmentation_versions, get_augmentation,get_augmentation_versions_image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from .selectedRotateImageFolder import SelectedRotateImageFolder
logger = logging.getLogger(__name__)
def get_transform(dataset_name, adaptation):
    if adaptation == "adacontrast":
        if dataset_name in {"cifar10", "cifar100", "cifar10_c", "cifar100_c"}:
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=32, crop_size=32)
        elif dataset_name == "imagenet_c":
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2-light", res_size=224, crop_size=224)
        elif dataset_name in {"domainnet126"}:
            transform = get_augmentation_versions(aug_versions="twss", aug_type="moco-v2", res_size=256, crop_size=224)
        else:
            transform = get_augmentation_versions_image(aug_versions="imnn", aug_type="moco-v2", res_size=256, crop_size=224)
    else:
        if dataset_name in {"cifar10", "cifar100"}:
            transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name in {"cifar10_c", "cifar100_c"}:
            transform =   transforms.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor(),
        ])
        elif dataset_name == "imagenet_c":
            transform = transforms.Compose([transforms.ToTensor()])
        elif dataset_name in {"domainnet126"}:
            transform = get_augmentation(aug_type="test", res_size=256, crop_size=224)
        else:
            normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
            transform = transforms.Compose([
                transforms.Resize(224, interpolation=BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    return transform
def get_test_loader_p(setting, adaptation, dataset_name, root_dir, domain_name, severity, num_examples,
                    domain_names_all, rng_seed, alpha_dirichlet=0, batch_size=128, shuffle=False, workers=4,trans=None):
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    data_dir = complete_data_dir_path(root=root_dir, dataset_name=dataset_name)
    transform = trans
    if domain_name == "none":
        test_dataset, _ = get_source_loader(dataset_name, root_dir, adaptation, batch_size, train_split=False)
    else:
        if dataset_name in {"cifar10_c", "cifar100_c"}:
            test_dataset = create_cifarc_dataset(dataset_name=dataset_name,
                                                 severity=severity,
                                                 data_dir=data_dir,
                                                 corruption=domain_name,
                                                 corruptions_seq=domain_names_all,
                                                 transform=transform,
                                                 setting=setting)
        elif dataset_name == "imagenet_c":
            ckpt = './ckpt/Datasets'
            teset, _ =  load_imagenet_c(root=os.path.join(root_dir, 'ImageNet-C'), batch_size=batch_size, corruption=domain_name,
                               level=severity, workers=workers,
                               transforms=transform, ckpt=ckpt)
            dataset_idx = Dataset_Idx(teset)
            data_loader = torch.utils.data.DataLoader(dataset_idx, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                                    drop_last=False)
            return data_loader
        elif dataset_name in {"imagenet_k", "imagenet_r", "imagenet_a","imagenet_v"}:
            test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
        elif dataset_name in {"imagenet_d", "imagenet_d109", "domainnet126", "office31", "visda"}:
            if dataset_name in {"imagenet_d", "imagenet_d109"}:
                for dom_name in domain_names_all:
                    if not os.path.exists(os.path.join(data_dir, dom_name)):
                        logger.info(f"Creating symbolical links for ImageNet-D {dom_name}...")
                        domainnet_dir = os.path.join(complete_data_dir_path(root=root_dir, dataset_name="domainnet126"), dom_name)
                        create_symlinks_and_get_imagenet_visda_mapping(domainnet_dir, map_dict)
            if "mixed_domains" in setting:
                data_files = [os.path.join("datasets", f"{dataset_name}_lists", dom_name + "_list.txt") for dom_name in domain_names_all]
            else:
                data_files = [os.path.join("src/data/datasets", f"{dataset_name}_lists", domain_name + "_list.txt")]
            test_dataset = ImageList_idx_aug_fix(image_root=data_dir,
                                     label_files=data_files,
                                     transform=transform)
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported!")
    try:
        if num_examples != -1:
            num_samples_orig = len(test_dataset)
            test_dataset.samples = random.sample(test_dataset.samples, k=min(num_examples, num_samples_orig))
        if "mixed_domains" in setting:
            logger.info(f"Successfully mixed the file paths of the following domains: {domain_names_all}")
        if "correlated" in setting:
            if alpha_dirichlet > 0:
                logger.info(f"Using Dirichlet distribution with alpha={alpha_dirichlet} to temporally correlated samples by class labels...")
                test_dataset.samples = sort_by_dirichlet(alpha_dirichlet, samples=test_dataset.samples)
            else:
                logger.info(f"Sorting the file paths by class labels...")
                test_dataset.samples.sort(key=lambda x: x[1])
    except AttributeError:
        logger.warning("Attribute 'samples' is missing. Continuing without shuffling, sorting or subsampling the files...")
    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)
def get_test_loader(adaptation, dataset_name, root_dir, domain_name, rng_seed, batch_size=128, shuffle=False, workers=4):
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    data_dir = complete_data_dir_path(root=root_dir, dataset_name=dataset_name)
    transform = get_transform(dataset_name, adaptation)
    if domain_name == "none":
        test_dataset, _ = get_source_loader(dataset_name, root_dir, adaptation, batch_size, train_split=False)
    else:
        if dataset_name in {"imagenet_k", "imagenet_r", "imagenet_a","imagenet_v"}:
            test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
        elif dataset_name in {"domainnet126"}:
            data_files = [os.path.join("./data", f"{dataset_name}", domain_name + "_list.txt")]
            test_dataset = ImageList(image_root=data_dir,
                                     label_files=data_files,
                                     transform=transform)
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported!")
    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)
def get_test_loader_aug(adaptation, dataset_name, root_dir, domain_name, rng_seed, batch_size=128, shuffle=False, workers=4):
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    data_dir = complete_data_dir_path(root=root_dir, dataset_name=dataset_name)
    transform = get_transform(dataset_name, adaptation)
    if domain_name == "none":
        test_dataset, _ = get_source_loader(dataset_name, root_dir, adaptation, batch_size, train_split=False)
    else:
        if dataset_name in {"imagenet_k", "imagenet_r", "imagenet_a","imagenet_v"}:
            test_dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform)
        elif dataset_name in {"domainnet126"}:
            data_files = [os.path.join("./data", f"{dataset_name}", domain_name + "_list.txt")]
            test_dataset = ImageList_idx_aug_fix(image_root=data_dir,
                                     label_files=data_files,
                                     transform=transform)
        else:
            raise ValueError(f"Dataset '{dataset_name}' is not supported!")
    return torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers, drop_last=False)
def get_source_loader(dataset_name, root_dir, adaptation, batch_size, train_split=True, ckpt_path=None, num_samples=None, percentage=1.0, workers=4):
    dataset_name = dataset_name.split("_")[0] if dataset_name in {"cifar10_c", "cifar100_c", "imagenet_c", "imagenet_k"} else dataset_name
    data_dir = complete_data_dir_path(root=root_dir, dataset_name=dataset_name)
    transform = get_transform(dataset_name, adaptation)
    if dataset_name == "cifar10":
        source_dataset = torchvision.datasets.CIFAR10(root=root_dir,
                                                      train=train_split,
                                                      download=True,
                                                      transform=transform)
    elif dataset_name == "cifar100":
        source_dataset = torchvision.datasets.CIFAR100(root=root_dir,
                                                       train=train_split,
                                                       download=True,
                                                       transform=transform)
    elif dataset_name == "imagenet":
        split = "train" if train_split else "val"
        source_dataset = torchvision.datasets.ImageNet(root=data_dir,
                                                       split=split,
                                                       transform=transform)
    elif dataset_name in {"domainnet126", "office31", "visda"}:
        src_domain = ckpt_path.replace('.pth', '').split(os.sep)[-1].split('_')[1]
        source_data_list = [os.path.join("datasets", f"{dataset_name}_lists", f"{src_domain}_list.txt")]
        source_dataset = ImageList(image_root=data_dir,
                                   label_files=source_data_list,
                                   transform=transform)
        logger.info(f"Loading source data from list: {source_data_list[0]}")
    elif dataset_name in {"imagenet_r", "imagenet_a", "imagenet_d", "imagenet_d109"}:
        split = "train" if train_split else "val"
        data_dir = complete_data_dir_path(root=root_dir, dataset_name="imagenet")
        source_dataset = create_imagenet_subset(data_dir=data_dir,
                                                dataset_name=dataset_name,
                                                split=split,
                                                transform=transform)
    else:
        raise ValueError("Dataset not supported.")
    if percentage < 1.0 or num_samples:    
        if dataset_name in {"cifar10", "cifar100"}:
            nr_src_samples = source_dataset.data.shape[0]
            nr_reduced = min(num_samples, nr_src_samples) if num_samples else int(np.ceil(nr_src_samples * percentage))
            inds = random.sample(range(0, nr_src_samples), nr_reduced)
            source_dataset.data = source_dataset.data[inds]
            source_dataset.targets = [source_dataset.targets[k] for k in inds]
        else:
            nr_src_samples = len(source_dataset.samples)
            nr_reduced = min(num_samples, nr_src_samples) if num_samples else int(np.ceil(nr_src_samples * percentage))
            source_dataset.samples = random.sample(source_dataset.samples, nr_reduced)
        logger.info(f"Number of images in source loader: {nr_reduced}/{nr_src_samples} \t Reduction factor = {nr_reduced / nr_src_samples:.4f}")
    source_loader = torch.utils.data.DataLoader(source_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=workers,
                                                drop_last=False)
    logger.info(f"Number of images and batches in source loader: #img = {len(source_dataset)} #batches = {len(source_loader)}")
    return source_dataset, source_loader
def sort_by_dirichlet(alpha_dirichlet, samples):
    N = len(samples)
    samples_sorted = []
    class_labels = np.array([val[1] for val in samples])
    num_classes = int(np.max(class_labels) + 1)
    dirichlet_numchunks = num_classes
    time_start = time.time()
    time_duration = 120  
    min_size = -1
    min_size_thresh = 10
    while min_size < min_size_thresh:  
        idx_batch = [[] for _ in range(dirichlet_numchunks)]
        idx_batch_cls = [[] for _ in range(dirichlet_numchunks)]  
        for k in range(num_classes):
            idx_k = np.where(class_labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha_dirichlet, dirichlet_numchunks))
            proportions = np.array([p * (len(idx_j) < N / dirichlet_numchunks) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            for idx_j, idx in zip(idx_batch_cls, np.split(idx_k, proportions)):
                idx_j.append(idx)
        if time.time() > time_start + time_duration:
            raise ValueError(f"Could not correlated sequence using dirichlet value '{alpha_dirichlet}'. Try other value!")
    sequence_stats = []
    for chunk in idx_batch_cls:
        cls_seq = list(range(num_classes))
        np.random.shuffle(cls_seq)
        for cls in cls_seq:
            idx = chunk[cls]
            samples_sorted.extend([samples[i] for i in idx])
            sequence_stats.extend(list(np.repeat(cls, len(idx))))
    return samples_sorted
def load_imagenet_c(root, corruption, transforms, level=5, batch_size=64, workers=4, ckpt=None):
    assert os.path.exists(root), f'Path {root} does not exist'
    assert corruption in ['brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost',
                          'gaussian_blur', 'gaussian_noise', 'glass_blur', 'impulse_noise', 'jpeg_compression',
                          'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow', 'spatter', 'speckle_noise',
                          'zoom_blur'], f'Unknown corruption: {corruption}'
    validdir = os.path.join(root,'noise', corruption, str(level))
    teset = SelectedRotateImageFolder(validdir, transforms, original=False,
                                      rotation=False)
    ckpt_dir = os.path.join(ckpt, 'imagenet_c')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, 'list.pickle')
    if not os.path.exists(ckpt_path):
        idx = torch.randperm(len(teset))
        idx = [i.item() for i in idx]
        with open(ckpt_path, 'wb') as f:
            pickle.dump(idx, f)
    else:
        with open(ckpt_path, 'rb') as f:
            idx = pickle.load(f)
    teset.samples = [teset.samples[i] for i in idx]
    teset.switch_mode(True, False)
    teloader = torch.utils.data.DataLoader(teset, batch_size=batch_size, shuffle=False,
                                           num_workers=workers, pin_memory=True)
    return teset, teloader