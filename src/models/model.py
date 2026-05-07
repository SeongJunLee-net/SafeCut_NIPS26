import logging
import timm
import torch
import torch.nn as nn
import torchvision.models as models
import os.path as osp
from robustbench.model_zoo.architectures.utils_architectures import normalize_model, ImageNormalizer
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from collections import OrderedDict
from copy import deepcopy
from src.models import resnet26
from src.data.datasets.imagenet_subsets import IMAGENET_A_MASK, IMAGENET_R_MASK, IMAGENET_D109_MASK,IMAGENET_V_MASK
logger = logging.getLogger(__name__)
def get_torchvision_model(model_name, weight_version="IMAGENET1K_V1"):
    name_to_weights = {name[:-8].lower(): name for name in dir(models) if "Weights" in name}
    if not model_name in name_to_weights.keys():
        raise ValueError(f"Model name '{model_name}' is not supported. Choose from: {name_to_weights.keys()}")
    model_weights = getattr(models, name_to_weights[model_name])
    available_weight_versions = [version for version in dir(model_weights) if "IMAGENET1K" in version]
    if not weight_version in available_weight_versions:
        raise ValueError(f"Weight type '{weight_version}' is not supported. Choose from: {available_weight_versions}")
    model_weights = getattr(model_weights, weight_version)
    model = getattr(models, model_name)
    model = model(weights=model_weights)
    return model
def get_timm_model(model_name):
    available_models = timm.list_models(pretrained=True)
    if not model_name in available_models:
        raise ValueError(f"Model '{model_name}' is not available. Choose from: {available_models}")
    model = timm.create_model(model_name, pretrained=True)
    logger.info(f"Successfully restored the weights of '{model_name}' from timm.")
    if hasattr(model, "pretrained_cfg"):
        logger.info(f"General model information: {model.pretrained_cfg}")
        logger.info(f"Adding input normalization to the model using: mean={model.pretrained_cfg['mean']} \t std={model.pretrained_cfg['std']}")
        model = normalize_model(model, mean=model.pretrained_cfg["mean"], std=model.pretrained_cfg["std"])
    else:
        raise AttributeError(f"Attribute 'pretrained_cfg' is missing for model '{model_name}' from timm."
                             f" This prevents adding the correct input normalization to the model!")
    return model
class ResNetDomainNet126(torch.nn.Module):
    def __init__(self, arch="resnet50", checkpoint_path=None, num_classes=126, bottleneck_dim=256):
        super().__init__()
        self.arch = arch
        self.bottleneck_dim = bottleneck_dim
        self.weight_norm_dim = 0
        if not self.use_bottleneck:
            model = models.__dict__[self.arch](pretrained=True)
            modules = list(model.children())[:-1]
            self.encoder = torch.nn.Sequential(*modules)
            self._output_dim = model.fc.in_features
        else:
            model = models.__dict__[self.arch](pretrained=True)
            model.fc = torch.nn.Linear(model.fc.in_features, self.bottleneck_dim)
            bn = torch.nn.BatchNorm1d(self.bottleneck_dim)
            self.encoder = torch.nn.Sequential(model, bn)
            self._output_dim = self.bottleneck_dim
        self.fc = torch.nn.Linear(self.output_dim, num_classes)
        if self.use_weight_norm:
            self.fc = torch.nn.utils.weight_norm(self.fc, dim=self.weight_norm_dim)
        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)
        else:
            logger.warning(f"No checkpoint path was specified. Continue with ImageNet pre-trained weights!")
        self.encoder = nn.Sequential(ImageNormalizer((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), self.encoder)
    def forward(self, x, return_feats=False):
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)
        logits = self.fc(feat)
        if return_feats:
            return feat, logits
        return logits
    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        model_state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint.keys() else checkpoint["model"]
        for name, param in model_state_dict.items():
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )
    def get_params(self):
        backbone_params = []
        extra_params = []
        if not self.use_bottleneck:
            backbone_params.extend(self.encoder.parameters())
        else:
            resnet = self.encoder[1][0]
            for module in list(resnet.children())[:-1]:
                backbone_params.extend(module.parameters())
            extra_params.extend(resnet.fc.parameters())
            extra_params.extend(self.encoder[1][1].parameters())
            extra_params.extend(self.fc.parameters())
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]
        return backbone_params, extra_params
    @property
    def num_classes(self):
        return self.fc.weight.shape[0]
    @property
    def output_dim(self):
        return self._output_dim
    @property
    def use_bottleneck(self):
        return self.bottleneck_dim > 0
    @property
    def use_weight_norm(self):
        return self.weight_norm_dim >= 0
class BaseModel(torch.nn.Module):
    def __init__(self, model, arch_name, dataset_name):
        super().__init__()
        self.encoder, self.fc = split_up_model(model, arch_name=arch_name, dataset_name=dataset_name)
        if isinstance(self.fc, nn.Sequential):
            for module in self.fc.modules():
                if isinstance(module, nn.Linear):
                    self._num_classes = module.oup_features
                    self._output_dim = module.in_features
        elif isinstance(self.fc, nn.Linear):
            self._num_classes = self.fc.oup_features
            self._output_dim = self.fc.in_features
        else:
            raise ValueError("Unable to detect output dimensions")
    def forward(self, x, return_feats=False):
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)
        logits = self.fc(feat)
        if return_feats:
            return feat, logits
        return logits
    @property
    def num_classes(self):
        return self._num_classes
    @property
    def output_dim(self):
        return self._output_dim
class ImageNetXMaskingLayer(torch.nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
    def forward(self, x):
        return x[:, self.mask]
class ImageNetXWrapper(torch.nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.__dict__ = model.__dict__.copy()
        self.masking_layer = ImageNetXMaskingLayer(mask)
    def forward(self, x):
        fea = self.netF(x)
        logits = self.netC(fea)
        return self.masking_layer(logits)
class ImageNetXWrapper_all(torch.nn.Module):
    def __init__(self, model, mask):
        super().__init__()
        self.__dict__ = model.__dict__.copy()
        self.masking_layer = ImageNetXMaskingLayer(mask)
    def forward(self, x):
        logits = self.model(x)
        return self.masking_layer(logits)
class TransformerWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.__dict__ = model.__dict__.copy()
    def forward(self, x):
        x = self.normalize(x)
        x = self.model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.model.encoder(x)
        x = x[:, 0]
        return x
def get_model(cfg, num_classes):
    if cfg.SETTING.DATASET == "domainnet126":
        cfg.output_dir_src = osp.join(cfg.output_dir_src,'best_' + cfg.domain[cfg.SETTING.S] +'_2020.pth')
        base_model = ResNetDomainNet126(arch=cfg.MODEL.ARCH, checkpoint_path=cfg.output_dir_src, num_classes=num_classes)
    else:
        base_model = get_torchvision_model(cfg.MODEL.ARCH, weight_version=cfg.MODEL.WEIGHTS)
        layers = OrderedDict([
        ('model', base_model)
            ])
        base_model = nn.Sequential(layers)
        if cfg.SETTING.DATASET == "imagenet_a":
            base_model = ImageNetXWrapper_all(base_model, IMAGENET_A_MASK)
        elif cfg.SETTING.DATASET == "imagenet_r":
            base_model = ImageNetXWrapper_all(base_model, IMAGENET_R_MASK)
        elif cfg.SETTING.DATASET == "imagenet_v":
            base_model = ImageNetXWrapper_all(base_model, IMAGENET_V_MASK)
    return base_model.cuda()
def split_up_model(model, arch_name, dataset_name):
    if hasattr(model, "model") and hasattr(model.model, "pretrained_cfg") and hasattr(model.model, model.model.pretrained_cfg["classifier"]):
        classifier = deepcopy(getattr(model.model, model.model.pretrained_cfg["classifier"]))
        encoder = model
        encoder.model.reset_classifier(0)
        if isinstance(model, ImageNetXWrapper):
            encoder = nn.Sequential(encoder.normalize, encoder.model)
    elif arch_name == "Standard" and dataset_name in {"cifar10", "cifar10_c"}:
        encoder = nn.Sequential(*list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_WRN":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:-1], nn.AvgPool2d(kernel_size=8, stride=8), nn.Flatten())
        classifier = model.fc
    elif arch_name == "Hendrycks2020AugMix_ResNeXt":
        normalization = ImageNormalizer(mean=model.mu, std=model.sigma)
        encoder = nn.Sequential(normalization, *list(model.children())[:2], nn.ReLU(), *list(model.children())[2:-1], nn.Flatten())
        classifier = model.classifier
    elif dataset_name == "domainnet126":
        encoder = model.encoder
        classifier = model.fc
    elif "resnet" in arch_name or "resnext" in arch_name or "wide_resnet" in arch_name or arch_name in {"Standard_R50", "Hendrycks2020AugMix", "Hendrycks2020Many", "Geirhos2018_SIN"}:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.Flatten())
        classifier = model.model.fc
    elif "densenet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "efficientnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool, nn.Flatten())
        classifier = model.model.classifier
    elif "mnasnet" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.layers, nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.classifier
    elif "shufflenet" in arch_name:
        encoder = nn.Sequential(model.normalize, *list(model.model.children())[:-1], nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten())
        classifier = model.model.fc
    elif "vit_" in arch_name and not "maxvit_" in arch_name:
        encoder = TransformerWrapper(model)
        classifier = model.model.heads.head
    elif "swin_" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.norm, model.model.permute, model.model.avgpool, model.model.flatten)
        classifier = model.model.head
    elif "convnext" in arch_name:
        encoder = nn.Sequential(model.normalize, model.model.features, model.model.avgpool)
        classifier = model.model.classifier
    elif arch_name == "mobilenet_v2":
        encoder = nn.Sequential(model.normalize, model.model.features, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        classifier = model.model.classifier
    else:
        raise ValueError(f"The model architecture '{arch_name}' is not supported for dataset '{dataset_name}'.")
    if dataset_name == "imagenet_a":
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(IMAGENET_A_MASK))
    elif dataset_name == "imagenet_r":
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(IMAGENET_R_MASK))
    elif dataset_name == "imagenet_d109":
        classifier = nn.Sequential(classifier, ImageNetXMaskingLayer(IMAGENET_D109_MASK))
    return encoder, classifier