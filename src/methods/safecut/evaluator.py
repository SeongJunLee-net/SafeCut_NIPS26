import torch
import torch.nn.functional as F
from .prompt import encode_text_with_prompt
@torch.no_grad()
def evaluate_clip_zeroshot(clip_model, prompt_learner, loader, device) -> float:
    clip_model.eval()
    text_features = F.normalize(
        encode_text_with_prompt(clip_model, prompt_learner), dim=-1
    )
    logit_scale = clip_model.logit_scale.exp()
    correct, total = 0, 0
    for _, img_clip, labels, __ in loader:
        img_clip, labels = img_clip.to(device), labels.to(device)
        img_features = F.normalize(
            clip_model.encode_image(img_clip).float(), dim=-1
        )
        logits   = logit_scale * (img_features @ text_features.t())
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total * 100.0