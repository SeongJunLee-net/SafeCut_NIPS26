import torch
import torch.nn.functional as F
def compute_iic_loss(probs_a: torch.Tensor, probs_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    n = probs_a.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=probs_a.device)
    P   = torch.mm(probs_a.t(), probs_b) / n   
    P   = (P + P.t()) / 2                       
    P   = P.clamp(min=eps)
    P_a = P.sum(dim=1, keepdim=True)            
    P_b = P.sum(dim=0, keepdim=True)            
    mi  = (P * (torch.log(P) - torch.log(P_a) - torch.log(P_b))).sum()
    return -mi
def compute_diversity_loss(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p_mean = probs.mean(dim=0)          
    p_mean = p_mean.clamp(min=eps)
    return (p_mean * p_mean.log()).sum()  