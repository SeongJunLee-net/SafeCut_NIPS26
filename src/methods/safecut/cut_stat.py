import torch
import torch.nn.functional as F
class CutStatisticCalculator:
    def __init__(
        self,
        k_neighbors: int        = 20,
        tau_sim: float          = 0.1,
        isolated_baseline: float = 0.1,
        isolated_fallback_regular_knn: bool = False,
    ):
        self.k                 = k_neighbors
        self.tau_sim           = tau_sim
        self.isolated_baseline = isolated_baseline
        self.isolated_fallback_regular_knn = isolated_fallback_regular_knn
    def _compute_fractional_cut_gpu(self, features: torch.Tensor, predictions: torch.Tensor):
        with torch.no_grad():
            N   = features.shape[0]
            f   = F.normalize(features, dim=1)
            sim = torch.mm(f, f.t())                     
            k_eff             = min(self.k + 1, N)
            topk_sim, indices = torch.topk(sim, k_eff, dim=1)
            topk_sim          = topk_sim[:, 1:]          
            nbr_idx           = indices[:, 1:]
            adj        = torch.zeros((N, N), dtype=torch.bool, device=features.device)
            adj.scatter_(1, nbr_idx, True)
            mutual_adj  = adj & adj.t()
            mutual_mask = mutual_adj.gather(1, nbr_idx) 
            if self.isolated_fallback_regular_knn:
                isolated_idx = (mutual_mask.sum(dim=1) == 0)
                mutual_mask[isolated_idx, :] = True
            distances = 1.0 - topk_sim
            weights   = (
                torch.exp(-distances / self.tau_sim)
                * mutual_mask.float()
            )
            pred_i        = predictions.unsqueeze(1)      
            pred_j        = predictions[nbr_idx]          
            cut_indicator = (pred_i != pred_j).float()
            cut_tum   = (weights * cut_indicator).sum(dim=1)
            degree    = weights.sum(dim=1)
            cut_ratio = cut_tum / (degree + 1e-8)
        return cut_ratio, degree, topk_sim, nbr_idx, mutual_mask, adj
    def _compute_reliability(self, cut_ratio: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
        r = torch.exp(-cut_ratio)
        r[degree == 0] = self.isolated_baseline
        return r
    def compute_cut_and_reliability(self, features: torch.Tensor, predictions: torch.Tensor):
        cut_ratio, degree, topk_sim, nbr_idx, mutual_mask, _ =            self._compute_fractional_cut_gpu(features, predictions)
        reliability = self._compute_reliability(cut_ratio, degree)
        return reliability, cut_ratio, degree, topk_sim