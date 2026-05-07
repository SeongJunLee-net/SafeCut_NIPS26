import os
import os.path as osp
import numpy as np
from collections import defaultdict
class DiagnosticLogger:
    def __init__(self, n_classes: int, log_file: str = None):
        self.n_classes = n_classes
        self.history   = defaultdict(list)
        self.epoch     = 0
        self.log_file  = log_file
        if log_file:
            os.makedirs(osp.dirname(log_file), exist_ok=True)
            self.fout = open(log_file, 'w')
        else:
            self.fout = None
    def _log(self, msg: str):
        print(msg)
        if self.fout:
            self.fout.write(msg + '\n')
            self.fout.flush()
    def close(self):
        if self.fout:
            self.fout.close()
            self.fout = None
    def log_agreement(self, t_pred, p_pred, true_labels=None):
        tp_agree = (t_pred == p_pred).mean()
        self.history['tp_agree'].append(float(tp_agree))
        self._log(f"  [A] Agreement | S-P: {tp_agree:.3f}")
        if true_labels is not None:
            t_acc      = (t_pred == true_labels).mean()
            p_acc      = (p_pred == true_labels).mean()
            both_wrong = ((t_pred != true_labels) & (p_pred != true_labels)).mean()
            self._log(
                f"  [A] PseudoAcc | T: {t_acc:.3f}  P: {p_acc:.3f}"
                f"  BothWrong: {both_wrong:.3f}"
            )
    def log_class_distribution(self, t_pred, p_pred, n_classes: int):
        def pred_entropy(pred):
            counts = np.bincount(pred, minlength=n_classes).astype(float)
            p = counts / counts.sum()
            return -(p * np.log(p + 1e-8)).sum()
        max_entropy  = np.log(n_classes)
        t_ratio      = pred_entropy(t_pred) / max_entropy
        t_top5_share = (
            np.sort(np.bincount(t_pred, minlength=n_classes))[-5:].sum()
            / len(t_pred)
        )
        self.history['t_pred_entropy_ratio'].append(float(t_ratio))
        self._log(
            f"  [C] ClassDist | Entropy(S): {t_ratio:.3f}"
            f"  Top5Share(S): {t_top5_share:.3f}"
        )
    def log_calibration(self, reliability, pred, true_labels):
        hi_mask = reliability > 0.8
        if hi_mask.sum() > 0:
            hi_acc = (pred[hi_mask] == true_labels[hi_mask]).mean()
            self._log(f"  [D] Calibration (r>0.8) acc={hi_acc:.3f}")
    def log_target_entropy(self, p_P2T, p_T2P):
        def ent(p):
            return -(p * np.log(p + 1e-8)).sum(axis=1).mean()
        self._log(
            f"  [E] TargetEnt | P2T: {ent(p_P2T):.4f}  T2P: {ent(p_T2P):.4f}"
        )
    def log_weight_distribution(self, w_P2T, w_T2P):
        def stats(w, name):
            return (
                f"{name}: mean={w.mean():.3f}  std={w.std():.3f}"
                f"  min={w.min():.3f}  max={w.max():.3f}"
            )
        self._log(
            f"  [F] WeightDist | {stats(w_P2T,'P2T')} | {stats(w_T2P,'T2P')}"
        )
    def log_loss_balance(
        self, L_P2T, L_iic_t, L_T2P, alpha,
        L_div_t=0.0, L_div_p=0.0,
        L_temporal_t=0.0, L_temporal_p=0.0,
    ):
        self._log(
            f"  [G] Loss | P2T:{L_P2T:.3f}(a={alpha:.1f})  "
            f"IIC:{L_iic_t:.3f}  T2P:{L_T2P:.3f}(a={alpha:.1f})  "
            f"DivT:{L_div_t:.4f}  DivP:{L_div_p:.4f}  "
            f"TempT:{L_temporal_t:.4f}  TempP:{L_temporal_p:.4f}"
        )
    def log_robustness_metrics(self, t_probs, p_probs):
        t_conf = t_probs.max(axis=1).mean()
        p_conf = p_probs.max(axis=1).mean()
        self._log(
            f"  [H] Robustness | Mean Conf T: {t_conf:.3f} | P: {p_conf:.3f}"
        )
    def log_adaptive_k_analysis(self, topk_sim, t_pred, labels, r_target_model_raw, n_classes):
        import torch
        with torch.no_grad():
            local_mean = topk_sim.mean(dim=1, keepdim=True)
            valid_cnt  = (topk_sim >= local_mean).float().sum(dim=1)
            total_k    = topk_sim.shape[1]
        vn = valid_cnt.cpu().numpy()
        self._log(
            f"  [AK] ValidNeighbors | mean_K={vn.mean():.1f}  std={vn.std():.1f}"
            f"  min={vn.min():.0f}  max={vn.max():.0f}  (K={total_k})"
        )
    def log_diversity(self, L_div_t, L_div_p):
        self._log(
            f"  [DIV] DiversityLoss | Target_Model: {L_div_t:.4f}  Peer_Model: {L_div_p:.4f}"
        )
    def epoch_summary(self, epoch: int):
        self.epoch = epoch
        self._log(f"{'─'*60}\n")