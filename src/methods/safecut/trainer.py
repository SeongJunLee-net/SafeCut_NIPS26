import os.path as osp
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.models.network import ResBase, feat_bottleneck, feat_classifier
from .losses import compute_iic_loss, compute_diversity_loss
from .cut_stat import CutStatisticCalculator
from .prompt import PromptLearner, encode_text_with_prompt
from .scheduler import (
    op_copy,
    target_model_lr_scheduler,
    target_model_flat_cosine_scheduler,
    peer_model_lr_scheduler,
)
from .logger import DiagnosticLogger
class SafeCutTrainer:
    def __init__(self, args, device, clip_model, prompt_learner: PromptLearner):
        self.args   = args
        self.device = device
        self.netF = ResBase(res_name=getattr(args, 'res_name', 'resnet50')).to(device)
        self.netB = feat_bottleneck(
            type='bn',
            feature_dim=self.netF.in_features,
            bottleneck_dim=args.bottleneck,
        ).to(device)
        self.netC = feat_classifier(
            type=args.layer,
            class_num=args.class_num,
            bottleneck_dim=args.bottleneck,
        ).to(device)
        self.netF.load_state_dict(torch.load(osp.join(args.weight_dir, 'source_F.pt')))
        self.netB.load_state_dict(torch.load(osp.join(args.weight_dir, 'source_B.pt')))
        self.netC.load_state_dict(torch.load(osp.join(args.weight_dir, 'source_C.pt')))
        self.clip_model     = clip_model
        self.prompt_learner = prompt_learner
        for p in self.clip_model.parameters():
            p.requires_grad = False
        param_group = []
        for _, v in self.netF.named_parameters():
            param_group += [{
                'params': v, 'lr': args.lr * args.lr_decay_backbone,
                'weight_decay': 1e-3, 'momentum': 0.9, 'nesterov': True,
            }]
        for _, v in self.netB.named_parameters():
            param_group += [{
                'params': v, 'lr': args.lr,
                'weight_decay': 1e-3, 'momentum': 0.9, 'nesterov': True,
            }]
        for _, v in self.netC.named_parameters():
            param_group += [{
                'params': v, 'lr': args.lr * 0.1,
                'weight_decay': 1e-3, 'momentum': 0.9, 'nesterov': True,
            }]
        self.target_model_opt = op_copy(optim.SGD(param_group))
        self.peer_model_opt = op_copy(optim.AdamW(
            self.prompt_learner.parameters(),
            lr=args.prompt_lr, betas=(0.9, 0.999), weight_decay=1e-4,
        ))
        self.cut_calc = CutStatisticCalculator(
            k_neighbors=args.k_neighbors,
            tau_sim=args.tau_sim,
            isolated_baseline=args.isolated_baseline,
            isolated_fallback_regular_knn=getattr(
                args, 'isolated_fallback_regular_knn', False
            ),
        )
        self.w_P2T        = None
        self.w_T2P        = None
        self.p_target_P2T = None
        self.p_target_T2P = None
        self.p_feat_cache: torch.Tensor | None = None  
        self.t_feat_cache: torch.Tensor | None = None  
        self.t_prob_cache: torch.Tensor | None = None  
        self.p_prob_cache: torch.Tensor | None = None  
        self.diag = DiagnosticLogger(
            n_classes=args.class_num,
            log_file=getattr(args, 'diag_log', None),
        )
    @staticmethod
    def _set_batchnorm_eval(module: nn.Module) -> None:
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
    @torch.no_grad()
    def warmup_cache(self, loader) -> None:
        import time as _time
        print("  [Warmup] Target_Model + CLIP Caching features and calculating initial reliability...", flush=True)
        _t0 = _time.time()
        self.netF.eval(); self.netB.eval(); self.netC.eval()
        self.clip_model.eval()
        text_features = F.normalize(
            encode_text_with_prompt(self.clip_model, self.prompt_learner), dim=-1
        )
        logit_scale = self.clip_model.logit_scale.exp()
        all_t_feat:  list[torch.Tensor] = []
        all_p_feat:  list[torch.Tensor] = []
        all_t_probs: list[torch.Tensor] = []
        all_p_probs: list[torch.Tensor] = []
        all_indices: list[torch.Tensor] = []
        for img_weak, img_clip, _, indices in loader:
            img_weak  = img_weak.to(self.device)
            img_clip  = img_clip.to(self.device)
            t_feat   = self.netB(self.netF(img_weak))
            t_logits = self.netC(t_feat)
            t_probs  = F.softmax(t_logits, dim=1)
            p_feat   = F.normalize(self.clip_model.encode_image(img_clip).float(), dim=-1)
            p_logits = logit_scale * (p_feat @ text_features.t())
            p_probs  = F.softmax(p_logits, dim=1)
            all_t_feat.append(t_feat.cpu())
            all_p_feat.append(p_feat.cpu())
            all_t_probs.append(t_probs.cpu())
            all_p_probs.append(p_probs.cpu())
            all_indices.append(indices)          
        all_indices_cat = torch.cat(all_indices).long()   
        idx_gpu         = all_indices_cat.to(self.device)
        all_t_feat_cat  = torch.cat(all_t_feat)           
        all_p_feat_cat  = torch.cat(all_p_feat)           
        all_t_probs_cat = torch.cat(all_t_probs)          
        all_p_probs_cat = torch.cat(all_p_probs)          
        N   = int(all_indices_cat.max().item()) + 1
        D_t = all_p_feat_cat.shape[1]
        D_s = all_t_feat_cat.shape[1]
        C   = all_t_probs_cat.shape[1]
        self.p_feat_cache = torch.zeros(N, D_t, device=self.device, dtype=torch.float32)
        self.p_feat_cache[idx_gpu] = all_p_feat_cat.to(self.device)
        self.t_feat_cache = torch.zeros(N, D_s, device=self.device, dtype=torch.float32)
        self.t_feat_cache[idx_gpu] = all_t_feat_cat.to(self.device)
        self.t_prob_cache = torch.zeros(N, C, device=self.device, dtype=torch.float32)
        self.t_prob_cache[idx_gpu] = all_t_probs_cat.to(self.device)
        self.p_prob_cache = torch.zeros(N, C, device=self.device, dtype=torch.float32)
        self.p_prob_cache[idx_gpu] = all_p_probs_cat.to(self.device)
        t_pred = self.t_prob_cache.argmax(dim=1)
        p_pred = self.p_prob_cache.argmax(dim=1)
        r_target_model_raw, _, _, _ = self.cut_calc.compute_cut_and_reliability(
            self.t_feat_cache, t_pred
        )
        r_peer_model_raw, _, _, _ = self.cut_calc.compute_cut_and_reliability(
            self.p_feat_cache, p_pred
        )
        self.w_P2T        = r_peer_model_raw.detach().clone()
        self.w_T2P        = r_target_model_raw.detach().clone()
        self.p_target_P2T = self.p_prob_cache.clone()
        self.p_target_T2P = self.t_prob_cache.clone()
        self.netF.train(); self.netB.train(); self.netC.train()
        t_warmup = _time.time() - _t0
        print(
            f"  [Warmup] Done: N={N}, D_s={D_s}, D_t={D_t}, C={C}, Elapsed={t_warmup:.1f}s\n"
            f"  [Warmup] r_target_model_init={r_target_model_raw.mean():.3f}  "
            f"r_peer_model_init={r_peer_model_raw.mean():.3f}",
            flush=True,
        )
    @torch.no_grad()
    def update_weights_from_cache(self, labels: torch.Tensor, epoch: int,
                                  indices: torch.Tensor | None = None) -> dict:
        import time as _time
        assert self.t_feat_cache is not None and self.p_feat_cache is not None,            "warmup_cache()must be called first."
        t_feat  = self.t_feat_cache   
        p_feat  = self.p_feat_cache   
        t_probs = self.t_prob_cache   
        p_probs = self.p_prob_cache   
        t_pred = t_probs.argmax(dim=1)
        p_pred = p_probs.argmax(dim=1)
        _t_knn_t = _time.time()
        r_target_model_raw, cut_t, degree_s, topk_sim_s =            self.cut_calc.compute_cut_and_reliability(t_feat, t_pred)
        t_knn_t = _time.time() - _t_knn_t
        _t_knn_p = _time.time()
        r_peer_model_raw, cut_p, degree_t, topk_sim_t =            self.cut_calc.compute_cut_and_reliability(p_feat, p_pred)
        t_knn_p = _time.time() - _t_knn_p
        m_start = getattr(self.args, 'safecut_ema_m_start', self.args.safecut_ema_m)
        m_end   = getattr(self.args, 'safecut_ema_m_end',   self.args.safecut_ema_m)
        t       = float(epoch) / max(float(self.args.epochs - 1), 1.0)
        m       = m_start + (m_end - m_start) * (1.0 - np.cos(np.pi * t)) / 2.0
        if self.w_P2T is None or self.w_P2T.shape[0] != r_peer_model_raw.shape[0]:
            self.w_P2T = r_peer_model_raw.detach().clone()
            self.w_T2P = r_target_model_raw.detach().clone()
        else:
            self.w_P2T = m * self.w_P2T + (1.0 - m) * r_peer_model_raw.detach()
            self.w_T2P = m * self.w_T2P + (1.0 - m) * r_target_model_raw.detach()
        self.p_target_P2T = p_probs.detach()
        self.p_target_T2P = t_probs.detach()
        if indices is not None:
            t_pred_eval    = t_pred[indices]
            p_pred_eval    = p_pred[indices]
            topk_sim_s_eval = topk_sim_s[indices]
            w_T2P_eval     = self.w_T2P[indices]
        else:
            t_pred_eval    = t_pred
            p_pred_eval    = p_pred
            topk_sim_s_eval = topk_sim_s
            w_T2P_eval     = self.w_T2P
        t_acc = (t_pred_eval == labels).float().mean().item() * 100.0
        p_acc = (p_pred_eval == labels).float().mean().item() * 100.0
        _t_diag = _time.time()
        print(f"\n{'='*60}\n  DIAGNOSTIC: Epoch {epoch+1}\n{'='*60}")
        self.diag.log_agreement(
            t_pred_eval.cpu().numpy(), p_pred_eval.cpu().numpy(), labels.cpu().numpy()
        )
        self.diag.log_robustness_metrics(
            t_probs.cpu().numpy(), p_probs.cpu().numpy()
        )
        self.diag.log_class_distribution(
            t_pred.cpu().numpy(), p_pred.cpu().numpy(), self.args.class_num
        )
        self.diag.log_calibration(
            w_T2P_eval.cpu().numpy(), t_pred_eval.cpu().numpy(), labels.cpu().numpy()
        )
        self.diag.log_target_entropy(
            self.p_target_P2T.cpu().numpy(), self.p_target_T2P.cpu().numpy()
        )
        self.diag.log_weight_distribution(
            self.w_P2T.cpu().numpy(), self.w_T2P.cpu().numpy()
        )
        self.diag.log_adaptive_k_analysis(
            topk_sim_s_eval, t_pred_eval, labels, r_target_model_raw[indices] if indices is not None else r_target_model_raw, self.args.class_num
        )
        self.diag.epoch_summary(epoch)
        t_diag = _time.time() - _t_diag
        t_total = t_knn_t + t_knn_p + t_diag
        print(
            f"  [TIME][Cache] KNN_T:{t_knn_t:.1f}s | KNN_P:{t_knn_p:.1f}s | "
            f"Diag:{t_diag:.2f}s | Total:{t_total:.1f}s"
        )
        return {
            'target_model_acc':        t_acc,
            'peer_model_acc':        p_acc,
            'N':                  len(t_pred),
            'ema_m':              f"{m:.4f}",
            'r_target_model_mean_raw': f"{r_target_model_raw.mean().item():.3f}",
            'r_peer_model_mean_raw': f"{r_peer_model_raw.mean().item():.3f}",
            'r_target_model_mean_ema': f"{self.w_T2P.mean().item():.3f}",
            'r_peer_model_mean_ema': f"{self.w_P2T.mean().item():.3f}",
            'cut_parget_model_mean':   f"{cut_t.mean().item():.3f}",
            'cut_peer_model_mean':   f"{cut_p.mean().item():.3f}",
        }
    def train_step(
        self,
        img_weak: torch.Tensor,
        indices: torch.Tensor,
        iter_num: int,
        max_iter: int,
    ) -> dict:
        args = self.args
        if getattr(self.args, 'target_model_lr_schedule', 'inv') == 'flat_cosine':
            target_model_flat_cosine_scheduler(
                self.target_model_opt,
                iter_num,
                max_iter,
                decay_start_ratio=getattr(self.args, 'target_model_decay_start_ratio', 0.3),
                lr_floor_ratio=getattr(self.args, 'target_model_lr_floor_ratio', 0.0),
            )
        else:
            target_model_lr_scheduler(self.target_model_opt, iter_num, max_iter)
        peer_model_lr_scheduler(
            self.peer_model_opt,
            iter_num,
            max_iter,
            lr_floor=getattr(self.args, 'peer_model_lr_floor', 0.0),
        )
        self.netF.train(); self.netB.train(); self.netC.train()
        if img_weak.size(0) == 1:
            self.netF.apply(self._set_batchnorm_eval)
            self.netB.apply(self._set_batchnorm_eval)
        w_P2T = self.w_P2T[indices]
        w_T2P = self.w_T2P[indices]
        den       = (w_P2T + w_T2P).clamp_min(1e-8)
        beta      = float(getattr(args, 'reliability_cmp_beta', 5.0))
        g_rel_P2T = (w_P2T / den).pow(args.fixed_alpha)
        g_rel_T2P = (w_T2P / den).pow(args.fixed_alpha)
        g_cmp_P2T = torch.sigmoid(beta * (w_P2T - w_T2P))
        g_cmp_T2P = torch.sigmoid(beta * (w_T2P - w_P2T))
        w_P2T_eff = g_cmp_P2T
        w_T2P_eff = g_cmp_T2P
        t_probs_prev = self.t_prob_cache[indices].detach()
        p_probs_prev = self.p_prob_cache[indices].detach()
        t_feat      = self.netB(self.netF(img_weak))
        t_logits    = self.netC(t_feat)
        t_probs     = F.softmax(t_logits, dim=1)
        t_log_probs = F.log_softmax(t_logits, dim=1)
        text_features = F.normalize(
            encode_text_with_prompt(self.clip_model, self.prompt_learner), dim=-1
        )
        img_features = self.p_feat_cache[indices]  
        logit_scale    = self.clip_model.logit_scale.exp()
        p_logits_train = logit_scale * (img_features @ text_features.t()) 
        p_probs_train  = F.softmax(p_logits_train, dim=1)
        p_log_probs    = F.log_softmax(p_logits_train, dim=1)
        with torch.no_grad():
            self.t_feat_cache[indices] = t_feat.detach()
            self.t_prob_cache[indices] = t_probs.detach()
            self.p_prob_cache[indices] = p_probs_train.detach()
        kl_P2T_per = F.kl_div(t_log_probs, p_probs_train.detach(), reduction='none').sum(dim=1)
        L_P2T      = (w_P2T_eff * kl_P2T_per).mean()
        L_iic_t    = compute_iic_loss(t_probs, p_probs_train.detach())
        L_div_t    = compute_diversity_loss(t_probs)
        L_temporal_t_per = F.kl_div(t_log_probs, t_probs_prev, reduction='none').sum(dim=1)
        L_temporal_t     = (w_T2P_eff * L_temporal_t_per).mean()
        target_model_loss = (
            L_P2T
            + args.iic_weight      * L_iic_t
            + args.div_weight_t    * L_div_t
            + args.temporal_weight * L_temporal_t
        )
        kl_T2P_per = F.kl_div(p_log_probs, t_probs.detach(), reduction='none').sum(dim=1)
        L_T2P      = (w_T2P_eff * kl_T2P_per).mean()
        L_iic_p    = compute_iic_loss(p_probs_train, t_probs.detach())
        L_div_p    = compute_diversity_loss(p_probs_train)
        L_temporal_p_per = F.kl_div(p_log_probs, p_probs_prev, reduction='none').sum(dim=1)
        L_temporal_p     = (w_P2T_eff * L_temporal_p_per).mean()
        peer_model_loss = (
            L_T2P
            + args.iic_weight      * L_iic_p
            + args.div_weight_p    * L_div_p
            + args.temporal_weight * L_temporal_p
        )
        self.target_model_opt.zero_grad()
        target_model_loss.backward()
        self.target_model_opt.step()
        self.peer_model_opt.zero_grad()
        peer_model_loss.backward()
        self.peer_model_opt.step()
        return {
            'target_model_loss': target_model_loss.item(),
            'peer_model_loss': peer_model_loss.item(),
            'L_P2T':        L_P2T.item(),
            'L_iic_t':      L_iic_t.item(),
            'L_T2P':        L_T2P.item(),
            'L_div_t':      L_div_t.item(),
            'L_div_p':      L_div_p.item(),
            'L_temporal_t': L_temporal_t.item(),
            'L_temporal_p': L_temporal_p.item(),
            'g_P2T':        w_P2T_eff.mean().item(),
            'g_T2P':        w_T2P_eff.mean().item(),
        }
    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        prev_modes = {
            'netF': self.netF.training,
            'netB': self.netB.training,
            'netC': self.netC.training,
            'clip_model': self.clip_model.training,
            'prompt_learner': self.prompt_learner.training,
        }
        self.netF.eval(); self.netB.eval(); self.netC.eval()
        self.clip_model.eval()
        self.prompt_learner.eval()
        try:
            text_features = F.normalize(
                encode_text_with_prompt(self.clip_model, self.prompt_learner), dim=-1
            )
            logit_scale = self.clip_model.logit_scale.exp()
            correct_t = correct_p = total = 0
            all_labels, all_pred_t, all_pred_p = [], [], []
            for img_weak, img_clip, labels, __ in loader:
                img_weak = img_weak.to(self.device)
                img_clip = img_clip.to(self.device)
                labels   = labels.to(self.device)
                t_probs = F.softmax(self.netC(self.netB(self.netF(img_weak))), dim=1)
                p_feat  = F.normalize(self.clip_model.encode_image(img_clip).float(), dim=-1)
                p_probs = F.softmax(logit_scale * (p_feat @ text_features.t()), dim=1)
                t_pred = t_probs.argmax(1)
                p_pred = p_probs.argmax(1)
                correct_t += (t_pred == labels).sum().item()
                correct_p += (p_pred == labels).sum().item()
                total     += labels.size(0)
                all_labels.append(labels.cpu())
                all_pred_t.append(t_pred.cpu())
                all_pred_p.append(p_pred.cpu())
            lbl = torch.cat(all_labels).numpy()
            ps  = torch.cat(all_pred_t).numpy()
            pt  = torch.cat(all_pred_p).numpy()
            n_cls = self.args.class_num
            per_class_t, per_class_p = [], []
            for c in range(n_cls):
                mask = (lbl == c)
                if mask.sum() > 0:
                    per_class_t.append(float((ps[mask] == c).mean() * 100.0))
                    per_class_p.append(float((pt[mask] == c).mean() * 100.0))
                else:
                    per_class_t.append(float('nan'))
                    per_class_p.append(float('nan'))
            mean_pc_s = float(np.nanmean(per_class_t))
            mean_pc_t = float(np.nanmean(per_class_p))
            return {
                'target_model_acc':            correct_t / total * 100.0,
                'peer_model_acc':            correct_p / total * 100.0,
                'target_model_per_class':      per_class_t,
                'peer_model_per_class':      per_class_p,
                'target_model_mean_per_class': mean_pc_s,
                'peer_model_mean_per_class': mean_pc_t,
            }
        finally:
            self.netF.train(prev_modes['netF'])
            self.netB.train(prev_modes['netB'])
            self.netC.train(prev_modes['netC'])
            self.clip_model.train(prev_modes['clip_model'])
            self.prompt_learner.train(prev_modes['prompt_learner'])
    def save_models(self, save_dir: str, prefix: str) -> None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.netF.state_dict(),
                   os.path.join(save_dir, f"{prefix}_target_model_F.pt"))
        torch.save(self.netB.state_dict(),
                   os.path.join(save_dir, f"{prefix}_target_model_B.pt"))
        torch.save(self.netC.state_dict(),
                   os.path.join(save_dir, f"{prefix}_target_model_C.pt"))
        torch.save(self.prompt_learner.state_dict(),
                   os.path.join(save_dir, f"{prefix}_peer_model_prompt.pt"))
        print(f"  [Save] models → {save_dir}/{prefix}_*.pt")