import os
import warnings
warnings.filterwarnings("ignore")
import csv
import time
import random
import argparse
import os.path as osp
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import clip
from .logger import DiagnosticLogger
from .prompt import PromptLearner
from .dataset import DATASET_CONFIG, get_dataloaders, get_weight_dir
from .evaluator import evaluate_clip_zeroshot
from .trainer import SafeCutTrainer
def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def load_classnames(dataset: str) -> list:
    cfg       = DATASET_CONFIG[dataset]
    path      = osp.join("./data", cfg["data_folder"], "classname.txt")
    with open(path) as f:
        names = [line.strip().replace('\r', '') for line in f if line.strip()]
    return names
def save_result_csv(
    result_dir: str,
    dataset: str,
    dset: str,
    seed: int,
    zt_acc: float,
    best_parget_model: float,
    best_peer_model: float,
    last_parget_model: float,
    last_peer_model: float,
    best_pc_parget_model: float = float('nan'),
    best_pc_peer_model: float = float('nan'),
    last_pc_parget_model: float = float('nan'),
    last_pc_peer_model: float = float('nan'),
) -> None:
    os.makedirs(result_dir, exist_ok=True)
    csv_path = osp.join(result_dir, f"{dataset}_results.csv")
    header   = [
        'timestamp', 'dset', 'seed',
        'zt_acc', 'best_parget_model', 'best_peer_model',
        'last_parget_model', 'last_peer_model',
        'best_pc_parget_model', 'best_pc_peer_model',
        'last_pc_parget_model', 'last_pc_peer_model',
    ]
    def _fmt(v):
        return f"{v:.2f}" if v == v else 'nan'  
    row = [
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        dset,
        seed,
        _fmt(zt_acc),
        _fmt(best_parget_model),
        _fmt(best_peer_model),
        _fmt(last_parget_model),
        _fmt(last_peer_model),
        _fmt(best_pc_parget_model),
        _fmt(best_pc_peer_model),
        _fmt(last_pc_parget_model),
        _fmt(last_pc_peer_model),
    ]
    write_header = True
    if osp.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            existing_header = next(csv.reader(f), None)
        if existing_header == header:
            write_header = False
        else:
            backup_path = csv_path.replace('.csv', '_backup.csv')
            os.replace(csv_path, backup_path)
            print(f"  [CSV]  Header mismatch -> Backup old file to {backup_path} ")
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
    print(f"  [CSV]  {csv_path}  →  {dset} Row appended")
def train(args):
    device = torch.device(f"cuda:{args.gpu_id}")
    clip_model, _, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.float()
    train_dataset, test_loader = get_dataloaders(args, clip_preprocess)
    def build_train_loader():
        return DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.worker,
            drop_last=False,      
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(args.seed),
        )
    train_loader = build_train_loader()
    classnames     = load_classnames(args.dataset)
    prompt_learner = PromptLearner(classnames, clip_model).to(device)
    print("=" * 80)
    print(f"[Phase 0] CLIP Zero-Shot | {args.dataset} | {args.dset}")
    zt_acc = evaluate_clip_zeroshot(clip_model, prompt_learner, test_loader, device)
    print(f"  => {zt_acc:.2f}%")
    print("=" * 80)
    trainer = SafeCutTrainer(
        args=args, device=device,
        clip_model=clip_model, prompt_learner=prompt_learner,
    )
    if args.weight_batch_size > 0:
        print("  [Loader] note: --weight_batch_size is ignored. warmup_cache reuses train_loader.")
    print(f"  [Loader] train_bs={args.batch_size}  workers={args.worker}")
    trainer.warmup_cache(train_loader)
    train_loader = build_train_loader()
    print("\n" + "=" * 80)
    print(f"[Phase 1] SafeCut | {args.dataset} | {args.dset}")
    print(f"  Alpha(ratio power): {args.fixed_alpha} | Rel-EMA m: {args.safecut_ema_m_start:.3f} -> {args.safecut_ema_m_end:.3f} (cosine)")
    print(f"  Robust: WD(1e-3) | AdamW")
    if args.target_model_lr_schedule == 'flat_cosine':
        print(
            f"  Target_Model LR: schedule=flat_cosine  base={args.lr:.2e}  "
            f"flat_until={args.target_model_decay_start_ratio:.2f}  "
            f"floor_ratio={args.target_model_lr_floor_ratio:.2f}"
        )
    else:
        print(f"  Target_Model LR: schedule=inv  base={args.lr:.2e}")
    print(f"  Peer_Model LR: base={args.prompt_lr:.2e}  floor={args.peer_model_lr_floor:.2e}")
    print(f"  Temporal KL: weight={args.temporal_weight} (previous-prediction consistency)")
    print(f"  DivLoss: div_s={args.div_weight_t}  div_t={args.div_weight_p}")
    print("=" * 80 + "\n")
    best_t = best_p = 0.0
    last_t = last_p = 0.0       
    best_pc_t = best_pc_p = 0.0  
    last_pc_t = last_pc_p = float('nan')  
    last_per_class_t = last_per_class_p = None  
    max_iter = args.epochs * len(train_loader)
    iter_num = 0
    knn_times   = []   
    train_times = []   
    eval_times  = []   
    for epoch in range(args.epochs):
        epoch_stats        = defaultdict(float)
        epoch_labels_list  = []   
        epoch_indices_list = []   
        _t0 = time.time()
        for img_weak, _, labels, indices in train_loader:   
            img_weak = img_weak.to(device)
            indices  = indices.to(device)
            epoch_labels_list.append(labels)             
            epoch_indices_list.append(indices.cpu())     
            iter_num += 1
            batch = trainer.train_step(img_weak, indices, iter_num, max_iter)
            for k, v in batch.items():
                epoch_stats[k] += v
            epoch_stats['batches'] += 1
        train_times.append(time.time() - _t0)
        if (epoch + 1) % args.update_interval == 0 and epoch_labels_list:
            all_idx_cat    = torch.cat(epoch_indices_list)
            all_lbl_cat    = torch.cat(epoch_labels_list)
            sort_order     = torch.argsort(all_idx_cat)
            sorted_indices = all_idx_cat[sort_order].to(device)
            sorted_labels  = all_lbl_cat[sort_order].to(device)
            _t0    = time.time()
            stats  = trainer.update_weights_from_cache(sorted_labels, epoch, sorted_indices)
            knn_times.append(time.time() - _t0)
            best_t = max(best_t, stats['target_model_acc'])
            best_p = max(best_p, stats['peer_model_acc'])
            last_t = stats['target_model_acc']
            last_p = stats['peer_model_acc']
            n = max(epoch_stats['batches'], 1)
            trainer.diag.log_loss_balance(
                epoch_stats['L_P2T']        / n,
                epoch_stats['L_iic_t']      / n,
                epoch_stats['L_T2P']        / n,
                args.fixed_alpha,
                L_div_t=epoch_stats['L_div_t'] / n,
                L_div_p=epoch_stats['L_div_p'] / n,
                L_temporal_t=epoch_stats['L_temporal_t'] / n,
                L_temporal_p=epoch_stats['L_temporal_p'] / n,
            )
            trainer.diag.log_diversity(
                epoch_stats['L_div_t'] / n,
                epoch_stats['L_div_p'] / n,
            )
            print(
                f"Ep {epoch+1:03d}/{args.epochs} | "
                f"EMA m:{stats['ema_m']} | "
                f"EMA Rel T:{stats['r_target_model_mean_ema']} P:{stats['r_peer_model_mean_ema']} | "
                f"Cut T:{stats['cut_parget_model_mean']} P:{stats['cut_peer_model_mean']} | "
                f"Gate P2P:{epoch_stats['g_P2T']/n:.3f} S2P:{epoch_stats['g_T2P']/n:.3f} | "
                f"KNN(cache) {knn_times[-1]:.1f}s\n"
                f"          L: P2T={epoch_stats['L_P2T']/n:.3f}  "
                f"IIC={epoch_stats['L_iic_t']/n:.3f}  "
                f"T2P={epoch_stats['L_T2P']/n:.3f}  "
                f"TempS={epoch_stats['L_temporal_t']/n:.4f}  "
                f"TempT={epoch_stats['L_temporal_p']/n:.4f}  "
                f"DivS={epoch_stats['L_div_t']/n:.4f}  "
                f"DivT={epoch_stats['L_div_p']/n:.4f}\n"
                f"          Acc T:{stats['target_model_acc']:.2f} P:{stats['peer_model_acc']:.2f} || "
                f"Best T:{best_t:.2f} P:{best_p:.2f}"
            )
        if (epoch + 1) % args.eval_interval == 0:
            _t0 = time.time()
            ev   = trainer.evaluate(test_loader)
            eval_times.append(time.time() - _t0)
            best_t = max(best_t, ev['target_model_acc'])
            best_p = max(best_p, ev['peer_model_acc'])
            last_t = ev['target_model_acc']
            last_p = ev['peer_model_acc']
            best_pc_t = max(best_pc_t, ev['target_model_mean_per_class'])
            best_pc_p = max(best_pc_p, ev['peer_model_mean_per_class'])
            last_pc_t = ev['target_model_mean_per_class']
            last_pc_p = ev['peer_model_mean_per_class']
            last_per_class_t = ev['target_model_per_class']
            last_per_class_p = ev['peer_model_per_class']
            print(
                f"          [Eval] T:{ev['target_model_acc']:.2f}(pc:{ev['target_model_mean_per_class']:.2f}) "
                f"P:{ev['peer_model_acc']:.2f}(pc:{ev['peer_model_mean_per_class']:.2f}) || "
                f"Best T:{best_t:.2f} P:{best_p:.2f} | Eval {eval_times[-1]:.1f}s"
            )
    if args.epochs % args.eval_interval != 0:
        _t0 = time.time()
        ev  = trainer.evaluate(test_loader)
        eval_times.append(time.time() - _t0)
        best_t = max(best_t, ev['target_model_acc'])
        best_p = max(best_p, ev['peer_model_acc'])
        last_t = ev['target_model_acc']
        last_p = ev['peer_model_acc']
        best_pc_t = max(best_pc_t, ev['target_model_mean_per_class'])
        best_pc_p = max(best_pc_p, ev['peer_model_mean_per_class'])
        last_pc_t = ev['target_model_mean_per_class']
        last_pc_p = ev['peer_model_mean_per_class']
        last_per_class_t = ev['target_model_per_class']
        last_per_class_p = ev['peer_model_per_class']
    avg_knn   = sum(knn_times)   / len(knn_times)   if knn_times   else 0.0
    avg_train = sum(train_times) / len(train_times) if train_times else 0.0
    avg_eval  = sum(eval_times)  / len(eval_times)  if eval_times  else 0.0
    print("\n" + "=" * 80)
    print(f"[Done] {args.dataset} | {args.dset}")
    print(f"  Zero-Shot : {zt_acc:.2f}%")
    print(f"  Best  T:{best_t:.2f}  P:{best_p:.2f}")
    print(f"  Last  T:{last_t:.2f}  P:{last_p:.2f}")
    print(f"  PerClass Best  T:{best_pc_t:.2f}  P:{best_pc_p:.2f}")
    print(f"  PerClass Last  T:{last_pc_t:.2f}  P:{last_pc_p:.2f}")
    if last_per_class_t is not None:
        classnames = load_classnames(args.dataset)
        pc_str_t = "  ".join(f"{n}:{v:.1f}" for n, v in zip(classnames, last_per_class_t))
        pc_str_p = "  ".join(f"{n}:{v:.1f}" for n, v in zip(classnames, last_per_class_p))
        print(f"  PerClass T: {pc_str_t}")
        print(f"  PerClass P: {pc_str_p}")
    print(f"  ---- Average elapsed time / epoch ----")
    print(f"  KNN (update_weights_from_cache) : {avg_knn:.1f}s  (Total {len(knn_times)}times)")
    print(f"  Train (train_step)              : {avg_train:.1f}s  (Total {len(train_times)}times)")
    print(f"  Eval  (evaluate)                : {avg_eval:.1f}s  (Total {len(eval_times)}times)")
    print(f"  Sum (KNN+Train+Eval): {avg_knn+avg_train+avg_eval:.1f}s/epoch")
    print("=" * 80)
    trainer.diag.close()
    ckpt_dir = osp.join(args.result_dir, "checkpoints", args.dataset)
    save_result_csv(
        result_dir=args.result_dir,
        dataset=args.dataset,
        dset=args.dset,
        seed=args.seed,
        zt_acc=zt_acc,
        best_parget_model=best_t,
        best_peer_model=best_p,
        last_parget_model=last_t,
        last_peer_model=last_p,
        best_pc_parget_model=best_pc_t,
        best_pc_peer_model=best_pc_p,
        last_pc_parget_model=last_pc_t,
        last_pc_peer_model=last_pc_p,
    )
    trainer.save_models(save_dir=ckpt_dir, prefix=args.dset)
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="SafeCut: Source-Free Domain Adaptation via "
                    "Cross-architecture Consistency + Cut Statistics",
    )
    parser.add_argument('--dataset',    type=str, default='office-home',
                        choices=list(DATASET_CONFIG.keys()),
                        help='Dataset name')
    parser.add_argument('--dset',       type=str, default='a2c',
                        help='{src}2{tgt} Domain pair format (e.g. a2c, c2p, t2v)')
    parser.add_argument('--gpu_id',     type=str, default='0',
                        help='GPU ID to use')
    parser.add_argument('--class_num',  type=int, default=None,
                        help='Number of classes (auto set if None)')
    parser.add_argument('--res_name',   type=str, default=None,
                        help='Backbone architecture (auto set if None, e.g. resnet50/resnet101)')
    parser.add_argument('--bottleneck', type=int, default=512)
    parser.add_argument('--layer',      type=str, default='wn',
                        choices=['wn', 'linear'], help='Classifier layer type')
    parser.add_argument('--batch_size',        type=int, default=192)
    parser.add_argument('--weight_batch_size', type=int, default=0,
                        help='deprecated: warmup_cache uses train_loader, so this is ignored')
    parser.add_argument('--worker',            type=int, default=8)
    parser.add_argument('--lr',                type=float, default=1e-2)
    parser.add_argument('--lr_decay_backbone', type=float, default=0.1)
    parser.add_argument('--target_model_lr_schedule', type=str, default='inv',
                        choices=['inv', 'flat_cosine'],
                        help='target_model lr scheduler type')
    parser.add_argument('--target_model_decay_start_ratio', type=float, default=0.3,
                        help='target_model flat+cosine scheduler cosine decay start ratio')
    parser.add_argument('--target_model_lr_floor_ratio', type=float, default=0.0,
                        help='LR floor ratio for target_model flat+cosine scheduler')
    parser.add_argument('--epochs',          type=int, default=50)
    parser.add_argument('--eval_interval',   type=int, default=50)
    parser.add_argument('--update_interval', type=int, default=1)
    parser.add_argument('--seed',            type=int, default=2026)
    parser.add_argument('--k_neighbors',       type=int,   default=50)
    parser.add_argument('--tau_sim',           type=float, default=0.1)
    parser.add_argument('--isolated_baseline', type=float, default=0.1)
    parser.add_argument('--isolated_fallback_regular_knn', action='store_true',
                        help='Fallback to regular k-NN for isolated samples with 0 mutual neighbors')
    parser.add_argument('--fixed_alpha',   type=float, default=1.0,
                        help='Exponent of normalized ratio weight')
    parser.add_argument('--reliability_cmp_beta', type=float, default=5.0,
                        help='Sigmoid beta of comparison reliability gate')
    parser.add_argument('--safecut_ema_m',       type=float, default=0.95,
                        help='Reliability EMA momentum fixed value')
    parser.add_argument('--safecut_ema_m_start', type=float, default=0.7,
                        help='Start value (epoch 0, fast update)')
    parser.add_argument('--safecut_ema_m_end',   type=float, default=0.95,
                        help='EMA momentum end value (last epoch, slow update)')
    parser.add_argument('--iic_weight',    type=float, default=1.3)
    parser.add_argument('--prompt_lr',     type=float, default=1e-2)
    parser.add_argument('--peer_model_lr_floor', '--prompt_lr_floor',
                        dest='peer_model_lr_floor', type=float, default=0.0,
                        help='Absolute lr floor for peer_model cosine scheduler')
    parser.add_argument('--diag_log',      type=str,   default=None,
                        help='Diagnostic log file path (auto set if None)')
    parser.add_argument('--result_dir',    type=str,   default='result',
                        help='Root directory for result CSV and model checkpoints')
    parser.add_argument('--div_weight_t', type=float, default=0.2,
                        help='Student Diversity loss -H(E[p_s]) weight')
    parser.add_argument('--div_weight_p', type=float, default=0.2,
                        help='Teacher Diversity loss -H(E[p_t]) weight')
    parser.add_argument('--temporal_weight', type=float, default=1.0,
                        help='Temporal consistency KL loss weight with previous prediction')
    return parser
def main():
    parser = _build_parser()
    args   = parser.parse_args()
    if args.class_num is None:
        args.class_num = DATASET_CONFIG[args.dataset]['class_num']
    if args.res_name is None:
        args.res_name = DATASET_CONFIG[args.dataset].get('res_name', 'resnet50')
    args.safecut_ema_m       = float(np.clip(args.safecut_ema_m,       0.0, 0.9999))
    args.safecut_ema_m_start = float(np.clip(args.safecut_ema_m_start, 0.0, 0.9999))
    args.safecut_ema_m_end   = float(np.clip(args.safecut_ema_m_end,   0.0, 0.9999))
    args.reliability_cmp_beta = float(max(args.reliability_cmp_beta, 0.0))
    args.target_model_decay_start_ratio = float(np.clip(args.target_model_decay_start_ratio, 0.0, 0.9999))
    args.target_model_lr_floor_ratio = float(np.clip(args.target_model_lr_floor_ratio, 0.0, 1.0))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    args.weight_dir = get_weight_dir(args)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(osp.join(args.result_dir, "checkpoints", args.dataset), exist_ok=True)
    if args.diag_log is None:
        os.makedirs("outer_log", exist_ok=True)
        args.diag_log = osp.join("outer_log", f"{args.dataset}_{args.dset}_diag.log")
    print(f"\n{'='*80}")
    print(f"  dataset   : {args.dataset}")
    print(f"  dset      : {args.dset}")
    print(f"  class_num : {args.class_num}")
    print(f"  weight_dir: {args.weight_dir}")
    print(f"  diag_log  : {args.diag_log}")
    print(f"  target_model_lr_schedule : {args.target_model_lr_schedule}")
    print(f"  target_model_decay_start_ratio : {args.target_model_decay_start_ratio}")
    print(f"  target_model_lr_floor_ratio : {args.target_model_lr_floor_ratio}")
    print(f"  reliability_cmp_beta : {args.reliability_cmp_beta}")
    print(f"  temporal_weight : {args.temporal_weight}")
    print(f"  isolated_fallback_regular_knn : {args.isolated_fallback_regular_knn}")
    print(f"{'='*80}\n")
    train(args)
if __name__ == "__main__":
    main()