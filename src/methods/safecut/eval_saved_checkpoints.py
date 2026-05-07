import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import clip
from src.models.network import ResBase, feat_bottleneck, feat_classifier
from .dataset import DATASET_CONFIG, get_dataloaders
from .prompt import PromptLearner, encode_text_with_prompt
DATASET_ALIASES = {
    "office-home": "office-home",
    "office_home": "office-home",
    "officehome": "office-home",
    "office31": "office-31",
    "office-31": "office-31",
    "office_31": "office-31",
    "domainnet": "domainnet126",
    "domainnet126": "domainnet126",
    "visda": "visda-c",
    "visda-c": "visda-c",
    "visdac": "visda-c",
    "all": "all",
}
@dataclass
class CheckpointSet:
    dataset: str
    dset: str
    subdir: str
    target_model_f: Path
    target_model_b: Path
    target_model_c: Path
    peer_model_prompt: Path
def normalize_dataset_name(name: str) -> str:
    key = name.strip().lower().replace(" ", "").replace("_", "-")
    if key in DATASET_ALIASET:
        return DATASET_ALIASES[key]
    key_alt = key.replace("-", "_")
    if key_alt in DATASET_ALIASET:
        return DATASET_ALIASES[key_alt]
    valid = ", ".join(["office-home", "office-31", "domainnet126", "visda-c", "all"])
    raise ValueError(f"Unsupported dataset '{name}'. Valid: {valid}")
def parse_dset_filter(raw: str):
    value = raw.strip().lower()
    if value == "all":
        return None
    return {v.strip().lower() for v in value.split(",") if v.strip()}
def load_classnames(repo_root: Path, dataset: str):
    cfg = DATASET_CONFIG[dataset]
    path = repo_root / "data" / cfg["data_folder"] / "classname.txt"
    with open(path, "r") as f:
        classnames = [line.strip().replace("_", " ") for line in f if line.strip()]
    if len(classnames) != cfg["class_num"]:
        raise ValueError(
            f"classname count mismatch for {dataset}: "
            f"{len(classnames)} vs class_num={cfg['class_num']}"
        )
    return classnames
def _state_numel(state_dict: dict) -> int:
    total = 0
    for value in state_dict.values():
        if torch.is_tensor(value):
            total += int(value.numel())
    return total
def _infer_bottleneck_dim(state_b: dict) -> int:
    if "bottleneck.weight" in state_b:
        return int(state_b["bottleneck.weight"].shape[0])
    if "bn.weight" in state_b:
        return int(state_b["bn.weight"].shape[0])
    raise KeyError("Could not infer bottleneck dim from target_model_B state_dict")
def discover_checkpoint_sets(
    checkpoint_root: Path,
    dataset: str,
    dset_filter,
    recursive: bool,
    dedupe_by_task: bool,
):
    dataset_root = checkpoint_root / dataset
    if not dataset_root.is_dir():
        return []
    pattern = "**/*_target_model_F.pt" if recursive else "*_target_model_F.pt"
    candidates = []
    for f_path in dataset_root.glob(pattern):
        stem = f_path.name
        if not stem.endswith("_target_model_F.pt"):
            continue
        dset = stem[: -len("_target_model_F.pt")]
        if dset_filter is not None and dset.lower() not in dset_filter:
            continue
        b_path = f_path.with_name(f"{dset}_target_model_B.pt")
        c_path = f_path.with_name(f"{dset}_target_model_C.pt")
        p_path = f_path.with_name(f"{dset}_peer_model_prompt.pt")
        if not (b_path.exists() and c_path.exists() and p_path.exists()):
            continue
        rel_subdir = f_path.parent.relative_to(dataset_root)
        subdir = "." if str(rel_subdir) == "." else str(rel_subdir)
        depth = 0 if subdir == "." else len(rel_subdir.parts)
        mtime = f_path.stat().st_mtime
        candidates.append(
            {
                "set": CheckpointSet(
                    dataset=dataset,
                    dset=dset,
                    subdir=subdir,
                    target_model_f=f_path,
                    target_model_b=b_path,
                    target_model_c=c_path,
                    peer_model_prompt=p_path,
                ),
                "depth": depth,
                "mtime": mtime,
            }
        )
    if not dedupe_by_task:
        return [x["set"] for x in sorted(candidates, key=lambda z: (z["set"].dset, z["depth"]))]
    best = {}
    for item in candidates:
        key = item["set"].dset.lower()
        if key not in best:
            best[key] = item
            continue
        old = best[key]
        if item["depth"] < old["depth"] or (
            item["depth"] == old["depth"] and item["mtime"] > old["mtime"]
        ):
            best[key] = item
    return [best[k]["set"] for k in sorted(best.keys())]
def build_target_model_models(dataset: str, class_num: int, bottleneck_dim: int, device):
    res_name = DATASET_CONFIG[dataset].get("res_name", "resnet50")
    net_f = ResBase(res_name=res_name).to(device)
    net_b = feat_bottleneck(
        type="bn",
        feature_dim=net_f.in_features,
        bottleneck_dim=bottleneck_dim,
    ).to(device)
    net_c = feat_classifier(
        type="wn",
        class_num=class_num,
        bottleneck_dim=bottleneck_dim,
    ).to(device)
    return net_f, net_b, net_c
@torch.no_grad()
def evaluate_target_model_peer_model(
    net_f,
    net_b,
    net_c,
    clip_model,
    prompt_learner,
    loader,
    class_num: int,
    device,
    max_batches: int = 0,
):
    net_f.eval()
    net_b.eval()
    net_c.eval()
    clip_model.eval()
    prompt_learner.eval()
    text_features = F.normalize(encode_text_with_prompt(clip_model, prompt_learner), dim=-1)
    logit_scale = clip_model.logit_scale.exp()
    correct_t = 0
    correct_p = 0
    total = 0
    target_correct_clip_correct = 0
    target_wrong_clip_correct = 0
    target_correct_clip_wrong = 0
    target_wrong_clip_wrong = 0
    all_labels = []
    all_pred_t = []
    all_pred_p = []
    for batch_idx, (img_weak, img_clip, labels, _) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        img_weak = img_weak.to(device)
        img_clip = img_clip.to(device)
        labels = labels.to(device)
        t_probs = F.softmax(net_c(net_b(net_f(img_weak))), dim=1)
        p_feat = F.normalize(clip_model.encode_image(img_clip).float(), dim=-1)
        p_probs = F.softmax(logit_scale * (p_feat @ text_features.t()), dim=1)
        t_pred = t_probs.argmax(dim=1)
        p_pred = p_probs.argmax(dim=1)
        t_ok = t_pred == labels
        p_ok = p_pred == labels
        correct_t += (t_pred == labels).sum().item()
        correct_p += (p_pred == labels).sum().item()
        total += labels.size(0)
        target_correct_clip_correct += (t_ok & p_ok).sum().item()
        target_wrong_clip_correct += ((~t_ok) & p_ok).sum().item()
        target_correct_clip_wrong += (t_ok & (~p_ok)).sum().item()
        target_wrong_clip_wrong += ((~t_ok) & (~p_ok)).sum().item()
        all_labels.append(labels.cpu())
        all_pred_t.append(t_pred.cpu())
        all_pred_p.append(p_pred.cpu())
    labels_np = torch.cat(all_labels).numpy()
    pred_t_np = torch.cat(all_pred_t).numpy()
    pred_p_np = torch.cat(all_pred_p).numpy()
    per_class_t = []
    per_class_p = []
    for cls_idx in range(class_num):
        mask = labels_np == cls_idx
        if mask.sum() > 0:
            per_class_t.append(float((pred_t_np[mask] == cls_idx).mean() * 100.0))
            per_class_p.append(float((pred_p_np[mask] == cls_idx).mean() * 100.0))
        else:
            per_class_t.append(float("nan"))
            per_class_p.append(float("nan"))
    return {
        "target_model_acc": (correct_t / total * 100.0) if total > 0 else 0.0,
        "peer_model_acc": (correct_p / total * 100.0) if total > 0 else 0.0,
        "target_model_mean_per_class": float(np.nanmean(per_class_t)),
        "peer_model_mean_per_class": float(np.nanmean(per_class_p)),
        "num_samples": total,
        "target_correct_clip_correct": target_correct_clip_correct,
        "target_wrong_clip_correct": target_wrong_clip_correct,
        "target_correct_clip_wrong": target_correct_clip_wrong,
        "target_wrong_clip_wrong": target_wrong_clip_wrong,
        "total": (
            target_correct_clip_correct
            + target_wrong_clip_correct
            + target_correct_clip_wrong
            + target_wrong_clip_wrong
        ),
    }
def evaluate_one_checkpoint_set(
    ckpt: CheckpointSet,
    repo_root: Path,
    clip_model,
    clip_preprocess,
    args,
    device,
):
    cfg = DATASET_CONFIG[ckpt.dataset]
    class_num = cfg["class_num"]
    classnames = load_classnames(repo_root, ckpt.dataset)
    state_f = torch.load(ckpt.target_model_f, map_location="cpu")
    state_b = torch.load(ckpt.target_model_b, map_location="cpu")
    state_c = torch.load(ckpt.target_model_c, map_location="cpu")
    state_p = torch.load(ckpt.peer_model_prompt, map_location="cpu")
    bottleneck_dim = _infer_bottleneck_dim(state_b)
    net_f, net_b, net_c = build_target_model_models(
        dataset=ckpt.dataset,
        class_num=class_num,
        bottleneck_dim=bottleneck_dim,
        device=device,
    )
    prompt_learner = PromptLearner(classnames, clip_model).to(device)
    net_f.load_state_dict(state_f, strict=args.strict)
    net_b.load_state_dict(state_b, strict=args.strict)
    net_c.load_state_dict(state_c, strict=args.strict)
    prompt_learner.load_state_dict(state_p, strict=args.strict)
    loader_args = SimpleNamespace(
        dataset=ckpt.dataset,
        dset=ckpt.dset,
        batch_size=args.batch_size,
        worker=args.worker,
    )
    _, test_loader = get_dataloaders(loader_args, clip_preprocess)
    eval_stats = evaluate_target_model_peer_model(
        net_f=net_f,
        net_b=net_b,
        net_c=net_c,
        clip_model=clip_model,
        prompt_learner=prompt_learner,
        loader=test_loader,
        class_num=class_num,
        device=device,
        max_batches=args.max_batches,
    )
    f_keys = len(state_f)
    b_keys = len(state_b)
    c_keys = len(state_c)
    p_keys = len(state_p)
    f_numel = _state_numel(state_f)
    b_numel = _state_numel(state_b)
    c_numel = _state_numel(state_c)
    p_numel = _state_numel(state_p)
    return {
        "dataset": ckpt.dataset,
        "dset": ckpt.dset,
        "checkpoint_subdir": ckpt.subdir,
        "class_num": class_num,
        "bottleneck_dim": bottleneck_dim,
        **eval_stats,
        "target_model_F_state_keys": f_keys,
        "target_model_B_state_keys": b_keys,
        "target_model_C_state_keys": c_keys,
        "peer_model_prompt_state_keys": p_keys,
        "total_state_keys": f_keys + b_keys + c_keys + p_keys,
        "target_model_F_state_numel": f_numel,
        "target_model_B_state_numel": b_numel,
        "target_model_C_state_numel": c_numel,
        "peer_model_prompt_state_numel": p_numel,
        "total_state_numel": f_numel + b_numel + c_numel + p_numel,
        "target_model_F_path": str(ckpt.target_model_f),
        "target_model_B_path": str(ckpt.target_model_b),
        "target_model_C_path": str(ckpt.target_model_c),
        "peer_model_prompt_path": str(ckpt.peer_model_prompt),
        "status": "ok",
        "error": "",
    }
def save_rows_csv(csv_path: Path, rows):
    fieldnames = [
        "dataset",
        "dset",
        "checkpoint_subdir",
        "class_num",
        "bottleneck_dim",
        "num_samples",
        "target_model_acc",
        "peer_model_acc",
        "target_model_mean_per_class",
        "peer_model_mean_per_class",
        "target_correct_clip_correct",
        "target_wrong_clip_correct",
        "target_correct_clip_wrong",
        "target_wrong_clip_wrong",
        "total",
        "target_model_F_state_keys",
        "target_model_B_state_keys",
        "target_model_C_state_keys",
        "peer_model_prompt_state_keys",
        "total_state_keys",
        "target_model_F_state_numel",
        "target_model_B_state_numel",
        "target_model_C_state_numel",
        "peer_model_prompt_state_numel",
        "total_state_numel",
        "target_model_F_path",
        "target_model_B_path",
        "target_model_C_path",
        "peer_model_prompt_path",
        "status",
        "error",
    ]
    csv_path.parent.mkdir(parents=True, exisp_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate saved checkpoints and save accuracy + last-state counts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="office-home | office-31 | domainnet126 | visda-c | all",
    )
    parser.add_argument(
        "--dset",
        type=str,
        default="all",
        help="Task filter: all or comma-separated list (e.g., a2c,c2p,t2v)",
    )
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--batch_size", type=int, default=192)
    parser.add_argument("--worker", type=int, default=8)
    parser.add_argument("--clip_model", type=str, default="ViT-B/32")
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="result/checkpoints",
        help="Root directory containing dataset checkpoint folders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="result/checkpoint_eval",
        help="Directory to save evaluation CSV",
    )
    parser.add_argument(
        "--no_recursive",
        action="store_true",
        help="Only search dataset root, not nested subfolders",
    )
    parser.add_argument(
        "--keep_duplicates",
        action="store_true",
        help="Do not dedupe same task across multiple subfolders",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict=True for state_dict loading",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=0,
        help="Debug option. If >0, evaluate only first N batches per task.",
    )
    return parser
def main():
    args = build_parser().parse_args()
    dataset_key = normalize_dataset_name(args.dataset)
    selected_datasets = (
        ["office-home", "office-31", "domainnet126", "visda-c"]
        if dataset_key == "all"
        else [dataset_key]
    )
    dset_filter = parse_dset_filter(args.dset)
    repo_root = Path(__file__).resolve().parents[3]
    checkpoint_root = (repo_root / args.checkpoint_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    clip_model, _, clip_preprocess = clip.load(args.clip_model, device=device)
    clip_model = clip_model.float().eval()
    print("\n" + "=" * 84)
    print(" Evaluate Saved Checkpoints (Target_Model + Peer_Model Prompt)")
    print(f" Device          : {device}")
    print(f" Dataset         : {selected_datasets}")
    print(f" Task filter     : {args.dset}")
    print(f" Checkpoint root : {checkpoint_root}")
    print("=" * 84 + "\n")
    all_rows = []
    for dataset in selected_datasets:
        ckpt_sets = discover_checkpoint_sets(
            checkpoint_root=checkpoint_root,
            dataset=dataset,
            dset_filter=dset_filter,
            recursive=not args.no_recursive,
            dedupe_by_task=not args.keep_duplicates,
        )
        print(f"[{dataset}] discovered {len(ckpt_sets)} checkpoint set(s)")
        for ckpt in ckpt_sets:
            print(f"  - evaluating {ckpt.dset} (subdir={ckpt.subdir})")
            try:
                row = evaluate_one_checkpoint_set(
                    ckpt=ckpt,
                    repo_root=repo_root,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                    args=args,
                    device=device,
                )
            except Exception as exc:
                row = {
                    "dataset": ckpt.dataset,
                    "dset": ckpt.dset,
                    "checkpoint_subdir": ckpt.subdir,
                    "class_num": "",
                    "bottleneck_dim": "",
                    "num_samples": "",
                    "target_model_acc": "",
                    "peer_model_acc": "",
                    "target_model_mean_per_class": "",
                    "peer_model_mean_per_class": "",
                    "target_correct_clip_correct": "",
                    "target_wrong_clip_correct": "",
                    "target_correct_clip_wrong": "",
                    "target_wrong_clip_wrong": "",
                    "total": "",
                    "target_model_F_state_keys": "",
                    "target_model_B_state_keys": "",
                    "target_model_C_state_keys": "",
                    "peer_model_prompt_state_keys": "",
                    "total_state_keys": "",
                    "target_model_F_state_numel": "",
                    "target_model_B_state_numel": "",
                    "target_model_C_state_numel": "",
                    "peer_model_prompt_state_numel": "",
                    "total_state_numel": "",
                    "target_model_F_path": str(ckpt.target_model_f),
                    "target_model_B_path": str(ckpt.target_model_b),
                    "target_model_C_path": str(ckpt.target_model_c),
                    "peer_model_prompt_path": str(ckpt.peer_model_prompt),
                    "status": "error",
                    "error": str(exc),
                }
                print(f"    -> ERROR: {exc}")
            else:
                print(
                    "    -> "
                    f"S={row['target_model_acc']:.2f}% | "
                    f"T={row['peer_model_acc']:.2f}% | "
                    f"CC={row['target_correct_clip_correct']} "
                    f"WC={row['target_wrong_clip_correct']} "
                    f"CW={row['target_correct_clip_wrong']} "
                    f"WW={row['target_wrong_clip_wrong']} | "
                    f"LastStates(sample)={row['total']} | "
                    f"ModelState(keys)={row['total_state_keys']} | "
                    f"ModelState(numel)={row['total_state_numel']}"
                )
            all_rows.append(row)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"checkpoint_eval_{now}.csv"
    save_rows_csv(csv_path, all_rows)
    ok_rows = [r for r in all_rows if r.get("status") == "ok"]
    print("\n" + "=" * 84)
    print(f"Saved CSV: {csv_path}")
    print(f"Total rows: {len(all_rows)} | Success: {len(ok_rows)} | Error: {len(all_rows) - len(ok_rows)}")
    if ok_rows:
        mean_s = np.mean([float(r["target_model_acc"]) for r in ok_rows])
        mean_t = np.mean([float(r["peer_model_acc"]) for r in ok_rows])
        print(f"Mean Target_Model Acc: {mean_s:.2f}%")
        print(f"Mean Peer_Model Acc: {mean_t:.2f}%")
    print("=" * 84)
if __name__ == "__main__":
    main()