from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    BEST_METRIC,
    CAPSNET_CFG,
    DETERMINISTIC,
    EPOCHS,
    LEARNING_RATE,
    NUM_WORKERS,
    PIN_MEMORY,
    USE_AMP,
    USE_CLASS_WEIGHTS,
    WEIGHT_DECAY,
    EARLY_STOP,
    EARLY_STOP_PATIENCE,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_MIN_EPOCHS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    IN_CHANNELS,
)
from losses import MarginLoss
from metrics import compute_classification_metrics, compute_confusion
from plotting import save_confusion_matrix_figure
from utils import (
    Timer,
    count_parameters,
    count_flops_thop,
    ensure_dir,
    save_json,
    seed_everything,
)


def _is_capsnet(model_name: str) -> bool:
    return model_name.lower().startswith("capsnet")


def build_dataloaders(
    train_ds: Dataset,
    valid_ds: Dataset,
    test_ds: Dataset,
    seed: int,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    g = torch.Generator()
    g.manual_seed(int(seed))

    persistent = (NUM_WORKERS > 0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        generator=g,
        drop_last=False,
        persistent_workers=persistent,
        prefetch_factor=4 if persistent else None,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=persistent,
        prefetch_factor=4 if persistent else None,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
        persistent_workers=persistent,
        prefetch_factor=4 if persistent else None,
    )

    return train_loader, valid_loader, test_loader


def build_loss_fn(model_name: str, num_classes: int, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    if _is_capsnet(model_name):
        return MarginLoss(
            m_plus=float(CAPSNET_CFG["m_plus"]),
            m_minus=float(CAPSNET_CFG["m_minus"]),
            lambda_=float(CAPSNET_CFG["lambda_"]),
        )

    if class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights)
    return nn.CrossEntropyLoss()


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    num_classes: int,
    class_names: List[str],
    use_amp: bool,
) -> Tuple[float, Dict[str, float], np.ndarray, Dict[str, float]]:
    model.eval()

    losses: List[float] = []
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    device_type = "cuda" if device.type == "cuda" else "cpu"

    use_cuda_events = (device.type == "cuda")
    if use_cuda_events:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        infer_time_ms = 0.0
        n_samples = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device, dtype=torch.long)

        if use_cuda_events:
            starter.record()

        with autocast(device_type=device_type, enabled=use_amp):
            out = model(x)
            loss = loss_fn(out, y)

        if use_cuda_events:
            ender.record()
            torch.cuda.synchronize()
            infer_time_ms += float(starter.elapsed_time(ender))
            n_samples += int(x.size(0))

        losses.append(float(loss.item()))
        pred = out.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
        all_pred.append(pred)
        all_true.append(y.detach().cpu().numpy().astype(np.int64))

    y_true = np.concatenate(all_true, axis=0) if all_true else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0,), dtype=np.int64)

    metrics = compute_classification_metrics(y_true, y_pred, num_classes=num_classes, class_names=class_names)
    cm = compute_confusion(y_true, y_pred, num_classes=num_classes)

    avg_loss = float(np.mean(losses)) if losses else 0.0

    timing: Dict[str, float] = {}
    if use_cuda_events and n_samples > 0:
        infer_time_sec_total = infer_time_ms / 1000.0
        timing["infer_time_sec_total"] = float(infer_time_sec_total)
        timing["infer_time_ms_per_sample"] = float(infer_time_ms / n_samples)
        timing["throughput_samples_per_sec"] = float(n_samples / infer_time_sec_total) if infer_time_sec_total > 0 else 0.0
    else:
        timing["infer_time_sec_total"] = 0.0
        timing["infer_time_ms_per_sample"] = 0.0
        timing["throughput_samples_per_sec"] = 0.0

    return avg_loss, metrics, cm, timing


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: Optional[GradScaler],
    *,
    epoch: int,
    epochs: int,
    num_classes: int,
    class_names: List[str],
) -> Tuple[float, Dict[str, float]]:
    model.train()
    losses: List[float] = []

    device_type = "cuda" if device.type == "cuda" else "cpu"

    correct = 0
    total = 0
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    pbar = tqdm(loader, desc=f"train {epoch}/{epochs}", leave=True, dynamic_ncols=True)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device_type, enabled=use_amp):
            out = model(x)
            loss = loss_fn(out, y)

        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_val = float(loss.item())
        losses.append(loss_val)

        pred = out.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        running_acc = (correct / total) if total > 0 else 0.0

        all_true.append(y.detach().cpu().numpy().astype(np.int64))
        all_pred.append(pred.detach().cpu().numpy().astype(np.int64))

        pbar.set_postfix(
            {
                "loss": f"{np.mean(losses):.4f}",
                "acc": f"{running_acc:.4f}",
            }
        )

    train_loss = float(np.mean(losses)) if losses else 0.0
    y_true = np.concatenate(all_true, axis=0) if all_true else np.zeros((0,), dtype=np.int64)
    y_pred = np.concatenate(all_pred, axis=0) if all_pred else np.zeros((0,), dtype=np.int64)

    train_metrics = compute_classification_metrics(y_true, y_pred, num_classes=num_classes, class_names=class_names)
    return train_loss, train_metrics


def _per_class_acc_from_cm(cm: np.ndarray, class_names: List[str]) -> Dict[str, float]:
    cm = np.asarray(cm)
    row_sum = cm.sum(axis=1).astype(np.float64)
    row_sum[row_sum == 0] = 1.0
    acc = (np.diag(cm).astype(np.float64) / row_sum)
    return {class_names[i]: float(acc[i]) for i in range(len(class_names))}


def run_one_seed(
    *,
    task_name: str,
    model_name: str,
    seed: int,
    num_classes: int,
    class_names: List[str],
    train_ds: Dataset,
    valid_ds: Dataset,
    test_ds: Dataset,
    train_label_counts: Dict[int, int],
    model_builder,
    device: torch.device,
    out_dir: str,
) -> Dict[str, float]:
    ensure_dir(out_dir)

    seed_everything(seed, deterministic=DETERMINISTIC)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, valid_loader, test_loader = build_dataloaders(
        train_ds, valid_ds, test_ds, seed=seed, batch_size=BATCH_SIZE
    )

    class_weights = None
    if USE_CLASS_WEIGHTS:
        counts = np.zeros(num_classes, dtype=np.float64)
        for k, v in train_label_counts.items():
            if 0 <= int(k) < num_classes:
                counts[int(k)] = float(v)
        counts[counts == 0] = 1.0
        weights = counts.sum() / (num_classes * counts)
        class_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)

    model = model_builder(model_name, num_classes).to(device)
    params = count_parameters(model)

    # FLOPs/MACs
    flops_info = count_flops_thop(
        model,
        input_shape=(IN_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),
        device=device,
        batch_size=1,
    )
    if flops_info.get("error"):
        print(f"[WARN] FLOPs calc failed for {model_name}: {flops_info['error']}", flush=True)
    else:
        print(
            f"[INFO] FLOPs({model_name}): gmacs={flops_info.get('gmacs')} gflops={flops_info.get('gflops')}",
            flush=True,
        )

    use_amp = bool(USE_AMP) and (not _is_capsnet(model_name))

    loss_fn = build_loss_fn(model_name, num_classes, class_weights=class_weights)
    optimizer = build_optimizer(model)

    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler(device_type, enabled=use_amp)

    best_score = -1e9
    best_epoch = -1
    best_path = os.path.join(out_dir, "best.pt")

    timer = Timer()
    timer.start()

    history: List[Dict[str, float]] = []

    # -------- Early Stopping state --------
    no_improve = 0
    stop_epoch = 0
    early_stop_triggered = False

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
            use_amp,
            scaler,
            epoch=epoch,
            epochs=EPOCHS,
            num_classes=num_classes,
            class_names=class_names,
        )

        val_loss, val_metrics, _, _ = evaluate(
            model, valid_loader, loss_fn, device, num_classes=num_classes, class_names=class_names, use_amp=use_amp
        )

        score = float(val_metrics.get(BEST_METRIC, val_metrics.get("macro_f1", 0.0)))

        history.append(
            dict(
                epoch=epoch,
                train_loss=float(train_loss),
                val_loss=float(val_loss),
                **{f"train_{k}": float(v) for k, v in train_metrics.items()},
                **{f"val_{k}": float(v) for k, v in val_metrics.items()},
            )
        )

        lr = float(optimizer.param_groups[0]["lr"])
        print(
            f"[EPOCH {epoch:03d}/{EPOCHS}] "
            f"lr={lr:.2e} "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_metrics.get('acc', 0.0):.4f} "
            f"train_macro_f1={train_metrics.get('macro_f1', 0.0):.4f} "
            f"train_weighted_f1={train_metrics.get('weighted_f1', 0.0):.4f} | "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_metrics.get('acc', 0.0):.4f} "
            f"val_macro_f1={val_metrics.get('macro_f1', 0.0):.4f} "
            f"val_weighted_f1={val_metrics.get('weighted_f1', 0.0):.4f} "
            f"best({BEST_METRIC})={best_score:.4f}@{best_epoch}",
            flush=True,
        )

        improved = score > (best_score + (EARLY_STOP_MIN_DELTA if EARLY_STOP else 0.0))

        if improved:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_score": score}, best_path)
            print(f"[INFO] New best at epoch {epoch}: {BEST_METRIC}={score:.4f} -> saved {best_path}", flush=True)
        else:
            if EARLY_STOP and epoch >= EARLY_STOP_MIN_EPOCHS:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    early_stop_triggered = True
                    stop_epoch = epoch
                    print(
                        f"[EARLY STOP] no improvement for {EARLY_STOP_PATIENCE} epochs "
                        f"(best {BEST_METRIC}={best_score:.4f} at epoch {best_epoch}). Stop at epoch {epoch}.",
                        flush=True,
                    )
                    break

    if stop_epoch == 0:
        stop_epoch = epoch  # last epoch actually ran

    train_time_sec = timer.stop()

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_loss, test_metrics, cm, timing = evaluate(
        model, test_loader, loss_fn, device, num_classes=num_classes, class_names=class_names, use_amp=use_amp
    )

    per_class_acc = _per_class_acc_from_cm(cm, class_names)

    save_json(
        {
            "task": task_name,
            "model": model_name,
            "seed": int(seed),
            "num_classes": int(num_classes),
            "params": params,
            "flops": flops_info,
            "best_metric": BEST_METRIC,
            "best_val_score": float(best_score),
            "best_epoch": int(best_epoch),
            "train_time_sec": float(train_time_sec),
            "epochs_ran": int(stop_epoch),
            "early_stop": {
                "enabled": bool(EARLY_STOP),
                "triggered": bool(early_stop_triggered),
                "patience": int(EARLY_STOP_PATIENCE),
                "min_delta": float(EARLY_STOP_MIN_DELTA),
                "min_epochs": int(EARLY_STOP_MIN_EPOCHS),
                "no_improve_count_at_stop": int(no_improve),
            },
            "test_loss": float(test_loss),
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "per_class_acc": per_class_acc,
            "timing": timing,
            "history": history,
        },
        os.path.join(out_dir, "metrics.json"),
    )

    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm.astype(np.int64), fmt="%d", delimiter=",")

    save_confusion_matrix_figure(
        cm,
        class_names=class_names,
        out_path=os.path.join(out_dir, "confusion_matrix_norm.png"),
        title=f"{task_name} | {model_name} | seed={seed} (row-normalized)",
        normalize=True,
        dpi=300,
    )
    save_confusion_matrix_figure(
        cm,
        class_names=class_names,
        out_path=os.path.join(out_dir, "confusion_matrix_raw.png"),
        title=f"{task_name} | {model_name} | seed={seed} (raw counts)",
        normalize=False,
        dpi=300,
    )

    flat: Dict[str, float] = {}
    flat["seed"] = float(seed)
    flat["best_epoch"] = float(best_epoch)
    flat["best_val_score"] = float(best_score)
    flat["train_time_sec"] = float(train_time_sec)
    flat["epochs_ran"] = float(stop_epoch)
    flat["early_stop_triggered"] = float(1.0 if early_stop_triggered else 0.0)
    flat["test_loss"] = float(test_loss)

    flat["params_total"] = float(params["total"])
    flat["params_trainable"] = float(params["trainable"])

    flat["gmacs"] = float(flops_info["gmacs"]) if flops_info.get("gmacs") is not None else float("nan")
    flat["gflops"] = float(flops_info["gflops"]) if flops_info.get("gflops") is not None else float("nan")

    flat["infer_time_sec_total"] = float(timing.get("infer_time_sec_total", 0.0))
    flat["infer_time_ms_per_sample"] = float(timing.get("infer_time_ms_per_sample", 0.0))
    flat["throughput_samples_per_sec"] = float(timing.get("throughput_samples_per_sec", 0.0))

    for k, v in test_metrics.items():
        flat[f"test_{k}"] = float(v)

    return flat
