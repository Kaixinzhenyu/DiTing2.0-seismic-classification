from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

import config as cfg
from datasets import build_datasets
from models import build_model
from plotting import save_confusion_matrix_figure
from train_eval import run_one_seed
from utils import ensure_dir, save_json


def summarize_across_seeds(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    Return mean/std/min/max for selected columns.
    """
    rows = []
    for col in metric_cols:
        if col not in df.columns:
            continue
        rows.append(
            dict(
                metric=col,
                mean=float(df[col].mean()),
                std=float(df[col].std(ddof=1)) if len(df) > 1 else 0.0,
                min=float(df[col].min()),
                max=float(df[col].max()),
            )
        )
    return pd.DataFrame(rows)


def _parse_csv_list(s: Optional[str]) -> Optional[List[str]]:
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _apply_preset(mode: str, preset: str) -> Tuple[List[str], List[str]]:
    """
    Presets for your requested GPU split (only for 4class):
      - gpu0: Expert_B + Junior_A, first 4 models
      - gpu1: Expert_C + Junior_A, last 3 models
    """
    if mode != "4class":
        raise ValueError("--preset currently designed for --mode 4class only.")

    tasks_gpu0 = ["4class_noise_Expert_B", "4class_noise_Junior_A"]
    tasks_gpu1 = ["4class_noise_Expert_C", "4class_noise_Junior_A"]

    models = list(cfg.MODELS_TO_RUN)
    if len(models) < 7:
        # still allow, but split by your intended 4/3 rule as much as possible
        first4 = models[:4]
        last3 = models[4:]
    else:
        first4 = models[:4]
        last3 = models[-3:]

    preset = preset.lower().strip()
    if preset == "gpu0":
        return tasks_gpu0, first4
    if preset == "gpu1":
        return tasks_gpu1, last3

    raise ValueError("Unknown preset. Use: gpu0 or gpu1.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["3class", "4class"], help="Which task group to run.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device string, e.g., cuda:0. (Use CUDA_VISIBLE_DEVICES for physical GPU selection.)",
    )

    # NEW: select subset tasks/models (for multi-GPU manual scheduling)
    parser.add_argument(
        "--task_names",
        type=str,
        default=None,
        help="Comma-separated task names to run, e.g. 4class_noise_Expert_B,4class_noise_Junior_A. Default: all tasks in mode.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model keys to run (must match models/__init__.py). Default: config.MODELS_TO_RUN",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["gpu0", "gpu1"],
        help="Convenience preset for your 2-GPU split (only meaningful for mode=4class).",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device} | cuda_available={torch.cuda.is_available()}")

    ensure_dir(cfg.OUTPUT_ROOT)

    tasks = cfg.build_tasks(args.mode)

    # Apply preset (if any)
    selected_task_names = _parse_csv_list(args.task_names)
    selected_models = _parse_csv_list(args.models)

    if args.preset is not None:
        preset_task_names, preset_models = _apply_preset(args.mode, args.preset)
        selected_task_names = preset_task_names
        selected_models = preset_models

    if selected_task_names is not None:
        tasks = [t for t in tasks if t.name in set(selected_task_names)]
        if not tasks:
            raise RuntimeError(f"No tasks matched task_names={selected_task_names}. Available={[t.name for t in cfg.build_tasks(args.mode)]}")

    if selected_models is None:
        models_to_run = list(cfg.MODELS_TO_RUN)
    else:
        models_to_run = list(selected_models)

    print(f"[INFO] Tasks to run: {[t.name for t in tasks]}")
    print(f"[INFO] Models to run: {models_to_run}")
    print(f"[INFO] Seeds: {cfg.SEEDS}")

    for task in tasks:
        print("\n" + "=" * 120)
        print(f"[TASK] {task.name} | num_classes={task.num_classes} | noise_variant={task.noise_variant}")
        print("=" * 120)

        task_out_root = os.path.join(cfg.OUTPUT_ROOT, task.name)
        ensure_dir(task_out_root)

        # Build datasets ONCE per task
        train_ds, valid_ds, test_ds, stats = build_datasets(task)
        save_json(stats, os.path.join(task_out_root, "dataset_stats.json"))
        print(f"[INFO] Dataset stats saved to: {os.path.join(task_out_root, 'dataset_stats.json')}")
        print(f"[INFO] Dataset label counts: {stats}")

        train_label_counts = stats["train"]

        model_summaries = []

        for model_name in models_to_run:
            print("\n" + "-" * 100)
            print(f"[MODEL] {model_name} | seeds={cfg.SEEDS}")
            print("-" * 100)

            model_out_root = os.path.join(task_out_root, model_name)
            ensure_dir(model_out_root)

            seed_rows: List[Dict[str, float]] = []
            cm_sum = None

            for seed in cfg.SEEDS:
                run_name = f"seed_{seed}"
                out_dir = os.path.join(model_out_root, run_name)
                print(f"\n[RUN] task={task.name} model={model_name} seed={seed} => {out_dir}")

                row = run_one_seed(
                    task_name=task.name,
                    model_name=model_name,
                    seed=int(seed),
                    num_classes=task.num_classes,
                    class_names=task.class_names,
                    train_ds=train_ds,
                    valid_ds=valid_ds,
                    test_ds=test_ds,
                    train_label_counts=train_label_counts,
                    model_builder=build_model,
                    device=device,
                    out_dir=out_dir,
                )
                seed_rows.append(row)

                cm_path = os.path.join(out_dir, "confusion_matrix.csv")
                if os.path.exists(cm_path):
                    cm = np.loadtxt(cm_path, delimiter=",", dtype=np.int64)
                    cm_sum = cm if cm_sum is None else (cm_sum + cm)

            df = pd.DataFrame(seed_rows)
            df.to_csv(os.path.join(model_out_root, "all_seeds_metrics.csv"), index=False)

            if cm_sum is not None:
                np.savetxt(os.path.join(model_out_root, "confusion_matrix_sum.csv"), cm_sum.astype(np.int64), fmt="%d", delimiter=",")
                save_confusion_matrix_figure(
                    cm_sum,
                    class_names=task.class_names,
                    out_path=os.path.join(model_out_root, "confusion_matrix_sum_norm.png"),
                    title=f"{task.name} | {model_name} | sum over seeds (row-normalized)",
                    normalize=True,
                    dpi=300,
                )
                save_confusion_matrix_figure(
                    cm_sum,
                    class_names=task.class_names,
                    out_path=os.path.join(model_out_root, "confusion_matrix_sum_raw.png"),
                    title=f"{task.name} | {model_name} | sum over seeds (raw counts)",
                    normalize=False,
                    dpi=300,
                )

            # base metrics
            metric_cols = [
                "test_acc",
                "test_macro_f1",
                "test_weighted_f1",
                "train_time_sec",
                "epochs_ran",
                "early_stop_triggered",
                "infer_time_ms_per_sample",
                "params_total",
                "gmacs",
                "gflops",
            ]

            # auto include per-class metrics for tables
            per_class_cols = sorted(
                [c for c in df.columns if c.startswith("test_precision_") or c.startswith("test_recall_") or c.startswith("test_f1_") or c.startswith("test_class_acc_")]
            )
            metric_cols_extended = metric_cols + per_class_cols

            summary_df = summarize_across_seeds(df, metric_cols=metric_cols_extended)
            summary_df.to_csv(os.path.join(model_out_root, "summary_mean_std.csv"), index=False)

            # One-row model summary
            one = {"model": model_name}
            for col in metric_cols_extended:
                if col not in df.columns:
                    continue
                one[f"{col}_mean"] = float(df[col].mean())
                one[f"{col}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0

            if "params_total" in df.columns:
                one["params_total_M"] = float(df["params_total"].mean()) / 1e6
            if "train_time_sec" in df.columns:
                one["train_time_hours_mean"] = float(df["train_time_sec"].mean()) / 3600.0

            model_summaries.append(one)

            print(f"[DONE] model={model_name} => {model_out_root}")

        task_df = pd.DataFrame(model_summaries)
        task_df.to_csv(os.path.join(task_out_root, "task_summary_models.csv"), index=False)
        print(f"\n[TASK DONE] {task.name} summary saved to: {os.path.join(task_out_root, 'task_summary_models.csv')}")


if __name__ == "__main__":
    main()
