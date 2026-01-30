
# DiTing 2.0 Benchmark Runner (GJI major revision)

This codebase is a lightweight, **copy-paste runnable** benchmark runner for:

- **3-class**: Earthquake / Explosion / Collapse (no noise)
- **4-class**: Earthquake / Explosion / Collapse / Noise
  - Noise subsets: **Expert_A, Expert_B, Expert_C, Junior_A**
- Each experiment runs **5 seeds** and outputs mean/std for:
  - Accuracy, Macro-F1, Weighted-F1
  - training time, inference time (ms/sample), parameter counts
  - confusion matrices (raw + normalized)

## 0) Folder structure

Recommended structure:

```
your_project/
  diting_benchmark_runner/          # (this repo)
    config.py
    run_task.py
    run_dual_gpu.sh
    datasets.py
    train_eval.py
    losses.py
    metrics.py
    plotting.py
    models/
      ...
  runs_gji_revision/                # outputs created automatically
  (your .pt data can stay anywhere)
```

You can keep your `.pt` files in:

`/home/zypei/DiTing2.0_dataset/remake_noise_experiment/EQ_EP_SS_Noise_datasets_enhancement_MFCC`

and just set `DATA_ROOT` in `config.py` to that folder.

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Configure

Edit **config.py**:

- `DATA_ROOT`: where your `.pt` files are
- `OUTPUT_ROOT`: where to save results
- `MODELS_TO_RUN`: list of models
- `SEEDS`: list of 5 seeds (default [0,1,2,3,4])
- `EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY`
- ViT / CapsNet settings

## 3) Run (single task)

### Run 3-class (on one GPU)

```bash
CUDA_VISIBLE_DEVICES=0 python run_task.py --mode 3class --device cuda:0
```

### Run 4-class (all noise subsets)

```bash
CUDA_VISIBLE_DEVICES=0 python run_task.py --mode 4class --device cuda:0
```

## 4) Run both tasks concurrently on two GPUs

This will:
- GPU0 => 3-class
- GPU1 => 4-class (Expert_A/B/C + Junior_A)

```bash
bash run_dual_gpu.sh
```

Logs will be written to:
- `logs_3class.txt`
- `logs_4class.txt`

## 5) Output files

For each task and model you will get:

- `runs_gji_revision/<task>/<model>/all_seeds_metrics.csv`
- `runs_gji_revision/<task>/<model>/summary_mean_std.csv`
- `runs_gji_revision/<task>/<model>/seed_<seed>/metrics.json`
- `runs_gji_revision/<task>/<model>/seed_<seed>/confusion_matrix.csv`
- `runs_gji_revision/<task>/<model>/seed_<seed>/confusion_matrix_norm.png`
- `runs_gji_revision/<task>/<model>/seed_<seed>/confusion_matrix_raw.png`

Task-level model comparison table:
- `runs_gji_revision/<task>/task_summary_models.csv`

Dataset sample counts (to fix Table-1 inconsistencies):
- `runs_gji_revision/<task>/dataset_stats.json`


### Confusion matrices across 5 seeds

For each (task, model), we also save a **sum confusion matrix across 5 seeds**:

- `confusion_matrix_sum.csv`
- `confusion_matrix_sum_raw.png`
- `confusion_matrix_sum_norm.png`
