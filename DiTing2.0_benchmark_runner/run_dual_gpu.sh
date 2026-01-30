#!/usr/bin/env bash
set -e

# Run 3-class on physical GPU 0, 4-class on physical GPU 1, concurrently.
# NOTE: CUDA_VISIBLE_DEVICES remaps the visible GPU(s), so inside Python each process uses cuda:0.

echo "[INFO] Launching 3-class task on GPU0 ..."
CUDA_VISIBLE_DEVICES=0 python -u run_task.py --mode 3class --device cuda:0 2>&1 | tee -a logs_3class.txt &

echo "[INFO] Launching 4-class tasks (4 noise subsets) on GPU1 ..."
CUDA_VISIBLE_DEVICES=1 python -u run_task.py --mode 4class --device cuda:0 2>&1 | tee -a logs_4class.txt &

wait
echo "[INFO] All tasks finished."
echo "[INFO] Logs: logs_3class.txt, logs_4class.txt"
