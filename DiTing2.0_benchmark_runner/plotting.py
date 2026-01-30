from __future__ import annotations

import os
from typing import List

# ✅ IMPORTANT: headless backend (no GUI, no X11 forwarding popups)
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir


def save_confusion_matrix_figure(
    cm: np.ndarray,
    class_names: List[str],
    out_path: str,
    title: str = "Confusion Matrix",
    normalize: bool = True,
    dpi: int = 300,
) -> None:
    """
    Save confusion matrix as a high-resolution figure.

    normalize=True => row-normalized (recall per class).
    """
    ensure_dir(os.path.dirname(out_path))

    cm_to_plot = cm.astype(np.float64)
    if normalize:
        row_sum = cm_to_plot.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        cm_to_plot = cm_to_plot / row_sum

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(cm_to_plot, interpolation="nearest")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # annotate
    thresh = cm_to_plot.max() * 0.5 if cm_to_plot.size > 0 else 0.5
    for i in range(cm_to_plot.shape[0]):
        for j in range(cm_to_plot.shape[1]):
            val = cm_to_plot[i, j]
            txt = f"{val:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
