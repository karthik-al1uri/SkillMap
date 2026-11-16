"""Regenerate the report's confusion matrix from its small, source-hashed input.

Run from this directory: python -B generate_figures.py
Requires the project's existing matplotlib/numpy installation; no training or network.
The input was reconciled against saved prediction arrays during final validation.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    data = json.loads((root / "figures/grouped_confusion_source.json").read_text())
    counts = np.asarray(data["confusion_matrix"], dtype=int)
    assert counts.shape == (3, 3) and int(counts.sum()) == data["assessment_jobs"]
    fractions = counts / counts.sum(axis=1, keepdims=True)
    with plt.rc_context({"font.size": 9, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(3.30, 2.75), layout="constrained")
        plot = ax.imshow(fractions, cmap="Blues", vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{counts[i, j]:,}", ha="center", va="center",
                        color="white" if fractions[i, j] > 0.6 else "black")
        ax.set_xticks(range(3), data["labels"])
        ax.set_yticks(range(3), data["labels"])
        ax.set_xlabel("Predicted salary tier")
        ax.set_ylabel("Actual salary tier")
        colorbar = fig.colorbar(plot, ax=ax, fraction=0.055, pad=0.04)
        colorbar.set_label("Fraction of actual class")
        fig.savefig(root / "figures/grouped_confusion.pdf", metadata={
            "Title": "Description-grouped fold-zero confusion matrix",
            "CreationDate": None, "ModDate": None,
        })
        plt.close(fig)


if __name__ == "__main__":
    main()
