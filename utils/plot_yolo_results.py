from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 10


DEFAULT_CSVS = [
    "runs/detect/results/yolo_roi/yolov8n_leaf_roi_stage12/results.csv",
    "runs/detect/results/yolo_roi/yolov8n_leaf_roi_stage2/results.csv",
]

METRICS_ALL = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "lr/pg0",
    "lr/pg1",
]

LOSS_PAIRS = [
    ("box_loss", "train/box_loss", "val/box_loss"),
    ("cls_loss", "train/cls_loss", "val/cls_loss"),
    ("dfl_loss", "train/dfl_loss", "val/dfl_loss"),
]

METRICS_COMPARE = [
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "val/box_loss",
    "train/box_loss",
]


def pretty_name(metric: str) -> str:
    return metric.replace("metrics/", "").replace("train/", "train_").replace("val/", "val_")


def ylabel_name(metric: str) -> str:
    if "loss" in metric:
        return "loss"
    if "lr/" in metric:
        return "learning rate"
    return "score"


def display_run_name(run_name: str) -> str:
    if "stage12" in run_name:
        return "Stage1"
    if "stage2" in run_name:
        return "Stage2"
    return run_name


def load_run(csv_path: Path):
    df = pd.read_csv(csv_path)
    run_name = csv_path.parent.name
    return run_name, df


def plot_single_run(run_name: str, df: pd.DataFrame, output_dir: Path):
    standalone = [
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "lr/pg0",
    ]
    standalone = [m for m in standalone if m in df.columns]
    panels = [("pair", p) for p in LOSS_PAIRS if p[1] in df.columns or p[2] in df.columns]
    panels += [("single", m) for m in standalone]
    n = len(panels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 3.6 * nrows), dpi=220)
    axes = axes.flatten()
    for i, panel in enumerate(panels):
        ax = axes[i]
        kind, content = panel
        if kind == "pair":
            title, train_metric, val_metric = content
            if train_metric in df.columns:
                ax.plot(df["epoch"], df[train_metric], linewidth=2.0, color="#1f77b4", label="train")
            if val_metric in df.columns:
                ax.plot(df["epoch"], df[val_metric], linewidth=2.0, color="#ff7f0e", label="val")
            ax.set_title(title)
            ax.set_ylabel("loss")
            ax.legend(loc="best", frameon=False)
        else:
            metric = content
            ax.plot(df["epoch"], df[metric], linewidth=2.0, color="#2ca02c", label=pretty_name(metric))
            ax.set_title(pretty_name(metric))
            ax.set_ylabel(ylabel_name(metric))
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        ax.margins(x=0.02)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"YOLO Training Curves ({display_run_name(run_name)})", y=0.985, fontsize=14)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.955], w_pad=1.2, h_pad=1.6)
    out = output_dir / f"{run_name}_custom_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_compare(runs, output_dir: Path):
    fig, axes = plt.subplots(2, 3, figsize=(17, 8.8), dpi=220)
    axes = axes.flatten()
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd"]
    for i, metric in enumerate(METRICS_COMPARE):
        ax = axes[i]
        for idx, (run_name, df) in enumerate(runs):
            if metric in df.columns:
                ax.plot(
                    df["epoch"],
                    df[metric],
                    linewidth=2.2,
                    label=display_run_name(run_name),
                    color=colors[idx % len(colors)],
                )
        ax.set_title(pretty_name(metric))
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel_name(metric))
        ax.grid(alpha=0.25)
        ax.margins(x=0.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=max(1, len(runs)),
        frameon=False,
    )
    fig.suptitle("YOLO Stage Comparison", y=0.995, fontsize=15)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92], w_pad=1.2, h_pad=1.6)
    out = output_dir / "yolo_compare_custom_curves.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_paths = [project_root / p for p in DEFAULT_CSVS]
    output_dir = project_root / "runs" / "detect" / "results" / "yolo_roi" / "custom_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for csv in csv_paths:
        if csv.exists():
            runs.append(load_run(csv))
    if not runs:
        raise FileNotFoundError("未找到可用 results.csv")

    single_outputs = []
    for run_name, df in runs:
        single_outputs.append(plot_single_run(run_name, df, output_dir))
    compare_output = plot_compare(runs, output_dir)

    for p in single_outputs:
        print(f"[INFO] 单模型曲线图: {p}")
    print(f"[INFO] 对比曲线图: {compare_output}")


if __name__ == "__main__":
    main()
