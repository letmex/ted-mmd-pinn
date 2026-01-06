"""
Example CLI to compute crack length a(t) from a COMSOL table.

The script reads a whitespace-delimited ``.txt`` file with a header containing
time, x, y and damage columns. Install dependencies with:

    pip install -r requirements.txt

SciPy is used for fast k-NN via ``cKDTree``; if unavailable, the code falls
back to a pure numpy implementation.
"""

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from source.postprocess.geodesic_crack_length import extract_a_t_geodesic


def compute_da_dN(times: np.ndarray, lengths: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finite-difference derivative da/dN between successive time steps.
    """
    if times.size < 2:
        return np.array([]), np.array([])
    delta_t = np.diff(times)
    delta_a = np.diff(lengths)
    mid_times = times[:-1] + 0.5 * delta_t
    with np.errstate(divide="ignore", invalid="ignore"):
        da_dN = np.divide(delta_a, delta_t, out=np.zeros_like(delta_a), where=delta_t != 0)
    return mid_times, da_dN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract crack length a(t) using geodesic distance from a COMSOL export.",
    )
    parser.add_argument("table", type=Path, help="Path to COMSOL .txt table with columns t/x/y/d")
    parser.add_argument("--x-root", type=float, required=True, help="x coordinate of the pre-crack tip")
    parser.add_argument("--y-root", type=float, required=True, help="y coordinate of the pre-crack tip")
    parser.add_argument("--d-threshold", type=float, default=0.5, help="Damage threshold for masking")
    parser.add_argument("--k", type=int, default=12, help="Number of kNN neighbours")
    parser.add_argument("--y-band", type=float, default=None, help="Optional |y - y_root| filter")
    parser.add_argument("--delimiter", type=str, default=None, help="Custom delimiter for the table")
    parser.add_argument("--savefig", type=Path, default=None, help="Optional path to save the plot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    times, lengths = extract_a_t_geodesic(
        str(args.table),
        x_root=args.x_root,
        y_root=args.y_root,
        d_threshold=args.d_threshold,
        k=args.k,
        y_band=args.y_band,
        delimiter=args.delimiter,
    )
    mid_times, da_dN = compute_da_dN(times, lengths)

    print("t, a(t):")
    for t_val, a_val in zip(times, lengths):
        print(f"{t_val:.6g}, {a_val:.6g}")

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=False)
    axes[0].plot(times, lengths, marker="o")
    axes[0].set_xlabel("t or N")
    axes[0].set_ylabel("a(t)")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    axes[1].plot(mid_times, da_dN, marker="s", color="C1")
    axes[1].set_xlabel("t or N")
    axes[1].set_ylabel("da/dN")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    if args.savefig:
        fig.savefig(args.savefig, dpi=200)
        print(f"Saved figure to {args.savefig}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

