"""
scripts/plot_coefficient.py  —  Part 1.8
=========================================
Plot the DDPM loss coefficient
    β_t² / (2 σ_t² α_t (1 - ᾱ_t))
vs. t on a log-scale y-axis.

Usage::
    python scripts/plot_coefficient.py --T 1000 --beta_start 1e-4 --beta_end 0.02
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np


def linear_schedule(T: int, beta_start: float, beta_end: float):
    return np.linspace(beta_start, beta_end, T)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T",          type=int,   default=1000)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end",   type=float, default=0.02)
    parser.add_argument("--out",        type=str,   default="coefficient_plot.png")
    args = parser.parse_args()

    # Compute schedule
    betas = linear_schedule(args.T, args.beta_start, args.beta_end)
    
    # Compute alpha and alpha_bar
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    
    # Compute sigma^2 = 1 - alpha_bar
    sigma_sq = 1.0 - alpha_bars
    
    # Compute the loss coefficient: β_t^2 / (2 σ_t^2 α_t (1 - ᾱ_t))
    # = β_t^2 / (2 σ_t^2 α_t σ_t^2)
    # = β_t^2 / (2 α_t (1 - ᾱ_t)^2)
    alpha_t = alphas
    coefficient = betas**2 / (2.0 * alpha_t * sigma_sq**2 + 1e-8)
    
    # Plot on log scale
    plt.figure(figsize=(10, 6))
    plt.semilogy(range(1, args.T + 1), coefficient, linewidth=2)
    plt.xlabel("Time step t")
    plt.ylabel("Loss coefficient (log scale)")
    plt.title("DDPM Loss Coefficient vs. Time")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
