"""
scripts/sample.py  —  Generate and compare samples (Parts 5C, 6B, 6D)
=======================================================================

Usage::
    # EM samples  (5.C.iii)
    python scripts/sample.py --method em --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000

    # PC samples  (5.C.iv)
    python scripts/sample.py --method pc --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000 --n_corrector 1
    python scripts/sample.py --method pc --checkpoint runs/vp/best.pt \\
        --beta_min 0.01 --beta_max 5.0 --num_steps 1000 --n_corrector 3

    # Rectified Flow Euler  (6.B)
    python scripts/sample.py --method rectflow --checkpoint runs/rectflow/best.pt \\
        --num_steps 100

    # One-step reflow  (6.C)
    python scripts/sample.py --method rectflow --checkpoint runs/rectflow_reflow/best.pt \\
        --num_steps 1

    # Side-by-side grid  (6.D): pass a fixed seed file
    python scripts/sample.py --method all --vp_checkpoint runs/vp/best.pt \\
        --rf_checkpoint runs/rectflow/best.pt \\
        --reflow_checkpoint runs/rectflow_reflow/best.pt \\
        --seed 42 --out comparison_grid.png
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


FASHION_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def save_grid(samples: torch.Tensor, path: str, nrow: int = 8, title: str = ""):
    """Save a (B,1,H,W) tensor as an image grid."""
    grid = make_grid(samples.clamp(-1, 1) * 0.5 + 0.5, nrow=nrow)
    plt.figure(figsize=(nrow, samples.size(0) // nrow + 1))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy(), cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method",      type=str, default="em",
                   choices=["em", "pc", "rectflow", "all"],
                   help="Sampler to run (or 'all' for side-by-side grid).")
    # VP checkpoints
    p.add_argument("--checkpoint",    type=str, default=None)
    p.add_argument("--vp_checkpoint", type=str, default=None)
    # Rect-flow checkpoints
    p.add_argument("--rf_checkpoint",     type=str, default=None)
    p.add_argument("--reflow_checkpoint", type=str, default=None)
    # VP schedule
    p.add_argument("--beta_min", type=float, default=0.01)
    p.add_argument("--beta_max", type=float, default=5.0)
    p.add_argument("--T",        type=int,   default=1000)
    # Sampler params
    p.add_argument("--num_steps",   type=int, default=1000)
    p.add_argument("--n_corrector", type=int, default=1)
    p.add_argument("--snr",         type=float, default=0.16)
    p.add_argument("--n_samples",   type=int, default=64)
    # Output
    p.add_argument("--out",    type=str, default="samples.png")
    p.add_argument("--seed",   type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_vp_model(checkpoint: str, device) -> tuple[VPSDE, UNet]:
    """Load a trained VP-SDE score model from a checkpoint."""
    model = UNet(in_channels=1, base_channels=64).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    sde = VPSDE(beta_min=0.01, beta_max=5.0, T=1000)
    return sde, model


def load_rf_model(checkpoint: str, device) -> tuple[RectifiedFlow, UNet]:
    """Load a trained Rectified Flow model from a checkpoint."""
    model = UNet(in_channels=1, base_channels=64).to(device)
    state_dict = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    rf = RectifiedFlow()
    return rf, model


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    shape = (args.n_samples, 1, 28, 28)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.method == "em":
        # EM samples (5.C.iii)
        sde, model = load_vp_model(args.checkpoint, device)
        model.eval()
        samples = sde.euler_maruyama(
            model, shape, num_steps=args.num_steps, device=device
        )
        save_grid(samples, args.out, nrow=8, title="VP-EM Sampling")

    elif args.method == "pc":
        # PC samples (5.C.iv)
        sde, model = load_vp_model(args.checkpoint, device)
        model.eval()
        samples = sde.predictor_corrector(
            model, shape, num_steps=args.num_steps, 
            n_corrector=args.n_corrector, snr=args.snr, device=device
        )
        save_grid(samples, args.out, nrow=8, 
                 title=f"VP-PC (n_corrector={args.n_corrector})")

    elif args.method == "rectflow":
        # Rectified Flow Euler sampler (6.B / 6.C)
        rf, model = load_rf_model(args.checkpoint, device)
        model.eval()
        samples = rf.euler_sample(
            model, shape, num_steps=args.num_steps, device=device
        )
        save_grid(samples, args.out, nrow=8, 
                 title=f"Rectified Flow ({args.num_steps} steps)")

    elif args.method == "all":
        # Side-by-side grid (6.D) — 4×8: 4 methods, 8 fixed seeds
        torch.manual_seed(args.seed)
        sde, vp_model = load_vp_model(args.vp_checkpoint, device)
        rf, rf_model = load_rf_model(args.rf_checkpoint, device)
        reflow, reflow_model = load_rf_model(args.reflow_checkpoint, device)
        
        vp_model.eval()
        rf_model.eval()
        reflow_model.eval()
        
        # Generate samples for each method
        samples_vp = sde.euler_maruyama(
            vp_model, shape, num_steps=1000, device=device
        )
        samples_rf = rf.euler_sample(
            rf_model, shape, num_steps=100, device=device
        )
        samples_reflow = reflow.euler_sample(
            reflow_model, shape, num_steps=1, device=device
        )
        
        # Arrange in 4×8 grid
        all_samples = torch.cat([
            samples_vp,
            samples_rf,
            samples_reflow,
        ], dim=0)
        
        # If we have exactly 3 models, fill 4th row with duplicates or blanks
        if all_samples.size(0) < 32:
            blank = torch.zeros((8, 1, 28, 28), device=device)
            all_samples = torch.cat([all_samples, blank], dim=0)
        
        save_grid(all_samples, args.out, nrow=8, 
                 title="Method Comparison")


if __name__ == "__main__":
    main()
