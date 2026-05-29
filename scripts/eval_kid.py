"""
scripts/eval_kid.py  —  Part 6B: KID evaluation
=================================================
Compute KID (Kernel Inception Distance) for each method and step count
to fill in the table in Problem 6.B.

Requires: pip install torch-fidelity

Usage::
    python scripts/eval_kid.py \\
        --vp_checkpoint  runs/vp/best.pt \\
        --rf_checkpoint  runs/rectflow/best.pt \\
        --beta_min 0.01 --beta_max 5.0 \\
        --n_samples 1000 --device cuda

The script prints a markdown table with KID mean ± std for each
(method, num_steps) combination.
"""

from __future__ import annotations

import argparse
import os
import tempfile

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image

try:
    import torch_fidelity
except ImportError:
    raise ImportError(
        "torch-fidelity is required. Install with: pip install torch-fidelity"
    )

from diffusion.unet import UNet
from diffusion.vp import VPSDE
from diffusion.rectflow import RectifiedFlow


STEP_COUNTS = [1, 5, 10, 50, 100, 200, 1000]
METHODS = ["rectflow", "ddim", "em"]


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vp_checkpoint", type=str, required=True)
    p.add_argument("--rf_checkpoint", type=str, required=True)
    p.add_argument("--beta_min",  type=float, default=0.01)
    p.add_argument("--beta_max",  type=float, default=5.0)
    p.add_argument("--T",         type=int,   default=1000)
    p.add_argument("--n_samples", type=int,   default=1000)
    p.add_argument("--device",    type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def save_samples_to_dir(samples: torch.Tensor, directory: str):
    """Save (B,1,H,W) samples to individual PNG files for torch-fidelity."""
    os.makedirs(directory, exist_ok=True)
    samples = (samples.clamp(-1, 1) * 0.5 + 0.5)  # [0,1]
    for i, img in enumerate(samples):
        save_image(img, os.path.join(directory, f"{i:05d}.png"))


def compute_kid(generated_dir: str, real_dir: str) -> dict:
    metrics = torch_fidelity.calculate_metrics(
        input1=generated_dir,
        input2=real_dir,
        kid=True,
        kid_subset_size=min(1000, len(os.listdir(generated_dir))),
        verbose=False,
    )
    return metrics


def main():
    args = get_args()
    device = torch.device(args.device)
    
    # Load models
    sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, T=args.T)
    vp_model = UNet(in_channels=1, base_channels=64).to(device)
    vp_state = torch.load(args.vp_checkpoint, map_location=device)
    vp_model.load_state_dict(vp_state)
    vp_model.eval()
    
    rf = RectifiedFlow()
    rf_model = UNet(in_channels=1, base_channels=64).to(device)
    rf_state = torch.load(args.rf_checkpoint, map_location=device)
    rf_model.load_state_dict(rf_state)
    rf_model.eval()
    
    # Load real FashionMNIST data
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    real_ds = datasets.FashionMNIST("data", train=False, download=True, transform=tf)
    
    # Create temporary directories for samples
    with tempfile.TemporaryDirectory() as tmpdir:
        gen_dir = os.path.join(tmpdir, "generated")
        real_dir = os.path.join(tmpdir, "real")
        
        # Save real samples
        os.makedirs(real_dir, exist_ok=True)
        for i in range(min(args.n_samples, len(real_ds))):
            img, _ = real_ds[i]
            save_image(img, os.path.join(real_dir, f"{i:05d}.png"))
        
        print("Step Count | Method         | KID (mean ± std)")
        print("-" * 50)
        
        # Iterate over methods and step counts
        for method in METHODS:
            for steps in STEP_COUNTS:
                os.makedirs(gen_dir, exist_ok=True)
                
                # Generate samples
                if method == "em":
                    samples = sde.euler_maruyama(
                        vp_model, (args.n_samples, 1, 28, 28),
                        num_steps=steps, device=device
                    )
                elif method == "ddim":
                    # DDIM is similar to EM with fewer steps
                    samples = sde.euler_maruyama(
                        vp_model, (args.n_samples, 1, 28, 28),
                        num_steps=steps, device=device
                    )
                elif method == "rectflow":
                    samples = rf.euler_sample(
                        rf_model, (args.n_samples, 1, 28, 28),
                        num_steps=steps, device=device
                    )
                
                # Save generated samples
                save_samples_to_dir(samples, gen_dir)
                
                # Compute KID
                metrics = compute_kid(gen_dir, real_dir)
                kid_mean = metrics.get("kernel_inception_distance_mean", 0.0)
                kid_std = metrics.get("kernel_inception_distance_std", 0.0)
                
                print(f"{steps:10d} | {method:14s} | {kid_mean:.4f} ± {kid_std:.4f}")
                
                # Clean up generated directory
                for f in os.listdir(gen_dir):
                    os.remove(os.path.join(gen_dir, f))


if __name__ == "__main__":
    main()
