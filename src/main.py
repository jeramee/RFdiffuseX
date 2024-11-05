# src/main.py

import argparse
from src.training.train_diffusion import train_diffusion
from src.training.train_ppi import train_ppi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFdiffuseX: Expanded RFdiffusion with PPI Prediction")
    parser.add_argument("--task", choices=["diffusion", "ppi"], required=True, help="Task to perform")
    args = parser.parse_args()

    if args.task == "diffusion":
        train_diffusion()
    elif args.task == "ppi":
        train_ppi()
