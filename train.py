"""
GuardNet – Training Script
Usage:
    python train.py --data_dir ./data --epochs 30
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import DATA_DIR, MODEL_PATH, EPOCHS, BATCH_SIZE
from utils.preprocessing import load_dataset
from models.guardnet_model import train, build_model


def parse_args():
    p = argparse.ArgumentParser(description="Train GuardNet violence detector")
    p.add_argument("--data_dir", default=DATA_DIR,
                   help="Root data directory (must contain violence/ and non-violence/ sub-dirs)")
    p.add_argument("--model_out", default=MODEL_PATH,
                   help="Where to save the trained .h5 model")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  GuardNet – Training Pipeline")
    print("=" * 60)
    print(f"  Data dir  : {args.data_dir}")
    print(f"  Model out : {args.model_out}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print("=" * 60)

    # ── Load data ──
    print("\n[Step 1/3] Loading & preprocessing dataset …")
    X, y = load_dataset(args.data_dir)
    print(f"  Loaded {len(X)} sequences | Class balance: "
          f"{(y==0).sum()} non-violent, {(y==1).sum()} violent")

    # ── Shuffle ──
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]

    # ── Train ──
    print("\n[Step 2/3] Training model …")
    model, history = train(X, y, model_save_path=args.model_out)

    # ── Report ──
    print("\n[Step 3/3] Training complete.")
    best_val_acc = max(history.history.get("val_accuracy", [0]))
    print(f"  Best validation accuracy : {best_val_acc * 100:.2f}%")
    print(f"  Model saved              : {args.model_out}")
    print("\nRun detection with:")
    print("  python demo.py --source 0          # webcam")
    print("  python demo.py --source video.mp4  # file")


if __name__ == "__main__":
    main()
