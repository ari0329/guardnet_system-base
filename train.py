"""
GuardNet – Training Script  (Memory-Safe Generator Edition)
===========================================================
Fixes:  "Unable to allocate 1011 MiB" OOM crash

Root cause: old script called load_dataset() which put the entire
dataset (~1 GB) into RAM before training started.

Fix: VideoDataGenerator streams data from disk one batch at a time,
so only `batch_size` sequences are ever in RAM simultaneously.

Usage:
    python train.py --data_dir ./data --epochs 30
    python train.py --data_dir ./data --epochs 50 --batch_size 2
    python train.py --data_dir ./data --epochs 30 --model_out models/guardnet_v2.h5
"""

import os
import sys
import argparse
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.config import DATA_DIR, MODEL_PATH, EPOCHS


def parse_args():
    p = argparse.ArgumentParser(description="Train GuardNet violence detector")
    p.add_argument(
        "--data_dir",   default=DATA_DIR,
        help="Root data dir — must have violence/ and non-violence/ sub-folders",
    )
    p.add_argument(
        "--model_out",  default=MODEL_PATH,
        help="Output path for the saved .h5 model",
    )
    p.add_argument("--epochs",     type=int, default=EPOCHS)
    p.add_argument(
        "--batch_size", type=int, default=2,
        help="Videos per batch  |  8 GB RAM → 2  |  16 GB RAM → 4  |  32 GB RAM → 8",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print("  GuardNet — Memory-Safe Training Pipeline")
    print("=" * 64)
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Model out  : {args.model_out}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch_size}  ← generator mode (no full RAM load)")
    print("=" * 64)

    # ── Step 1: Scan dataset paths (NO frame loading here) ─────────────────
    print("\n[Step 1/4]  Scanning dataset …")
    from utils.preprocessing import scan_dataset, VideoDataGenerator
    paths, labels = scan_dataset(args.data_dir)

    total = len(paths)
    n_vio = sum(1 for l in labels if l == 1)
    n_nv  = sum(1 for l in labels if l == 0)

    print(f"\n  Total clips    : {total}")
    print(f"  Violence       : {n_vio}")
    print(f"  Non-violence   : {n_nv}")

    if total < 4:
        print("\n[ERROR] Need at least 4 clips (2 per class) to train. Aborting.")
        sys.exit(1)

    if n_vio == 0 or n_nv == 0:
        print("\n[ERROR] Both 'violence' and 'non-violence' folders must have clips.")
        sys.exit(1)

    # ── Step 2: Train / validation split ───────────────────────────────────
    print("\n[Step 2/4]  Splitting 80 % train / 20 % validation …")
    combined = list(zip(paths, labels))
    random.shuffle(combined)
    split   = max(1, int(0.8 * len(combined)))
    train_c = combined[:split]
    val_c   = combined[split:] if split < len(combined) else combined[-1:]

    train_paths,  train_labels  = zip(*train_c)
    val_paths,    val_labels    = zip(*val_c)

    print(f"  Train : {len(train_paths)} clips")
    print(f"  Val   : {len(val_paths)}  clips")

    train_gen = VideoDataGenerator(
        list(train_paths), list(train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        augment=True,          # flip + brightness jitter
    )
    val_gen = VideoDataGenerator(
        list(val_paths), list(val_labels),
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    )

    # ── Step 3: Build model ─────────────────────────────────────────────────
    print("\n[Step 3/4]  Building model …")
    import tensorflow as tf
    from models.guardnet_model import build_model, unfreeze_top_layers

    model = build_model()
    model.summary()

    os.makedirs(os.path.dirname(args.model_out) or ".", exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            args.model_out,
            save_best_only=True,
            monitor="val_accuracy",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    # ── Step 4: Train ───────────────────────────────────────────────────────
    print("\n[Step 4/4]  Training …\n")

    # Phase 1 — backbone frozen (fast initial convergence)
    print("  ► Phase 1: frozen backbone …")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        workers=1,                  # keep at 1 on Windows; 4 on Linux/Mac
        use_multiprocessing=False,  # keep False on Windows
    )

    # Phase 2 — fine-tune top backbone layers
    print("\n  ► Phase 2: fine-tuning top layers …")
    model = unfreeze_top_layers(model, num_layers=30)
    fine_epochs = max(5, args.epochs // 3)
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=fine_epochs,
        callbacks=callbacks,
        workers=1,
        use_multiprocessing=False,
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    best_val_acc = max(history.history.get("val_accuracy", [0]))

    print("\n" + "=" * 64)
    print("  ✓  Training complete!")
    print(f"  Best val accuracy : {best_val_acc * 100:.2f}%")
    print(f"  Model saved       : {args.model_out}")
    print("=" * 64)
    print("\nNext steps:")
    print("  python demo.py --source 0          # webcam")
    print("  python demo.py --source video.mp4  # video file")
    print("  streamlit run dashboard.py         # dashboard\n")


if __name__ == "__main__":
    main()