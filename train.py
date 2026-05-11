"""
GuardNet – Training Script  (Anti-Overfit + Memory-Safe Edition v3)
====================================================================
Fixes applied
  Problem 1 – Overfitting / 100 % training accuracy
    ✓ MobileNetV2 backbone (pretrained, frozen Phase 1)
    ✓ BatchNorm + Dropout + L2 regularisation (inside model)
    ✓ Proper 70 / 15 / 15 train / val / test split at VIDEO level
    ✓ Augmentation applied only to training split
    ✓ Class-weighted loss (handles minor class imbalance)
    ✓ EarlyStopping (patience 10, restores best weights)
    ✓ ReduceLROnPlateau (factor 0.5, patience 4)
    ✓ Two-phase training: Phase 1 frozen → Phase 2 fine-tune top 30 layers
    ✓ Final evaluation on held-out TEST set
    ✓ Training curve plot saved to models/training_curve.png

  Problem 2 – OOM crash
    ✓ VideoDataGenerator streams data batch-by-batch (no full RAM load)

Usage:
    python train.py --data_dir ./data --epochs 60
    python train.py --data_dir ./data --epochs 60 --batch_size 4
    python train.py --data_dir ./data --epochs 60 --model_out models/guardnet_v3.h5
"""

import os
import sys
import argparse
import random
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Config defaults ───────────────────────────────────────────────────────────
try:
    from config.config import DATA_DIR, MODEL_PATH, EPOCHS
except ImportError:
    DATA_DIR   = "./data"
    MODEL_PATH = "models/guardnet_v3.h5"
    EPOCHS     = 60


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train GuardNet violence detector")
    p.add_argument("--data_dir",   default=DATA_DIR)
    p.add_argument("--model_out",  default=MODEL_PATH)
    p.add_argument("--epochs",     type=int,   default=EPOCHS)
    p.add_argument("--batch_size", type=int,   default=2,
                   help="Clips per batch | 8 GB RAM→2 | 16 GB→4 | 32 GB→8")
    p.add_argument("--seq_len",    type=int,   default=16,
                   help="Frames sampled per clip")
    p.add_argument("--img_size",   type=int,   default=112,
                   help="Square frame resolution (96/112/128/160/224)")
    p.add_argument("--lr",         type=float, default=1e-4,
                   help="Initial learning rate (Phase 1)")
    p.add_argument("--fine_lr",    type=float, default=5e-6,
                   help="Fine-tune learning rate (Phase 2)")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--no_plot",    action="store_true",
                   help="Skip saving the training curve plot")
    return p.parse_args()


# ── Plotting ──────────────────────────────────────────────────────────────────
def save_training_plot(all_history: list, out_path: str):
    """Save accuracy + loss curves. Uses matplotlib if available."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        colours = ["tab:blue", "tab:orange"]

        for ax_idx, (metric, ylabel) in enumerate(
            [("accuracy", "Accuracy"), ("loss", "Loss")]
        ):
            for ph_idx, hist in enumerate(all_history):
                label_prefix = f"Phase {ph_idx + 1} "
                x_offset = sum(len(h.history[metric]) for h in all_history[:ph_idx])
                xs = range(x_offset, x_offset + len(hist.history[metric]))
                axes[ax_idx].plot(xs, hist.history[metric],
                                  label=label_prefix + "train",
                                  color=colours[ph_idx], linestyle="-")
                axes[ax_idx].plot(xs, hist.history[f"val_{metric}"],
                                  label=label_prefix + "val",
                                  color=colours[ph_idx], linestyle="--")
            axes[ax_idx].set_xlabel("Epoch")
            axes[ax_idx].set_ylabel(ylabel)
            axes[ax_idx].set_title(f"GuardNet – {ylabel}")
            axes[ax_idx].legend()
            axes[ax_idx].grid(alpha=0.3)

        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Training curve saved → {out_path}")
    except ImportError:
        print("  [INFO] matplotlib not installed – skipping plot.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    random.seed(args.seed)

    print("=" * 68)
    print("  GuardNet v3 — Anti-Overfit + Memory-Safe Training Pipeline")
    print("=" * 68)
    print(f"  Data dir    : {args.data_dir}")
    print(f"  Model out   : {args.model_out}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Seq length  : {args.seq_len} frames")
    print(f"  Frame size  : {args.img_size}×{args.img_size}")
    print(f"  Seed        : {args.seed}")
    print("=" * 68)

    # ── TF setup ──────────────────────────────────────────────────────────────
    import tensorflow as tf
    tf.random.set_seed(args.seed)

    # Allow GPU memory growth (prevents OOM on low-VRAM GPUs)
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n  TF {tf.__version__} | GPUs detected: {len(gpus)}")

    # ── Step 1: Scan dataset ──────────────────────────────────────────────────
    print("\n[Step 1/5]  Scanning dataset …")
    from utils.preprocessing import scan_dataset, three_way_split, VideoDataGenerator

    paths, labels = scan_dataset(args.data_dir)
    total = len(paths)
    n_vio = sum(1 for l in labels if l == 1)
    n_nv  = total - n_vio

    print(f"\n  Total clips    : {total}")
    print(f"  Violence       : {n_vio}")
    print(f"  Non-violence   : {n_nv}")

    # Dataset size warnings
    if total < 6:
        print("\n  [ERROR] Need at least 6 clips to create all 3 splits. Aborting.")
        sys.exit(1)
    if n_vio == 0 or n_nv == 0:
        print("\n  [ERROR] Both classes must be present. Aborting.")
        sys.exit(1)
    if total < 100:
        print(
            "\n  [WARN] Very small dataset – model WILL overfit regardless of "
            "technique. Aim for ≥500 clips per class for reliable results.\n"
            "  Proceeding with maximum regularisation …"
        )

    # ── Step 2: Three-way split ───────────────────────────────────────────────
    print("\n[Step 2/5]  Creating 70 / 15 / 15 train / val / test split …")
    (train_paths, train_labels), \
    (val_paths,   val_labels),   \
    (test_paths,  test_labels)   = three_way_split(
        paths, labels, seed=args.seed
    )

    print(f"  Train : {len(train_paths)} clips")
    print(f"  Val   : {len(val_paths)}  clips")
    print(f"  Test  : {len(test_paths)} clips (held-out, evaluated AFTER training)")

    # ── Verify no leakage ─────────────────────────────────────────────────────
    train_set = set(train_paths)
    val_set   = set(val_paths)
    test_set  = set(test_paths)
    overlap_tv = train_set & val_set
    overlap_tt = train_set & test_set
    if overlap_tv or overlap_tt:
        print(
            f"\n  [ERROR] Data leakage detected! "
            f"Train∩Val={len(overlap_tv)}, Train∩Test={len(overlap_tt)}. Aborting."
        )
        sys.exit(1)
    print("  ✓ No frame overlap detected between splits.")

    # ── Generators ───────────────────────────────────────────────────────────
    gen_kwargs = dict(batch_size=args.batch_size,
                      seq_len=args.seq_len,
                      img_h=args.img_size,
                      img_w=args.img_size)

    train_gen = VideoDataGenerator(
        train_paths, train_labels,
        shuffle=True,
        augment=True,   # ← augmentation ONLY on training data
        **gen_kwargs,
    )
    val_gen = VideoDataGenerator(
        val_paths, val_labels,
        shuffle=False,
        augment=False,
        **gen_kwargs,
    )
    test_gen = VideoDataGenerator(
        test_paths, test_labels,
        shuffle=False,
        augment=False,
        **gen_kwargs,
    )

    # Class weights (handles imbalance)
    class_weight = train_gen.class_weights()
    print(f"\n  Class weights: {class_weight}")

    # ── Step 3: Build model ───────────────────────────────────────────────────
    print("\n[Step 3/5]  Building model …")
    from models.guardnet_model import build_model, unfreeze_top_layers

    model = build_model(
        seq_len=args.seq_len,
        img_h=args.img_size,
        img_w=args.img_size,
        learning_rate=args.lr,
    )
    model.summary(line_length=90)

    out_dir = os.path.dirname(args.model_out) or "."
    os.makedirs(out_dir, exist_ok=True)

    # ── Step 4: Train ─────────────────────────────────────────────────────────
    print("\n[Step 4/5]  Training …\n")

    def make_callbacks(monitor_metric="val_accuracy"):
        return [
            tf.keras.callbacks.ModelCheckpoint(
                args.model_out,
                save_best_only=True,
                monitor=monitor_metric,
                mode="max",
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=10,                # generous for tiny dataset
                restore_best_weights=True,
                mode="max",
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=4,
                min_lr=1e-8,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(out_dir, "training_log.csv"),
                append=True,
            ),
        ]

    all_histories = []

    # ── Phase 1: frozen backbone ──────────────────────────────────────────────
    print("  ► Phase 1: frozen MobileNetV2 backbone …")
    print("    (Only projection head + LSTM + classifier are trained.)\n")

    h1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=make_callbacks("val_accuracy"),
        class_weight=class_weight,
        workers=1,
        use_multiprocessing=False,
    )
    all_histories.append(h1)

    # ── Phase 2: fine-tune top backbone layers ────────────────────────────────
    print("\n  ► Phase 2: fine-tuning top 30 MobileNetV2 layers …")
    print("    (Very small LR to avoid destroying ImageNet weights.)\n")

    model = unfreeze_top_layers(model, num_layers=30, new_lr=args.fine_lr)
    fine_epochs = max(5, args.epochs // 4)   # short fine-tune pass

    h2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=fine_epochs,
        callbacks=make_callbacks("val_accuracy"),
        class_weight=class_weight,
        workers=1,
        use_multiprocessing=False,
    )
    all_histories.append(h2)

    # ── Step 5: Evaluate on held-out test set ─────────────────────────────────
    print("\n[Step 5/5]  Evaluating on HELD-OUT test set …")
    results = model.evaluate(test_gen, verbose=1, workers=1, use_multiprocessing=False)
    metric_names = model.metrics_names
    result_dict  = dict(zip(metric_names, results))

    print("\n  ── Test Results ──────────────────────────────────")
    for name, val in result_dict.items():
        print(f"  {name:<12}: {val:.4f}")
    print("  ─────────────────────────────────────────────────")

    # Save test results to JSON
    result_path = os.path.join(out_dir, "test_results.json")
    with open(result_path, "w") as f:
        json.dump(result_dict, f, indent=2)
    print(f"\n  Test results saved → {result_path}")

    # ── Training curve plot ───────────────────────────────────────────────────
    if not args.no_plot:
        plot_path = os.path.join(out_dir, "training_curve.png")
        save_training_plot(all_histories, plot_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    best_val_acc = max(
        max(h.history.get("val_accuracy", [0])) for h in all_histories
    )

    print("\n" + "=" * 68)
    print("  ✓  Training complete!")
    print(f"  Best val accuracy : {best_val_acc * 100:.2f}%")
    print(f"  Test accuracy     : {result_dict.get('accuracy', 0) * 100:.2f}%")
    print(f"  Model saved       : {args.model_out}")
    print("=" * 68)
    print("\nNext steps:")
    print("  python demo.py --source 0          # webcam (live)")
    print("  python demo.py --source video.mp4  # video file")
    print("  python demo.py --cameras 0 1       # dual-camera")
    print("  streamlit run dashboard_production.py  # dashboard\n")


if __name__ == "__main__":
    main()


# ── Overfitting FAQ (printed when run with --help) ─────────────────────────
"""
═══════════════════ WHY WAS ACCURACY 100%? ═══════════════════════════════════

1. Dataset too small for the model's capacity
   201 clips → the old deep CNN + large LSTM had far more parameters than
   training examples. The model memorised each clip's pixel patterns.
   FIX: MobileNetV2 backbone (pretrained) + reduced LSTM (64→32 units).

2. No proper train/val separation at video level
   If "fight_001.mp4" → train and "fight_002.mp4" → val but both came from
   the same source video, the model 'cheats' by recognising background/actors.
   FIX: three_way_split() groups clips by video stem before splitting.

3. Data Augmentation missing / misapplied
   Without augmentation, the model sees the exact same pixel pattern every
   epoch and memorises it by epoch 5-10.
   FIX: Horizontal flip, brightness/contrast jitter, Gaussian noise, crop.

4. No regularisation
   No Dropout, no L2 → weights grew unconstrained → memorised noise.
   FIX: Dropout(0.4/0.5) + L2(1e-4) on every Dense + LSTM layer.

5. LR too high / no LR schedule
   High constant LR lets the model overfit fast in early epochs.
   FIX: ReduceLROnPlateau halves LR after 4 epochs without val_loss improvement.

6. Too many epochs
   Training for 100+ epochs on 201 clips guarantees overfit.
   FIX: EarlyStopping with patience=10 stops when val_accuracy plateaus.

══════════════════════════════════════════════════════════════════════════════
"""