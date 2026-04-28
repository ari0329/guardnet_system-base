"""
GuardNet — CNN + Bidirectional LSTM Violence Detection Model

Architecture:
    Input  (batch, SEQ_LEN, 224, 224, 3)
        │
    TimeDistributed(MobileNetV2 — ImageNet pretrained)   ← spatial features
        │
    TimeDistributed(BatchNorm + Dropout)
        │
    Bidirectional LSTM (256 units, return_sequences=True) ← temporal reasoning
        │  Dropout
    Bidirectional LSTM (128 units)
        │  Dropout
    Dense(128, ReLU) → Dropout → Dense(2, Softmax)
        │
    [P(non-violent), P(violent)]
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    MODEL_PATH, CNN_BACKBONE, LSTM_UNITS,
    SEQUENCE_LENGTH, FRAME_SIZE, NUM_CLASSES,
    BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, LEARNING_RATE,
)


# ══════════════════════════════════════════════════════════════════════════════
# Build Model
# ══════════════════════════════════════════════════════════════════════════════

def build_model(
    sequence_length: int   = SEQUENCE_LENGTH,
    frame_size:      tuple = FRAME_SIZE,
    backbone:        str   = CNN_BACKBONE,
    lstm_units:      int   = LSTM_UNITS,
    num_classes:     int   = NUM_CLASSES,
):
    """Return a compiled Keras CNN-LSTM model ready for training."""
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2, ResNet50

    input_shape = (*frame_size, 3)

    # ── CNN Backbone (frozen initially) ────────────────────────────────────
    if backbone == "ResNet50":
        base_cnn = ResNet50(
            include_top=False, weights="imagenet",
            input_shape=input_shape, pooling="avg",
        )
    else:  # default: MobileNetV2
        base_cnn = MobileNetV2(
            include_top=False, weights="imagenet",
            input_shape=input_shape, pooling="avg",
        )
    base_cnn.trainable = False   # frozen during Phase 1

    # ── Sequence Input ─────────────────────────────────────────────────────
    seq_input = layers.Input(
        shape=(sequence_length, *frame_size, 3),
        name="sequence_input",
    )

    # ── Spatial Feature Extraction (per frame) ────────────────────────────
    x = layers.TimeDistributed(base_cnn,                      name="td_backbone")(seq_input)
    x = layers.TimeDistributed(layers.BatchNormalization(),   name="td_bn")(x)
    x = layers.TimeDistributed(layers.Dropout(0.3),           name="td_drop")(x)

    # ── Temporal Reasoning ────────────────────────────────────────────────
    x = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, dropout=0.3),
            name="bilstm_1",
        )(x)
    x = layers.Dropout(0.4, name="drop_1")(x)

    x = layers.Bidirectional(
            layers.LSTM(lstm_units // 2, dropout=0.3),
            name="bilstm_2",
        )(x)
    x = layers.Dropout(0.4, name="drop_2")(x)

    # ── Classifier Head ───────────────────────────────────────────────────
    x   = layers.Dense(128, activation="relu", name="fc_128")(x)
    x   = layers.Dropout(0.5, name="drop_cls")(x)
    out = layers.Dense(num_classes, activation="softmax", name="output")(x)

    model = models.Model(inputs=seq_input, outputs=out, name="GuardNet")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Fine-Tune Helper  ← THIS WAS MISSING IN YOUR OLD FILE
# ══════════════════════════════════════════════════════════════════════════════

def unfreeze_top_layers(model, num_layers: int = 30):
    """
    Unfreeze the last N layers of the CNN backbone for fine-tuning (Phase 2).
    Recompiles the model with a lower learning rate.

    Args:
        model      : compiled Keras model returned by build_model()
        num_layers : how many backbone layers to unfreeze (default 30)

    Returns:
        model  (recompiled, ready to continue training)
    """
    import tensorflow as tf

    try:
        backbone = model.get_layer("td_backbone").layer
        for layer in backbone.layers[-num_layers:]:
            layer.trainable = True
        print(f"  [unfreeze] Unfroze top {num_layers} backbone layers.")
    except Exception as e:
        print(f"  [unfreeze] Warning: {e} — skipping fine-tune phase.")
        return model

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE / 10   # 10× smaller LR for fine-tuning
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Legacy train() — kept for backwards compatibility with old train.py
# (new train.py calls model.fit directly with generators)
# ══════════════════════════════════════════════════════════════════════════════

def train(
    X:               np.ndarray,
    y:               np.ndarray,
    model_save_path: str   = MODEL_PATH,
    epochs:          int   = EPOCHS,
    batch_size:      int   = BATCH_SIZE,
    val_split:       float = VALIDATION_SPLIT,
):
    """
    Legacy in-memory training.
    ⚠️  Use the new train.py with VideoDataGenerator for large datasets.
    """
    import tensorflow as tf

    os.makedirs(os.path.dirname(model_save_path) or ".", exist_ok=True)
    model = build_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_save_path, save_best_only=True,
            monitor="val_accuracy", verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3,
            min_lr=1e-6, verbose=1,
        ),
    ]

    history = model.fit(
        X, y,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=val_split,
        callbacks=callbacks,
        shuffle=True,
    )
    model.save(model_save_path)
    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# Inference Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class GuardNetInference:
    """
    Loads a saved .h5 model and runs real-time inference.
    Thread-safe — safe to call from a background detection thread.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[GuardNet] Model not found: {model_path}\n"
                "Train first:  python train.py --data_dir ./data --epochs 30"
            )
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)

        # Warm-up pass — JIT-compiles graph ops on first call
        dummy = np.zeros(
            (1, SEQUENCE_LENGTH, *FRAME_SIZE, 3), dtype=np.float32
        )
        self.model.predict(dummy, verbose=0)
        print(f"[GuardNet] Model loaded ✓  ({model_path})")

    def predict(self, sequence: np.ndarray) -> float:
        """
        Args:
            sequence : float32 array  (1, SEQ_LEN, H, W, 3)  normalised [0, 1]
        Returns:
            Violence probability in [0.0, 1.0]
        """
        if sequence.ndim == 4:
            sequence = sequence[np.newaxis]        # add batch dim
        preds = self.model.predict(sequence, verbose=0)
        return float(preds[0, 1])                  # index 1 = violent class