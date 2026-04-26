"""
GuardNet – CNN + LSTM Model
Spatial features (MobileNetV2 / ResNet50) → LSTM temporal classifier.
"""

import os
import sys
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2, ResNet50

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    FRAME_WIDTH, FRAME_HEIGHT, SEQUENCE_LENGTH,
    CNN_BACKBONE, LSTM_UNITS, DROPOUT_RATE, NUM_CLASSES,
    LEARNING_RATE, BATCH_SIZE, EPOCHS,
    VALIDATION_SPLIT, EARLY_STOPPING_PATIENCE,
    MODEL_PATH
)


# ─── Feature Extractor ────────────────────────────────────────────────────────

def build_cnn_encoder(backbone: str = CNN_BACKBONE) -> tf.keras.Model:
    """Return a frozen ImageNet-pretrained CNN without the top classifier."""
    input_shape = (FRAME_HEIGHT, FRAME_WIDTH, 3)

    if backbone == "MobileNetV2":
        base = MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
    elif backbone == "ResNet50":
        base = ResNet50(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet",
            pooling="avg"
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # Freeze first 80 % of layers; fine-tune the rest
    freeze_until = int(len(base.layers) * 0.8)
    for layer in base.layers[:freeze_until]:
        layer.trainable = False
    for layer in base.layers[freeze_until:]:
        layer.trainable = True

    return base


# ─── Full CNN-LSTM Architecture ───────────────────────────────────────────────

def build_model(
    seq_len: int = SEQUENCE_LENGTH,
    backbone: str = CNN_BACKBONE,
) -> tf.keras.Model:
    """
    TimeDistributed CNN → BiLSTM → Dense classifier.
    Input: (batch, seq_len, H, W, 3)
    Output: (batch, 2) softmax probabilities [non-violent, violent]
    """
    cnn_encoder = build_cnn_encoder(backbone)
    feature_dim = cnn_encoder.output_shape[-1]   # e.g. 1280 for MobileNetV2

    # ── Input ──
    inp = layers.Input(shape=(seq_len, FRAME_HEIGHT, FRAME_WIDTH, 3),
                       name="frame_sequence")

    # ── Spatial features (applied independently per frame) ──
    x = layers.TimeDistributed(cnn_encoder, name="cnn_encoder")(inp)
    x = layers.TimeDistributed(layers.BatchNormalization())(x)
    x = layers.TimeDistributed(layers.Dropout(DROPOUT_RATE * 0.5))(x)

    # ── Temporal reasoning ──
    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS, return_sequences=True), name="bilstm_1"
    )(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    x = layers.Bidirectional(
        layers.LSTM(LSTM_UNITS // 2, return_sequences=False), name="bilstm_2"
    )(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    # ── Classifier head ──
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(DROPOUT_RATE * 0.5)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="class_probs")(x)

    model = models.Model(inputs=inp, outputs=out, name="GuardNet")
    return model


# ─── Compile Helper ───────────────────────────────────────────────────────────

def compile_model(model: tf.keras.Model) -> tf.keras.Model:
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ─── Training Pipeline ────────────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray, model_save_path: str = MODEL_PATH):
    """
    Train GuardNet on pre-loaded sequences.
    X: shape (N, seq_len, H, W, 3)
    y: shape (N,)  values in {0, 1}
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    model = build_model()
    compile_model(model)
    model.summary()

    cb_list = [
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=3, min_lr=1e-6, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(os.path.dirname(model_save_path), "tb_logs"),
            histogram_freq=1
        ),
    ]

    history = model.fit(
        X, y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT,
        callbacks=cb_list,
        shuffle=True
    )
    print(f"[INFO] Model saved → {model_save_path}")
    return model, history


# ─── Inference Helper ─────────────────────────────────────────────────────────

class GuardNetInference:
    """Lightweight wrapper for real-time inference."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at {model_path}.\n"
                "Run  python train.py  first, or place a .h5 file there."
            )
        print(f"[INFO] Loading model from {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.model.make_predict_function()   # warm-up JIT
        print("[INFO] Model ready.")

    def predict(self, sequence: np.ndarray) -> float:
        """
        sequence: np.ndarray shape (1, seq_len, H, W, 3)
        Returns violence probability (float in [0, 1]).
        """
        probs = self.model.predict(sequence, verbose=0)[0]
        return float(probs[1])              # index 1 = violent class


# ─── Utility: create a lightweight demo model (no real training needed) ──────

def create_demo_model(save_path: str = MODEL_PATH):
    """
    Builds and saves an UNTRAINED model so the demo script can run
    without a dataset.  Weights are random – predictions are meaningless
    but the pipeline works end-to-end.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model = build_model()
    compile_model(model)
    model.save(save_path)
    print(f"[INFO] Demo (untrained) model saved → {save_path}")
    return model


if __name__ == "__main__":
    # Quick architecture sanity-check
    m = build_model()
    m.summary()
