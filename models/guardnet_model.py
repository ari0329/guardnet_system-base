"""
GuardNet Model Architecture  ―  Anti-Overfit Edition v3
========================================================
Changes vs old version
  ✓ MobileNetV2 backbone (pretrained ImageNet, ~3.4M params, fast)
  ✓ GlobalAveragePooling2D instead of Flatten (no spatial overfit)
  ✓ BatchNormalization after every Dense / LSTM block
  ✓ Dropout(0.5) on all FC paths
  ✓ L2 regularization on Dense + LSTM kernels
  ✓ Stacked LSTM 64 → 32 (down from typical 128 → 64)
  ✓ LayerNormalization inside the recurrent stack
  ✓ unfreeze_top_layers() supports fine-tuning Phase 2
  ✓ MobileNetV2 input preprocessed automatically via Lambda
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

# ── Sequence / resolution config ────────────────────────────────────────────
SEQ_LEN   = 16    # frames per clip  (keep low for small dataset)
IMG_H     = 112   # MobileNetV2 sweet-spot (also 96, 128, 160, 192, 224)
IMG_W     = 112
CHANNELS  = 3

# ── Regularization strength ──────────────────────────────────────────────────
L2        = 1e-4


def _mobilenet_preprocess():
    """Wrap MobileNetV2 preprocess_input in a Lambda layer."""
    return layers.Lambda(
        tf.keras.applications.mobilenet_v2.preprocess_input,
        name="mobilenet_preprocess"
    )


def build_cnn_encoder(img_h: int = IMG_H, img_w: int = IMG_W) -> Model:
    """
    CNN feature extractor built on MobileNetV2.

    Output:  dense 256-D embedding per frame.
    Phase 1: backbone is FROZEN → only the head is trained.
    Phase 2: call unfreeze_top_layers() to open the top N backbone layers.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_h, img_w, CHANNELS),
        include_top=False,
        weights="imagenet",
        alpha=1.0,              # use 0.75 to shave another ~25 % params
    )
    base.trainable = False      # frozen in Phase 1

    inp = tf.keras.Input(shape=(img_h, img_w, CHANNELS), name="frame_input")

    # MobileNetV2 expects pixels in [-1, 1]
    x = _mobilenet_preprocess()(inp)
    x = base(x, training=False)           # training=False → BN in inference mode
    x = layers.GlobalAveragePooling2D()(x)

    # Projection head
    x = layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=regularizers.l2(L2),
            name="proj_dense",
        )(x)
    x = layers.BatchNormalization(name="proj_bn")(x)
    x = layers.Dropout(0.4, name="proj_drop")(x)

    return Model(inp, x, name="cnn_encoder")


def build_model(
    seq_len: int = SEQ_LEN,
    img_h:   int = IMG_H,
    img_w:   int = IMG_W,
    learning_rate: float = 1e-4,
) -> Model:
    """
    Full CNN + LSTM violence detection model.

    Architecture:
        [seq_len, H, W, 3]
            → TimeDistributed(MobileNetV2 + proj head)  → [seq_len, 256]
            → LSTM(64, return_sequences=True)            → [seq_len, 64]
            → LayerNorm → Dropout(0.4)
            → LSTM(32)                                   → [32]
            → LayerNorm → Dropout(0.5)
            → Dense(64, relu) → BN → Dropout(0.5)
            → Dense(1, sigmoid)                          → probability
    """
    cnn = build_cnn_encoder(img_h, img_w)

    inp = tf.keras.Input(
        shape=(seq_len, img_h, img_w, CHANNELS), name="video_input"
    )

    # ── TimeDistributed feature extraction ──────────────────────────────────
    x = layers.TimeDistributed(cnn, name="td_cnn")(inp)

    # ── LSTM stack ──────────────────────────────────────────────────────────
    x = layers.LSTM(
            64,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.2,
            kernel_regularizer=regularizers.l2(L2),
            recurrent_regularizer=regularizers.l2(L2),
            name="lstm_1",
        )(x)
    x = layers.LayerNormalization(name="ln_1")(x)
    x = layers.Dropout(0.4, name="drop_lstm1")(x)

    x = layers.LSTM(
            32,
            dropout=0.3,
            recurrent_dropout=0.2,
            kernel_regularizer=regularizers.l2(L2),
            recurrent_regularizer=regularizers.l2(L2),
            name="lstm_2",
        )(x)
    x = layers.LayerNormalization(name="ln_2")(x)
    x = layers.Dropout(0.5, name="drop_lstm2")(x)

    # ── Classifier head ──────────────────────────────────────────────────────
    x = layers.Dense(
            64,
            activation="relu",
            kernel_regularizer=regularizers.l2(L2),
            name="fc_1",
        )(x)
    x   = layers.BatchNormalization(name="fc_bn")(x)
    x   = layers.Dropout(0.5, name="fc_drop")(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inp, out, name="GuardNet_v3")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def unfreeze_top_layers(
    model: Model,
    num_layers: int = 30,
    new_lr:     float = 5e-6,
) -> Model:
    """
    Phase 2 fine-tuning: open the top `num_layers` of the MobileNetV2 backbone.

    Uses a very small LR (5e-6) to avoid destroying the pretrained weights.
    """
    td_layer    = model.get_layer("td_cnn")
    cnn_model   = td_layer.layer            # the cnn_encoder Model
    base        = None

    # Find MobileNetV2 inside the encoder
    for lyr in cnn_model.layers:
        if "mobilenetv2" in lyr.name.lower():
            base = lyr
            break

    if base is None:
        print("[WARN] MobileNetV2 sub-model not found. Skipping unfreeze.")
        return model

    base.trainable = True
    for layer in base.layers[:-num_layers]:
        layer.trainable = False

    trainable_count = sum(
        1 for l in base.layers if l.trainable
    )
    print(
        f"  ► Unfroze top {num_layers} MobileNetV2 layers "
        f"({trainable_count} trainable total). LR → {new_lr}"
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=new_lr),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model