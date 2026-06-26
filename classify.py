#!/usr/bin/env python3
"""MTARSI Aircraft Classification — Kaggle-ready single-file training script.

Two-phase transfer learning (default) with an optional from-scratch ResNet+SE
fallback. Self-contained: model, data, training, evaluation, inference.

FIX SUMMARY (vs. the broken version):
  • EfficientNet now uses the official `preprocess_input` INSIDE the model.
  • Removed the hardcoded `base(inputs, training=False)` that pinned BatchNorm
    to ImageNet stats against raw 0–255 inputs (caused single-class collapse).
  • Transfer-mode data pipeline now feeds raw 0–255 (model normalizes itself).
  • Inference feeds raw 0–255 for transfer mode — no double normalization.

Dataset layout:
    dataset_root/
        A-10/        A-10_001.jpg …
        B-1/         B-1_001.jpg …
        …
        Twin-Prop/   Twin-Prop_100.jpg …
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────

DATA_ROOT = "/MTARSI"
OUTPUT_DIR = "/output"

MODE = "transfer"          # "transfer" (EfficientNetB0) or "scratch" (ResNet+SE)
IMG_SIZE = 128
BATCH_SIZE = 32

# Phase 1: train the head with backbone frozen
EPOCHS_HEAD = 25
LR_HEAD = 1e-3

# Phase 2: fine-tune top of backbone (transfer mode only)
EPOCHS_FINETUNE = 40
LR_FINETUNE = 1e-5
UNFREEZE_FROM = 0.5        # unfreeze the top 50% of backbone layers

# Scratch-mode training (used only when MODE == "scratch")
EPOCHS_SCRATCH = 80
LR_SCRATCH = 1e-3
WEIGHT_DECAY = 1e-4        # AdamW weight decay (replaces per-layer L2)
DROPOUT = 0.5
USE_SE = True

# Callbacks
PATIENCE = 12
LR_PATIENCE = 5
LR_FACTOR = 0.5
MONITOR = "val_loss"       # single metric used by all callbacks
MONITOR_MODE = "min"

# Splits
VAL_SPLIT = 0.15
TEST_SPLIT = 0.10
SEED = 42

# ──────────────────────────────────────────────────────────────────────
#  MODELS
# ──────────────────────────────────────────────────────────────────────

def squeeze_excite(x: tf.Tensor, reduction: int = 16) -> tf.Tensor:
    channels = x.shape[-1]
    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Reshape((1, 1, channels))(se)
    se = layers.Dense(max(channels // reduction, 4), activation="relu", use_bias=False)(se)
    se = layers.Dense(channels, activation="sigmoid", use_bias=False)(se)
    return layers.Multiply()([x, se])


def residual_block(x, filters, stride=1, se=True, reduction=16, name=""):
    shortcut = x
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False,
                                 name=f"{name}_skip")(x)
        shortcut = layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)

    out = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False,
                        name=f"{name}_conv1")(x)
    out = layers.BatchNormalization(name=f"{name}_bn1")(out)
    out = layers.ReLU(name=f"{name}_relu1")(out)
    out = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                        name=f"{name}_conv2")(out)
    out = layers.BatchNormalization(name=f"{name}_bn2")(out)

    if se:
        out = squeeze_excite(out, reduction)

    out = layers.Add(name=f"{name}_add")([shortcut, out])
    return layers.ReLU(name=f"{name}_relu2")(out)


def build_scratch_cnn(input_shape, num_classes, dropout=0.5, se=True) -> keras.Model:
    """Custom ResNet+SE. Regularization handled by AdamW weight_decay (no L2 here)."""
    inputs = keras.Input(shape=input_shape, name="input")
    x = layers.Conv2D(64, 3, padding="same", use_bias=False, name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)

    for i in range(3):
        x = residual_block(x, 64, 1, se, name=f"s1_b{i}")
    for i in range(4):
        x = residual_block(x, 128, 2 if i == 0 else 1, se, name=f"s2_b{i}")
    for i in range(6):
        x = residual_block(x, 256, 2 if i == 0 else 1, se, name=f"s3_b{i}")
    for i in range(3):
        x = residual_block(x, 512, 2 if i == 0 else 1, se, name=f"s4_b{i}")

    x = layers.GlobalAveragePooling2D(name="head_pool")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    return keras.Model(inputs, outputs, name="MTARSI_ResNetSE")


def build_transfer_model(input_shape, num_classes, dropout=0.5):
    """EfficientNetB0 backbone.

    FIX: feeds raw 0–255 through the official `preprocess_input`, and does NOT
    hardcode `training=False`. Keras manages BatchNorm mode automatically:
    inference mode while frozen, proper updates once unfrozen in Phase 2.
    """
    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    base.trainable = False

    inputs = keras.Input(shape=input_shape, name="input")
    # EfficientNet's official preprocessing expects raw 0–255 pixels.
    x = keras.applications.efficientnet.preprocess_input(inputs)
    x = base(x)                              # ← no hardcoded training=False
    x = layers.GlobalAveragePooling2D(name="head_pool")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="output")(x)
    model = keras.Model(inputs, outputs, name="MTARSI_EfficientNetB0")
    model.base_model = base  # keep handle for fine-tuning
    return model


# ──────────────────────────────────────────────────────────────────────
#  DATA
# ──────────────────────────────────────────────────────────────────────

def load_image_paths(data_root):
    root = Path(data_root)
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    class_names = [d.name for d in class_dirs]
    paths, labels = [], []
    for idx, cdir in enumerate(class_dirs):
        for img in sorted(cdir.glob("*.jpg")):
            paths.append(str(img))
            labels.append(idx)
    return paths, np.array(labels), class_names


def split_dataset(paths, labels, val_split, test_split, seed):
    idx = np.arange(len(paths))
    trainval, test = train_test_split(idx, test_size=test_split, stratify=labels, random_state=seed)
    val_frac = val_split / (1 - test_split)
    train, val = train_test_split(trainval, test_size=val_frac,
                                  stratify=labels[trainval], random_state=seed)
    print(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return {"train": train, "val": val, "test": test}


def compute_weights(labels):
    w = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
    return {i: float(x) for i, x in enumerate(w)}


# Single source of truth for preprocessing (used by train AND inference).
# IMPORTANT (transfer mode): we feed RAW 0–255 here; normalization happens
# INSIDE the model via efficientnet.preprocess_input. Do NOT scale to [0,1].
def decode_image(file_path, img_size, normalize_01):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32)
    if normalize_01:               # scratch mode only: scale to [0,1]
        img = img / 255.0
    # transfer mode: keep raw 0–255; model normalizes internally
    return img


def augment_image(img):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.image.random_saturation(img, lower=0.85, upper=1.15)
    # Brightness/contrast/saturation can push values slightly out of range;
    # clip to the valid pixel range expected by the chosen preprocessing.
    upper = 1.0 if img.dtype == tf.float32 and tf.reduce_max(img) <= 1.0 else 255.0
    img = tf.clip_by_value(img, 0.0, upper)
    return img


def create_tf_datasets(data_root, batch_size, val_split, test_split, seed, normalize_01):
    paths, labels, class_names = load_image_paths(data_root)
    splits = split_dataset(paths, labels, val_split, test_split, seed)
    paths = np.array(paths)
    AUTOTUNE = tf.data.AUTOTUNE

    def make(split, training):
        ds = tf.data.Dataset.from_tensor_slices((paths[splits[split]], labels[splits[split]]))
        # Decode once, then cache decoded tensors in RAM.
        ds = ds.map(lambda p, l: (decode_image(p, IMG_SIZE, normalize_01), l),
                    num_parallel_calls=AUTOTUNE).cache()
        if training:
            ds = ds.shuffle(len(splits[split]), seed=seed, reshuffle_each_iteration=True)
            ds = ds.map(lambda x, l: (augment_image(x), l), num_parallel_calls=AUTOTUNE)
        return ds.batch(batch_size).prefetch(AUTOTUNE)

    return {
        "train_ds": make("train", True),
        "val_ds": make("val", False),
        "test_ds": make("test", False),
        "class_names": class_names,
        "class_weight": compute_weights(labels[splits["train"]]),
        "num_classes": len(class_names),
        "test_paths": list(paths[splits["test"]]),
        "normalize_01": normalize_01,
    }


# ──────────────────────────────────────────────────────────────────────
#  TRAINING HELPERS
# ──────────────────────────────────────────────────────────────────────

def build_callbacks(output_dir, ckpt_name, patience, lr_factor, lr_patience):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(str(out / ckpt_name), monitor=MONITOR,
                                        save_best_only=True, mode=MONITOR_MODE, verbose=1),
        keras.callbacks.EarlyStopping(monitor=MONITOR, patience=patience, mode=MONITOR_MODE,
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor=MONITOR, factor=lr_factor, patience=lr_patience,
                                          mode=MONITOR_MODE, min_lr=1e-7, verbose=1),
        keras.callbacks.CSVLogger(str(out / "history.csv"), append=True),
        keras.callbacks.TerminateOnNaN(),
    ]


def merge_history(h1, h2):
    if h1 is None:
        return h2.history
    merged = {}
    for k in h1:
        merged[k] = h1[k] + h2.history.get(k, [])
    return merged


def collect_predictions(model, dataset):
    probs = model.predict(dataset, verbose=0)
    y_pred = np.argmax(probs, axis=1)
    y_true = np.concatenate([y.numpy() for _, y in dataset])
    return y_true, y_pred


def plot_history(history, output_dir):
    out = Path(output_dir)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["accuracy"], label="train")
    ax1.plot(history["val_accuracy"], label="val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)
    ax2.plot(history["loss"], label="train")
    ax2.plot(history["val_loss"], label="val")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=120); plt.close(fig)
    print(f"Training curves → {out / 'training_curves.png'}")


def plot_confusion(cm, class_names, output_dir):
    out = Path(output_dir)
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    n = len(class_names)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, shrink=0.78)
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=120); plt.close(fig)
    print(f"Confusion matrix → {out / 'confusion_matrix.png'}")


def evaluate_and_report(model, datasets, output_dir, full_history):
    out = Path(output_dir)
    res = model.evaluate(datasets["test_ds"], verbose=0, return_dict=True)
    print(f"\n{'='*60}\n  Test accuracy: {res['accuracy']:.4f}  |  "
          f"Test loss: {res['loss']:.4f}\n{'='*60}")

    y_true, y_pred = collect_predictions(model, datasets["test_ds"])
    print("\n", classification_report(y_true, y_pred,
                                      target_names=datasets["class_names"], zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(out / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    plot_history(full_history, output_dir)
    plot_confusion(cm, datasets["class_names"], output_dir)

    model.save(out / "final_model.keras")
    (out / "class_names.txt").write_text("\n".join(datasets["class_names"]))
    print(f"\nModel → {out / 'final_model.keras'}\nClasses → {out / 'class_names.txt'}")


def sanity_check_not_collapsed(model, normalize_01):
    """Quick probe: a healthy model must give DIFFERENT / non-saturated outputs
    for zeros vs ones vs noise. If everything saturates to the same class at
    1.0, the backbone has collapsed again."""
    print("\n" + "=" * 60 + "\n  Sanity check — input-dependence probe\n" + "=" * 60)
    hi = 1.0 if normalize_01 else 255.0
    probes = {
        "zeros": np.zeros((1, IMG_SIZE, IMG_SIZE, 3), "float32"),
        "ones":  np.ones((1, IMG_SIZE, IMG_SIZE, 3), "float32") * hi,
        "noise": np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype("float32") * hi,
    }
    argmaxes = []
    for name, x in probes.items():
        p = model.predict(x, verbose=0)[0]
        argmaxes.append(int(p.argmax()))
        print(f"  {name:6} argmax {p.argmax():2d}  max {p.max():.4f}")
    if len(set(argmaxes)) == 1:
        print("  ⚠️  All probes gave the SAME class — model may still be collapsed.")
    else:
        print("  ✓  Outputs are input-dependent.")


# ──────────────────────────────────────────────────────────────────────
#  TRAINING DRIVERS
# ──────────────────────────────────────────────────────────────────────

def train_transfer(model, datasets, output_dir):
    cw = datasets["class_weight"]

    # ── Phase 1: frozen backbone, train head ──
    print("\n" + "=" * 60 + "\n  PHASE 1 — training head (backbone frozen)\n" + "=" * 60)
    model.compile(optimizer=keras.optimizers.Adam(LR_HEAD),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(datasets["train_ds"], validation_data=datasets["val_ds"],
                   epochs=EPOCHS_HEAD, class_weight=cw,
                   callbacks=build_callbacks(output_dir, "best_head.keras",
                                             PATIENCE, LR_FACTOR, LR_PATIENCE), verbose=1)
    history = h1.history

    # ── Phase 2: unfreeze top of backbone, fine-tune ──
    print("\n" + "=" * 60 + "\n  PHASE 2 — fine-tuning backbone\n" + "=" * 60)
    base = model.base_model
    base.trainable = True
    cutoff = int(len(base.layers) * UNFREEZE_FROM)
    for layer in base.layers[:cutoff]:
        layer.trainable = False
    # Keep BatchNorm frozen during fine-tuning for stability. This is now safe:
    # Phase 1 already produced real features because the backbone runs in proper
    # inference mode against correctly normalized inputs.
    for layer in base.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(optimizer=keras.optimizers.Adam(LR_FINETUNE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(datasets["train_ds"], validation_data=datasets["val_ds"],
                   epochs=EPOCHS_FINETUNE, class_weight=cw,
                   callbacks=build_callbacks(output_dir, "best_finetune.keras",
                                             PATIENCE, LR_FACTOR, LR_PATIENCE), verbose=1)
    return model, merge_history(history, h2)


def train_scratch(model, datasets, output_dir):
    print("\n" + "=" * 60 + "\n  Training from scratch (ResNet+SE)\n" + "=" * 60)
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LR_SCRATCH, weight_decay=WEIGHT_DECAY),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(datasets["train_ds"], validation_data=datasets["val_ds"],
                  epochs=EPOCHS_SCRATCH, class_weight=datasets["class_weight"],
                  callbacks=build_callbacks(output_dir, "best_model.keras",
                                            PATIENCE, LR_FACTOR, LR_PATIENCE), verbose=1)
    return model, h.history


# ──────────────────────────────────────────────────────────────────────
#  INFERENCE  (uses identical TF preprocessing as training)
# ──────────────────────────────────────────────────────────────────────

def infer_image(model, image_path, class_names, normalize_01, top_k=3):
    # Transfer mode: feed RAW 0–255 (model normalizes internally).
    # Scratch mode: feed [0,1]. decode_image handles both via normalize_01.
    x = decode_image(tf.constant(image_path), IMG_SIZE, normalize_01)
    x = tf.expand_dims(x, 0)
    probs = model.predict(x, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    print(f"\nImage: {image_path}\n" + "-" * 40)
    for i, idx in enumerate(top_idx):
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        print(f"  {i + 1}. {name:<25s} {probs[idx] * 100:6.2f}%")


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60 + "\n  MTARSI Aircraft Classifier — Kaggle Training\n" + "=" * 60)
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow {tf.__version__}  |  GPUs: {len(gpus)}  |  MODE: {MODE}")

    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    # Transfer mode: feed raw 0–255 (model applies efficientnet.preprocess_input).
    # Scratch mode: scale to [0,1].
    normalize_01 = (MODE == "scratch")

    print(f"\nData root: {DATA_ROOT}")
    datasets = create_tf_datasets(DATA_ROOT, BATCH_SIZE, VAL_SPLIT, TEST_SPLIT, SEED, normalize_01)
    print(f"Classes ({datasets['num_classes']}): {datasets['class_names']}")

    input_shape = (IMG_SIZE, IMG_SIZE, 3)
    if MODE == "transfer":
        model = build_transfer_model(input_shape, datasets["num_classes"], DROPOUT)
        model.summary()
        model, full_history = train_transfer(model, datasets, OUTPUT_DIR)
    elif MODE == "scratch":
        model = build_scratch_cnn(input_shape, datasets["num_classes"], DROPOUT, USE_SE)
        model.summary()
        model, full_history = train_scratch(model, datasets, OUTPUT_DIR)
    else:
        raise ValueError(f"Unknown MODE: {MODE!r} (use 'transfer' or 'scratch')")

    evaluate_and_report(model, datasets, OUTPUT_DIR, full_history)

    # Confirm the backbone is alive (catches the old collapse bug immediately).
    sanity_check_not_collapsed(model, datasets["normalize_01"])

    print("\n" + "=" * 60 + "\n  Inference Demo (sample test images)\n" + "=" * 60)
    for img_path in datasets["test_paths"][:5]:
        infer_image(model, img_path, datasets["class_names"], datasets["normalize_01"])

    print("\n✓ Done.")


if __name__ == "__main__":
    main()