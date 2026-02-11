import os
import random
from typing import List

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from patching import PatchConfig, extract_line_patches_from_gray
from utils import (
    ensure_dir,
    extract_writer_label,
    is_png,
    load_gray_image,
    save_class_map,
)

# -----------------------------
# Paths
# -----------------------------
TRAIN_DIR = "dataset/train"
PROCESSED_TRAIN_DIR = "processed/train"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "writer_model.keras")
CLASS_MAP_PATH = "class_map.json"

# -----------------------------
# Patch + model settings
# -----------------------------
CFG = PatchConfig(
    patch_h=128,
    patch_w=256,
    seg_per_line=8,
    min_ink_ratio=0.01,
)

IMG_H, IMG_W = CFG.patch_h, CFG.patch_w
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_patches_for_training() -> List[str]:
    """
    Generate and save patches from dataset/train.
    Patches are extracted using line detection + crop/pad only (no resizing).
    Returns the list of writer labels found in the training set.
    """
    ensure_dir(PROCESSED_TRAIN_DIR)

    train_files = sorted([f for f in os.listdir(TRAIN_DIR) if is_png(f)])
    labels = sorted({extract_writer_label(f) for f in train_files})

    # Create label dirs
    for lab in labels:
        ensure_dir(os.path.join(PROCESSED_TRAIN_DIR, lab))

    total_saved = 0

    for fname in train_files:
        label = extract_writer_label(fname)
        in_path = os.path.join(TRAIN_DIR, fname)
        out_dir = os.path.join(PROCESSED_TRAIN_DIR, label)

        gray = load_gray_image(in_path)  # uint8 grayscale
        patches = extract_line_patches_from_gray(gray, CFG)

        # Save patches
        base = os.path.splitext(fname)[0]
        for i, p in enumerate(patches):
            out_name = f"{label}_{base}_P{i:04d}.png"
            out_path = os.path.join(out_dir, out_name)

            # Patch should already be (128,256) due to patching.py (crop/pad only)
            # Save as grayscale png
            import cv2

            cv2.imwrite(out_path, p)
            total_saved += 1

    print(f"[INFO] Saved train patches: {total_saved}")
    print(f"[INFO] Writers/classes: {len(labels)}")
    return labels


def build_model(num_classes: int) -> keras.Model:
    """
    CNN classifier for writer identification.
    Input shape is fixed (128x256x1) and obtained via crop/pad only.
    """
    inputs = keras.Input(shape=(IMG_H, IMG_W, 1))

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(256, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    training: bool,
) -> tf.data.Dataset:
    """
    Build tf.data pipeline for patch images saved on disk.
    No scaling is performed here; as a safety net we use crop_or_pad only.
    """

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=1)          # (H,W,1)
        img = tf.image.convert_image_dtype(img, tf.float32) # [0,1]
        img = 1.0 - img                                     # ink bright

        # Safety only: crop/pad to expected size WITHOUT resizing/scaling
        img = tf.image.resize_with_crop_or_pad(img, IMG_H, IMG_W)

        if training:
            # Light photometric augmentation only (does not distort geometry)
            img = tf.image.random_brightness(img, 0.12)
            img = tf.image.random_contrast(img, 0.85, 1.15)

        y = tf.one_hot(label, num_classes)
        return img, y

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(8000, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def main() -> None:
    set_seeds(SEED)

    # 1) Generate patches
    labels = save_patches_for_training()

    # 2) Create class map (label -> index)
    class_to_idx = {lab: i for i, lab in enumerate(labels)}
    save_class_map(CLASS_MAP_PATH, class_to_idx)
    num_classes = len(labels)

    # 3) Gather patch paths + labels
    all_paths: List[str] = []
    all_y: List[int] = []

    for lab in labels:
        d = os.path.join(PROCESSED_TRAIN_DIR, lab)
        for f in os.listdir(d):
            if is_png(f):
                all_paths.append(os.path.join(d, f))
                all_y.append(class_to_idx[lab])

    all_paths = np.array(all_paths)
    all_y = np.array(all_y, dtype=np.int32)

    print(f"[INFO] Total patches: {len(all_paths)}")

    # Patch-level split is only for early stopping / stability
    X_train, X_val, y_train, y_val = train_test_split(
        all_paths,
        all_y,
        test_size=0.2,
        random_state=SEED,
        stratify=all_y,
    )

    train_ds = make_dataset(X_train, y_train, num_classes, training=True)
    val_ds = make_dataset(X_val, y_val, num_classes, training=False)

    # 4) Train model
    model = build_model(num_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, factor=0.5, min_lr=1e-6),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    # 5) Save model
    ensure_dir(MODEL_DIR)
    model.save(MODEL_PATH)
    print(f"[INFO] Saved model: {MODEL_PATH}")
    print(f"[INFO] Saved class map: {CLASS_MAP_PATH}")


if __name__ == "__main__":
    main()
