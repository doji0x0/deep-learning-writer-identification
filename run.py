import os
import json
from typing import List

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from patching import PatchConfig, extract_line_patches_from_gray
from utils import is_png, load_gray_image

# -----------------------------
# Absolute Paths (Mac Safe)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
MODEL_PATH = os.path.join(BASE_DIR, "models", "writer_model.keras")
CLASS_MAP_PATH = os.path.join(BASE_DIR, "class_map.json")
OUT_CSV = os.path.join(BASE_DIR, "outputs", "result.csv")

# -----------------------------
# Patch config (MUST match training)
# -----------------------------
CFG = PatchConfig(
    patch_h=128,
    patch_w=256,
    seg_per_line=8,
    min_ink_ratio=0.01,
)

IMG_H, IMG_W = CFG.patch_h, CFG.patch_w


def load_class_map(path: str):
    with open(path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


def preprocess_patch_numpy(patch: np.ndarray) -> tf.Tensor:
    x = patch.astype(np.float32) / 255.0
    x = 1.0 - x
    x = x[..., None]
    return tf.convert_to_tensor(x)


def predict_page_probs(
    model: keras.Model,
    gray: np.ndarray,
    num_classes: int,
) -> np.ndarray:

    patches = extract_line_patches_from_gray(gray, CFG)

    if len(patches) == 0:
        raise RuntimeError("No patches extracted from test image.")

    probs_sum = np.zeros((num_classes,), dtype=np.float32)

    for p in patches:
        x = preprocess_patch_numpy(p)
        x = tf.expand_dims(x, 0)
        probs = model.predict(x, verbose=0)[0]
        probs_sum += probs

    return probs_sum


def main():
    print("[INFO] Loading model...")
    model = keras.models.load_model(MODEL_PATH)

    print("[INFO] Loading class map...")
    class_to_idx, idx_to_class = load_class_map(CLASS_MAP_PATH)
    num_classes = len(class_to_idx)

    y_true: List[str] = []
    y_pred: List[str] = []

    y_true_idx: List[int] = []
    y_score: List[np.ndarray] = []

    # -----------------------------
    # Load test images recursively
    # -----------------------------
    test_files = []

    for root, _, files in os.walk(TEST_DIR):
        for f in files:
            if is_png(f):
                test_files.append(os.path.join(root, f))

    test_files = sorted(test_files)

    print(f"[INFO] Test images found: {len(test_files)}")

    for filepath in test_files:
        fname = os.path.basename(filepath)
        true_label = fname[:2]

        gray = load_gray_image(filepath)

        if gray is None:
            print(f"[WARNING] Could not load image: {filepath}")
            continue

        probs = predict_page_probs(model, gray, num_classes)

        pred_idx = int(np.argmax(probs))
        pred_label = idx_to_class[pred_idx]

        y_true.append(true_label)
        y_pred.append(pred_label)

        y_true_idx.append(class_to_idx[true_label])
        y_score.append(probs)

    # -----------------------------
    # Metrics
    # -----------------------------
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    y_true_oh = keras.utils.to_categorical(y_true_idx, num_classes)
    y_score = np.vstack(y_score)

    auc = roc_auc_score(
        y_true_oh,
        y_score,
        average="macro",
        multi_class="ovr"
    )

    print(f"[RESULT] Test Accuracy : {acc:.4f}")
    print(f"[RESULT] Macro F1      : {f1:.4f}")
    print(f"[RESULT] Macro AUC     : {auc:.4f}")

    # -----------------------------
    # Save CSV
    # -----------------------------
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)

    with open(OUT_CSV, "w") as f:
        f.write("filename,actual_label,predicted_label\n")
        for filepath, yt, yp in zip(test_files, y_true, y_pred):
            fname = os.path.basename(filepath)
            f.write(f"{fname},{yt},{yp}\n")

    print(f"[INFO] Saved results to: {OUT_CSV}")


if __name__ == "__main__":
    main()
