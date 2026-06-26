#!/usr/bin/env python3
"""MTARSI Aircraft Classifier — Inference + Prediction Logging.

Loads the trained model + class names, predicts on one image / folder / glob,
and SAVES results to a predictions folder (CSV + JSON + summary).

Preprocessing MUST match training:
    Transfer model (final_model.keras): preprocess_input baked inside → feed RAW 0–255.
    Scratch  model (best_model.keras) : no internal preprocessing      → feed [0,1].
Auto-detected from model.name so you can't double-normalize.

Usage:
    python infer.py --image plane.jpg
    python infer.py --dir ./test_images --top-k 5
    python infer.py --glob "MTARSI/F-22/*.jpg" --out-dir predictions
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ──────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────

MODEL_PATH = "output/final_model.keras"
CLASS_NAMES_PATH = "output/class_names.txt"
OUTPUT_DIR = "predictions"
IMG_SIZE = 128
DEFAULT_TOP_K = 3

# ──────────────────────────────────────────────────────────────────────
#  LOAD
# ──────────────────────────────────────────────────────────────────────

def load_class_names(path):
    names = Path(path).read_text().strip().splitlines()
    if not names:
        raise ValueError(f"No class names found in {path}")
    return names


def model_normalizes_internally(model):
    """Transfer model bakes efficientnet.preprocess_input inside → feed 0–255.
    Scratch model does not → feed [0,1]. Detect by model name."""
    name = model.name.lower()
    if "efficientnet" in name:
        return True
    if "resnetse" in name:
        return False
    print(f"⚠️  Unknown model name '{model.name}'. Assuming raw 0–255 input.")
    return True


# ──────────────────────────────────────────────────────────────────────
#  PREPROCESS  (identical to training's decode_image)
# ──────────────────────────────────────────────────────────────────────

def preprocess(image_path, img_size, normalize_01):
    img = tf.io.read_file(tf.constant(str(image_path)))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (img_size, img_size))
    img = tf.cast(img, tf.float32)
    if normalize_01:
        img = img / 255.0
    return tf.expand_dims(img, 0)


# ──────────────────────────────────────────────────────────────────────
#  PREDICT
# ──────────────────────────────────────────────────────────────────────

def predict_one(model, class_names, image_path, normalize_01, top_k):
    x = preprocess(image_path, IMG_SIZE, normalize_01)
    probs = model.predict(x, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    top = [{"class": class_names[i] if i < len(class_names) else f"class_{i}",
            "prob": float(probs[i])} for i in top_idx]

    print(f"\nImage: {image_path}\n" + "-" * 44)
    for rank, t in enumerate(top, 1):
        print(f"  {rank}. {t['class']:<25s} {t['prob'] * 100:6.2f}%")

    return {
        "image": str(image_path),
        "pred_class": top[0]["class"],
        "pred_prob": top[0]["prob"],
        "top_k": top,
    }


def gather_images(args):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    if args.image:
        return [Path(args.image)]
    if args.dir:
        return sorted(p for p in Path(args.dir).rglob("*") if p.suffix.lower() in exts)
    if args.glob:
        return sorted(Path(".").glob(args.glob))
    raise SystemExit("Provide one of --image, --dir, or --glob")


# ──────────────────────────────────────────────────────────────────────
#  SAVE RESULTS
# ──────────────────────────────────────────────────────────────────────

def save_results(results, out_dir):
    """Write CSV (flat), JSON (full top-k), and a human-readable summary."""
    run_dir = Path(out_dir) / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    # CSV — one row per image
    csv_path = run_dir / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "pred_class", "pred_prob"])
        for r in results:
            w.writerow([r["image"], r["pred_class"], f"{r['pred_prob']:.6f}"])

    # JSON — full top-k probabilities
    json_path = run_dir / "predictions.json"
    json_path.write_text(json.dumps(results, indent=2))

    # Summary — counts per predicted class
    counts = Counter(r["pred_class"] for r in results)
    summary_path = run_dir / "summary.txt"
    lines = [
        f"Total images: {len(results)}",
        f"Distinct predicted classes: {len(counts)}",
        "",
        "Predictions per class (desc):",
    ]
    lines += [f"  {cls:<25s} {n}" for cls, n in counts.most_common()]
    summary_path.write_text("\n".join(lines) + "\n")

    print(f"\nSaved results → {run_dir}/")
    print(f"  • predictions.csv   ({len(results)} rows)")
    print(f"  • predictions.json  (full top-k)")
    print(f"  • summary.txt")
    return run_dir


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="MTARSI aircraft inference + logging")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--classes", default=CLASS_NAMES_PATH)
    ap.add_argument("--image", help="single image path")
    ap.add_argument("--dir", help="folder of images (recursive)")
    ap.add_argument("--glob", help="glob pattern, e.g. 'data/*.jpg'")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ap.add_argument("--out-dir", default=OUTPUT_DIR, help="where to save predictions")
    args = ap.parse_args()

    print(f"Loading model:   {args.model}")
    model = keras.models.load_model(args.model)
    class_names = load_class_names(args.classes)
    normalize_01 = not model_normalizes_internally(model)

    print(f"Model:           {model.name}")
    print(f"Classes:         {len(class_names)}")
    print(f"Input mode:      {'[0,1] (scratch)' if normalize_01 else 'raw 0–255 (transfer)'}")

    images = gather_images(args)
    if not images:
        raise SystemExit("No images found.")
    print(f"Found {len(images)} image(s).")

    results = [predict_one(model, class_names, img, normalize_01, args.top_k)
               for img in images]
    save_results(results, args.out_dir)

    print("\n✓ Done.")


if __name__ == "__main__":
    main()