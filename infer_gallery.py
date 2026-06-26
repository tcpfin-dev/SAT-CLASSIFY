#!/usr/bin/env python3
"""MTARSI — Predict one image per class and save ANNOTATED images + a combined grid.

Walks each class folder under DATA_ROOT, picks one image per class, predicts,
and writes the image with the prediction drawn on it to predictions/.

Annotation shows:
    • true class   (from the folder name)
    • predicted class + confidence
    • GREEN text if correct, RED if wrong
    • semi‑transparent black background for readability

Then combines all annotated images into a single grid (contact sheet)
and saves as gallery_grid.png.

Preprocessing auto-detected (transfer = raw 0–255, scratch = [0,1]).

Usage:
    python infer_gallery.py
    python infer_gallery.py --data-root /kaggle/input/.../MTARSI --first
    python infer_gallery.py --seed 42 --grid-cols 5
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageDraw, ImageFont

# ──────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────

MODEL_PATH = "output/final_model.keras"
CLASS_NAMES_PATH = "output/class_names.txt"
DATA_ROOT = Path(__file__).resolve().parent / "MTARSI"
OUTPUT_DIR = "predictions"
IMG_SIZE = 128
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ──────────────────────────────────────────────────────────────────────
#  LOAD / DETECT
# ──────────────────────────────────────────────────────────────────────

def load_class_names(path):
    names = Path(path).read_text().strip().splitlines()
    if not names:
        raise ValueError(f"No class names in {path}")
    return names


def model_normalizes_internally(model):
    name = model.name.lower()
    if "efficientnet" in name:
        return True
    if "resnetse" in name:
        return False
    print(f"⚠️  Unknown model '{model.name}'. Assuming raw 0–255 input.")
    return True


# ──────────────────────────────────────────────────────────────────────
#  PREPROCESS (matches training's decode_image)
# ──────────────────────────────────────────────────────────────────────

def preprocess(image_path, normalize_01):
    img = tf.io.read_file(tf.constant(str(image_path)))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32)
    if normalize_01:
        img = img / 255.0
    return tf.expand_dims(img, 0)


# ──────────────────────────────────────────────────────────────────────
#  PICK ONE IMAGE PER CLASS
# ──────────────────────────────────────────────────────────────────────

def pick_one_per_class(data_root, use_first, rng):
    root = Path(data_root)
    class_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    if not class_dirs:
        raise SystemExit(f"No class folders found in {data_root}")

    picks = []
    for d in class_dirs:
        imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not imgs:
            print(f"⚠️  No images in {d.name}, skipping.")
            continue
        chosen = imgs[0] if use_first else rng.choice(imgs)
        picks.append((d.name, chosen))   # (true_class, path)
    return picks


# ──────────────────────────────────────────────────────────────────────
#  ANNOTATE (improved style)
# ──────────────────────────────────────────────────────────────────────

def _font(size, bold=True):
    # Try to use a nice font; fallback to default if none found.
    candidates = [
        "/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf",
        "/usr/share/fonts/Adwaita/AdwaitaSans-Italic.ttf",
    ]
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/System/Library/Fonts/Helvetica.ttc",   # macOS
        ])
    candidates.extend([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ])
    for fp in candidates:
        if Path(fp).exists():
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def annotate(image_path, true_cls, pred_cls, conf, correct, out_path,
             display_w=400):
    """Draw a clean prediction overlay on a copy of the image."""
    img = Image.open(image_path).convert("RGB")
    # Upscale small images so text is readable
    if img.width < display_w:
        h = int(img.height * display_w / img.width)
        img = img.resize((display_w, h), Image.LANCZOS)
    w, h = img.size

    # Create a semi-transparent overlay at the bottom
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    rect_h = 60
    draw_overlay.rectangle([(0, h - rect_h), (w, h)], fill=(0, 0, 0, 180))

    # Choose colours
    colour = (40, 200, 60) if correct else (220, 50, 50)  # green / red
    mark = "✓" if correct else "✗"

    # Draw text on the overlay
    f_big = _font(20, bold=True)
    f_small = _font(15, bold=False)

    # First line: mark + pred + confidence
    line1 = f"{mark} {pred_cls}  ({conf * 100:.1f}%)"
    # Second line: true class
    line2 = f"true: {true_cls}"

    # Coordinates: left-aligned with some margin
    margin = 10
    y0 = h - rect_h + 6
    draw_overlay.text((margin, y0), line1, fill=colour, font=f_big)
    draw_overlay.text((margin, y0 + 28), line2, fill=(255, 255, 255), font=f_small)

    # Composite overlay onto original image
    img_rgba = img.convert("RGBA")
    combined = Image.alpha_composite(img_rgba, overlay)
    combined = combined.convert("RGB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(out_path)


# ──────────────────────────────────────────────────────────────────────
#  CREATE GRID (contact sheet)
# ──────────────────────────────────────────────────────────────────────

def create_grid(image_paths, cols, thumb_size=300, spacing=10, title=None):
    """
    Combine a list of images into a single grid.
    Each image is resized to thumb_size (square) with padding to keep aspect ratio.
    """
    if not image_paths:
        return None

    # Sort by true class (filename pattern: true__pred.png)
    image_paths = sorted(image_paths, key=lambda p: p.stem.split("__")[0])

    total = len(image_paths)
    rows = math.ceil(total / cols)

    # Prepare a blank canvas (RGBA)
    grid_w = cols * thumb_size + (cols + 1) * spacing
    grid_h = rows * thumb_size + (rows + 1) * spacing

    # Add space for title if provided
    title_h = 0
    if title:
        title_h = 60
        grid_h += title_h

    grid = Image.new("RGB", (grid_w, grid_h), color=(30, 30, 30))

    # Draw title
    if title:
        draw = ImageDraw.Draw(grid)
        font_title = _font(28, bold=True)
        # compute text size
        bbox = draw.textbbox((0, 0), title, font=font_title)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((grid_w - tw) // 2, 10), title, fill=(255, 255, 255), font=font_title)

    for idx, img_path in enumerate(image_paths):
        row = idx // cols
        col = idx % cols
        x = spacing + col * (thumb_size + spacing)
        y = spacing + row * (thumb_size + spacing) + title_h

        # Open and resize with padding (letterbox)
        img = Image.open(img_path).convert("RGB")
        # Resize to fit inside thumb_size while preserving aspect ratio
        img.thumbnail((thumb_size, thumb_size), Image.LANCZOS)
        # Paste centered onto a square background
        bg = Image.new("RGB", (thumb_size, thumb_size), color=(50, 50, 50))
        offset = ((thumb_size - img.width) // 2, (thumb_size - img.height) // 2)
        bg.paste(img, offset)
        grid.paste(bg, (x, y))

    return grid


# ──────────────────────────────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="One annotated prediction per class + grid")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--classes", default=CLASS_NAMES_PATH)
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--out-dir", default=OUTPUT_DIR)
    ap.add_argument("--first", action="store_true",
                    help="always pick the first image (default: random)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for random pick (reproducible)")
    ap.add_argument("--grid-cols", type=int, default=6,
                    help="number of columns in the gallery grid (default: 6)")
    ap.add_argument("--grid-size", type=int, default=300,
                    help="thumbnail size (square) for grid images (default: 300)")
    ap.add_argument("--no-grid", action="store_true",
                    help="skip generating the combined grid")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"Loading model: {args.model}")
    model = keras.models.load_model(args.model)
    class_names = load_class_names(args.classes)
    normalize_01 = not model_normalizes_internally(model)
    print(f"Model: {model.name} | classes: {len(class_names)} | "
          f"input: {'[0,1]' if normalize_01 else 'raw 0–255'}")

    picks = pick_one_per_class(args.data_root, args.first, rng)
    out_dir = Path(args.out_dir) / "gallery"
    print(f"Predicting on {len(picks)} images (one per class)...\n")

    generated_paths = []  # store for grid
    n_correct = 0
    for true_cls, img_path in picks:
        x = preprocess(img_path, normalize_01)
        probs = model.predict(x, verbose=0)[0]
        top = int(np.argmax(probs))
        pred_cls = class_names[top] if top < len(class_names) else f"class_{top}"
        conf = float(probs[top])
        correct = (pred_cls == true_cls)
        n_correct += correct

        out_path = out_dir / f"{true_cls}__pred_{pred_cls}.png"
        annotate(img_path, true_cls, pred_cls, conf, correct, out_path)
        generated_paths.append(out_path)

        mark = "✓" if correct else "✗"
        print(f"  {mark} {true_cls:<22s} → {pred_cls:<22s} {conf*100:5.1f}%")

    acc = n_correct / len(picks) if picks else 0.0
    print(f"\nGallery accuracy (1 img/class): {n_correct}/{len(picks)} = {acc*100:.1f}%")
    print(f"Annotated images → {out_dir}/")

    # ─── Build combined grid ──────────────────────────────────────────────
    if not args.no_grid and generated_paths:
        # Exclude any file with "output_pred" in its name (as requested)
        grid_paths = [p for p in generated_paths if "output_pred" not in p.name]
        if grid_paths:
            print(f"\nBuilding gallery grid with {len(grid_paths)} images...")
            grid_img = create_grid(
                grid_paths,
                cols=args.grid_cols,
                thumb_size=args.grid_size,
                spacing=10,
                title=f"MTARSI Predictions Gallery  (acc: {acc*100:.1f}%)"
            )
            if grid_img:
                grid_out = out_dir / "gallery_grid.png"
                grid_img.save(grid_out)
                print(f"Grid saved → {grid_out}")
        else:
            print("No images to include in grid (all filtered out?)")
    elif args.no_grid:
        print("Grid generation skipped (--no-grid).")


if __name__ == "__main__":
    main()