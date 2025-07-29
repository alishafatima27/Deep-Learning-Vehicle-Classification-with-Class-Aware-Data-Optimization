import os
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

def load_image_paths(root_dir, split_name):
    records = []
    classes = sorted(os.listdir(root_dir))
    for cls in classes:
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fname in os.listdir(cls_path):
            fpath = os.path.join(cls_path, fname)
            if os.path.isfile(fpath):
                records.append({
                    "set": split_name,
                    "class": cls,
                    "path": fpath,
                    "filename": fname
                })
    return records

def compute_brightness(df):
    brightness = []
    for p in tqdm(df['path'], desc="Computing brightness"):
        try:
            img = Image.open(p).convert('L')
            brightness.append(np.array(img).mean())
        except:
            brightness.append(np.nan)
    df['brightness'] = brightness
    return df

def save_plot(fig, output_path, filename):
    os.makedirs(output_path, exist_ok=True)
    fig.savefig(os.path.join(output_path, filename), dpi=300, bbox_inches='tight')
    plt.close(fig)

def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    train_dir = config['train_dir']
    val_dir = config['val_dir']
    output_dir = config['output_dir']
    vis_dir = os.path.join(output_dir, 'visualizations', 'eda')

    # Load data
    train_records = load_image_paths(train_dir, "train")
    val_records = load_image_paths(val_dir, "val")
    df = pd.DataFrame(train_records + val_records)
    print(f"Total images: {len(df)} across {len(df['class'].unique())} classes")
    print("Available classes:", sorted(df['class'].unique()))

    # Class distribution
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.countplot(data=df, y='class', hue='set', palette='tab20', ax=ax)
    ax.set_title("Class Distribution by Train/Validation Split")
    ax.set_xlabel("Image Count")
    ax.set_ylabel("Class")
    save_plot(fig, vis_dir, 'class_distribution.png')

    class_split_summary = df.groupby(['class', 'set']).size().unstack(fill_value=0)
    class_split_summary['total'] = class_split_summary.sum(axis=1)
    print("\nClass counts per split and totals:")
    print(class_split_summary)
    imbalance_ratio = class_split_summary['total'].max() / class_split_summary['total'].min()
    print(f"\nClass imbalance ratio (max/min): {imbalance_ratio:.2f}")

    # Check corrupt images
    corrupted = []
    for p in tqdm(df['path'], desc="Verifying images"):
        try:
            with Image.open(p) as img:
                img.verify()
        except:
            corrupted.append(p)
    print(f"Corrupted images found: {len(corrupted)}")

    # Check duplicates
    df['hash'] = [hashlib.md5(open(p, 'rb').read()).hexdigest() for p in tqdm(df['path'], desc="Hashing files")]
    duplicate_df = df[df.duplicated('hash', keep=False)]
    print(f"Number of duplicate images: {duplicate_df.shape[0]}")

    # Image sizes and aspect ratios
    widths, heights = [], []
    for p in tqdm(df['path'], desc="Reading image dimensions"):
        try:
            with Image.open(p) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
        except:
            widths.append(np.nan)
            heights.append(np.nan)
    df['width'] = widths
    df['height'] = heights
    df['aspect_ratio'] = df['width'] / df['height']
    print("Image dimension statistics:")
    print(df[['width', 'height', 'aspect_ratio']].describe())

    # Resolution distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.histplot(df['width'].dropna(), bins=30, color='blue', label='Width', kde=True, ax=ax)
    sns.histplot(df['height'].dropna(), bins=30, color='orange', label='Height', kde=True, ax=ax)
    ax.set_title('Image Width and Height Distribution')
    ax.legend()
    save_plot(fig, vis_dir, 'resolution_distribution.png')

    # Brightness analysis
    df = compute_brightness(df)
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(df['brightness'].dropna(), bins=50, color='gray', ax=ax)
    ax.set_title('Brightness Distribution')
    ax.set_xlabel('Mean Pixel Intensity')
    save_plot(fig, vis_dir, 'brightness_distribution.png')

    # Save EDA results
    eda_results = {
        'timestamp': datetime.now().isoformat(),
        'total_images': int(len(df)),
        'corrupted_count': int(len(corrupted)),
        'duplicate_count': int(duplicate_df.shape[0]),
        'small_images_count': int(df[df['width'] < 180].shape[0] + df[df['height'] < 180].shape[0]),
        'class_distribution': {k: int(v) for k, v in df['class'].value_counts().to_dict().items()},
        'imbalance_ratio': float(imbalance_ratio)
    }
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    with open(os.path.join(output_dir, 'reports', 'eda_results.json'), 'w') as f:
        json.dump(eda_results, f, indent=2)
    print("✅ EDA results saved!")

if __name__ == "__main__":
    config_path = r"C:\Users\HP\Desktop\vehicle_classification\config.json"
    main(config_path)