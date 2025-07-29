import json
from sklearn.model_selection import train_test_split
import os
import shutil
from pathlib import Path

def main(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    original_dir = Path(config['train_dir'])
    new_train_dir = Path(config['output_dir']) / 'new_train'
    new_val_dir = Path(config['output_dir']) / 'new_val'
    new_train_dir.mkdir(parents=True, exist_ok=True)
    new_val_dir.mkdir(parents=True, exist_ok=True)

    for cls in original_dir.iterdir():
        if not cls.is_dir():
            continue
        images = [f for f in cls.glob("*.*") if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        train_imgs, val_imgs = train_test_split(images, test_size=0.15, random_state=42)
        (new_train_dir / cls.name).mkdir(exist_ok=True)
        (new_val_dir / cls.name).mkdir(exist_ok=True)
        for img in train_imgs:
            shutil.copy2(img, new_train_dir / cls.name / img.name)
        for img in val_imgs:
            shutil.copy2(img, new_val_dir / cls.name / img.name)

    print("Dataset re-split complete. Update config.json with:")
    print(f"  \"train_dir\": \"{new_train_dir}\",")
    print(f"  \"val_dir\": \"{new_val_dir}\",")

if __name__ == "__main__":
    config_path = r"C:\Users\HP\Desktop\vehicle_classification\config.json"
    main(config_path)