import json
import os
import torchvision.transforms as T
from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from PIL import Image
from augmentation import AugmentationStrategy

class ClassAwareDataset(Dataset):
    def __init__(self, root_dir, config=None, is_training=True):
        self.root_dir = Path(root_dir)
        self.config = config
        self.is_training = is_training
        self.samples = []
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self._build_dataset()
        self.augmentation_strategy = AugmentationStrategy(config, self) if config and is_training else None

    def _build_dataset(self):
        print(f"Scanning directory: {self.root_dir}")
        if not self.root_dir.exists():
            print(f"Error: Directory {self.root_dir} does not exist")
            return
        for class_dir in self.root_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                class_idx = self.class_to_idx[class_name]
                images = [f for f in class_dir.glob("*.*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
                print(f"Found {len(images)} images in class {class_name}")
                for img_path in images:
                    self.samples.append({
                        'path': str(img_path),
                        'class': class_name,
                        'class_idx': class_idx,
                        'is_synthetic': False
                    })
        print(f"Total samples found: {len(self.samples)}")
        if not self.samples:
            print(f"Warning: No valid images found in {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            image = Image.open(sample['path']).convert('RGB')
        except Exception as e:
            print(f"Error loading image {sample['path']}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        if self.augmentation_strategy and self.is_training:
            transform = self.augmentation_strategy.get_transform_for_class(sample['class'])
            image = transform(image)
        else:
            transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = transform(image)
        return image, sample['class_idx']

    def get_class_weights(self):
        class_counts = Counter([s['class_idx'] for s in self.samples])
        total_samples = len(self.samples)
        if total_samples == 0:
            print("Error: No samples in dataset, cannot compute class weights")
            return torch.FloatTensor([])
        weights = [total_samples / (len(self.classes) * class_counts.get(i, 1)) for i in range(len(self.classes))]
        return torch.FloatTensor(weights)

    def get_dataset_info(self):
        class_counts = Counter([s['class'] for s in self.samples])
        return {
            'total_images': len(self.samples),
            'num_classes': len(self.classes),
            'class_breakdown': {class_name: int(class_counts[class_name]) for class_name in self.classes}
        }

class VehicleDatasetManager:
    def __init__(self, config):
        self.config = config
        self.train_dir = Path(config['train_dir'])
        self.val_dir = Path(config['val_dir'])
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._train_dataset = None
        self._val_dataset = None

    def create_datasets(self):
        if self._train_dataset is None:
            self._train_dataset = ClassAwareDataset(self.train_dir, config=self.config, is_training=True)
        if self._val_dataset is None:
            self._val_dataset = ClassAwareDataset(self.val_dir, config=self.config, is_training=False)
        return self._train_dataset, self._val_dataset

    def create_data_loaders(self):
        train_dataset, val_dataset = self.create_datasets()
        print(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
        if len(train_dataset) == 0:
            raise ValueError("Training dataset is empty, cannot create data loader")
        class_indices = Counter([s['class_idx'] for s in train_dataset.samples])
        weights = [1.0 / class_indices[s['class_idx']] for s in train_dataset.samples]
        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            sampler=sampler,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        return train_loader, val_loader, train_dataset, val_dataset

    def save_dataset_info(self):
        train_dataset, _ = self.create_datasets()
        stats = train_dataset.get_dataset_info()
        (self.output_dir / 'logs').mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / 'logs' / 'dataset_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        return stats

def main():
    with open(r"C:\Users\HP\Desktop\vehicle_classification\config.json") as f:
        config = json.load(f)
    manager = VehicleDatasetManager(config)
    stats = manager.save_dataset_info()
    print("✅ Dataset stats saved!")
    print("Dataset Stats:", json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()