import json
import torch
import numpy as np
from torchvision import transforms as T
from collections import Counter
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

class AugmentationStrategy:
    def __init__(self, config, dataset):
        self.dataset = dataset
        self.target_min_samples = config['target_min_samples']
        self.image_size = config['image_size']
        self.class_counts = Counter([s['class'] for s in dataset.samples])
        self.augmentation_configs = self._create_augmentation_configs()
        self.mixup_alpha = 0.2
        self.cutout_size = self.image_size // 4

    def _create_augmentation_configs(self):
        configs = {}
        for class_name, count in self.class_counts.items():
            intensity = 'extreme' if count < 10 else 'high' if count < 50 else 'medium' if count < 100 else 'low'
            configs[class_name] = {
                'intensity': intensity,
                'current_samples': count,
                'target_samples': max(count, self.target_min_samples),
                'augmentation_multiplier': max(1, self.target_min_samples // count) if count > 0 else 1
            }
        return configs

    def get_transform_for_class(self, class_name):
        config = self.augmentation_configs.get(class_name, {'intensity': 'low'})
        intensity = config['intensity']
        base_transform = [T.Resize((256, 256))]
        if intensity == 'extreme':
            return T.Compose(base_transform + [
                T.RandomCrop((self.image_size, self.image_size), padding=32),
                T.RandomHorizontalFlip(p=0.8),
                T.RandomVerticalFlip(p=0.3),
                T.RandomRotation(degrees=30),
                T.RandomAffine(degrees=25, translate=(0.15, 0.15), scale=(0.8, 1.2), shear=15),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                T.RandomGrayscale(p=0.2),
                T.RandomPerspective(distortion_scale=0.3, p=0.5),
                T.ToTensor(),
                T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                T.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif intensity == 'high':
            return T.Compose(base_transform + [
                T.RandomCrop((self.image_size, self.image_size), padding=24),
                T.RandomHorizontalFlip(p=0.7),
                T.RandomRotation(degrees=20),
                T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
                T.RandomGrayscale(p=0.1),
                T.RandomPerspective(distortion_scale=0.2, p=0.3),
                T.ToTensor(),
                T.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        elif intensity == 'medium':
            return T.Compose(base_transform + [
                T.RandomCrop((self.image_size, self.image_size), padding=16),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=15),
                T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                T.ToTensor(),
                T.RandomErasing(p=0.1, scale=(0.02, 0.08), ratio=(0.3, 3.3)),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            return T.Compose([
                T.Resize((self.image_size, self.image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def apply_synthetic_augmentation(self, dataset):
        existing_synthetic = [s for s in dataset.samples if s.get('is_synthetic', False)]
        if existing_synthetic:
            print("Synthetic samples already exist, skipping augmentation")
            return dataset
        original_samples = [s for s in dataset.samples]
        total_synthetic_generated = 0
        for class_name, config in self.augmentation_configs.items():
            if config['augmentation_multiplier'] > 1:
                current_samples = [s for s in original_samples if s['class'] == class_name]
                target_samples = config['target_samples']
                needed_samples = target_samples - len(current_samples)
                if needed_samples > 0 and current_samples:
                    synthetic_count = 0
                    while synthetic_count < needed_samples:
                        source_sample = random.choice(current_samples)
                        synthetic_sample = source_sample.copy()
                        synthetic_sample['is_synthetic'] = True
                        synthetic_sample['synthetic_id'] = synthetic_count
                        dataset.samples.append(synthetic_sample)
                        synthetic_count += 1
                    total_synthetic_generated += synthetic_count
        print(f"✅ Synthetic augmentation complete: {total_synthetic_generated} samples generated")
        return dataset

    def mixup(self, images, labels):
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        alpha = self.mixup_alpha
        lam = np.random.beta(alpha, alpha)
        mixed_images = lam * images + (1 - lam) * images[indices]
        return mixed_images, labels, labels[indices], lam

    def cutout(self, images):
        batch_size = images.size(0)
        c, h, w = images.size()[1:]
        for i in range(batch_size):
            if random.random() < 0.5:
                x = random.randint(0, w - self.cutout_size)
                y = random.randint(0, h - self.cutout_size)
                images[i, :, y:y+self.cutout_size, x:x+self.cutout_size] = 0
        return images

    def visualize_augmentation(self, dataset, output_dir):
        output_dir = Path(output_dir) / 'visualizations' / 'augmentation'
        output_dir.mkdir(parents=True, exist_ok=True)
        for class_name in random.sample(dataset.classes, min(len(dataset.classes), 3)):
            class_samples = [s for s in dataset.samples if s['class'] == class_name][:5]
            transform = self.get_transform_for_class(class_name)
            plt.figure(figsize=(15, 5))
            for i, sample in enumerate(class_samples):
                try:
                    img = Image.open(sample['path']).convert('RGB')
                    aug_img = transform(img)
                    aug_img = aug_img.permute(1, 2, 0).numpy() * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                    aug_img = np.clip(aug_img, 0, 1)
                    plt.subplot(1, 5, i+1)
                    plt.imshow(aug_img)
                    plt.title(f"{class_name} Sample {i+1}")
                    plt.axis('off')
                except Exception as e:
                    print(f"Error processing image {sample['path']}: {e}")
                    continue
            plt.tight_layout()
            plt.savefig(output_dir / f"{class_name}_augmented.jpg", dpi=300)
            plt.close()

    def save_strategy(self, filepath):
        strategy_data = {
            'class_counts': {k: int(v) for k, v in self.class_counts.items()},
            'target_min_samples': self.target_min_samples,
            'image_size': self.image_size,
            'augmentation_configs': self.augmentation_configs
        }
        with open(filepath, 'w') as f:
            json.dump(strategy_data, f, indent=2)

def main():
    with open(r"C:\Users\HP\Desktop\vehicle_classification\config.json") as f:
        config = json.load(f)
    from dataset import ClassAwareDataset
    train_dataset = ClassAwareDataset(Path(config['train_dir']), config=config, is_training=True)
    aug_strategy = AugmentationStrategy(config, train_dataset)
    train_dataset = aug_strategy.apply_synthetic_augmentation(train_dataset)
    aug_strategy.save_strategy(Path(config['output_dir']) / 'logs' / 'augmentation_stats.json')
    aug_strategy.visualize_augmentation(train_dataset, config['output_dir'])
    print("✅ Augmentation stats and visualizations saved!")

if __name__ == "__main__":
    main()