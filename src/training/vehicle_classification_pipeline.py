import json
import torch
from pathlib import Path
from dataset import VehicleDatasetManager
from augmentation import AugmentationStrategy
from src.models.model_utils import initialize_models, export_to_onnx
from train import train_model
from verify import verify_onnx_model, save_verification_results
from preprocessing import load_verification_images
from torchvision import transforms as T

def main():
    with open(r"C:\Users\HP\Desktop\vehicle_classification\config.json") as f:
        config = json.load(f)
    manager = VehicleDatasetManager(config)
    train_dataset, val_dataset = manager.create_datasets()
    aug_strategy = AugmentationStrategy(config, train_dataset)
    train_dataset = aug_strategy.apply_synthetic_augmentation(train_dataset)
    train_loader, val_loader, train_dataset, val_dataset = manager.create_data_loaders()
    class_weights = train_dataset.get_class_weights()
    stats = train_dataset.get_dataset_info()
    with open(Path(config['output_dir']) / 'logs' / 'dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("Dataset Stats:", json.dumps(stats, indent=2))
    with open(Path(config['output_dir']) / 'classes.txt', 'w') as f:
        f.write('\n'.join(train_dataset.classes))
    models_dict = initialize_models(len(train_dataset.classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    verification_images = load_verification_images(config['verification_dir'], config['image_size'])
    for model_name, model in models_dict.items():
        print(f"Training {model_name}...")
        result = train_model(
            model, model_name, train_loader, val_loader, class_weights,
            train_dataset.classes, config, aug_strategy
        )
        model.load_state_dict(torch.load(result['best_model_path']))
        export_to_onnx(model, model_name, config['output_dir'])
        predictions = verify_onnx_model(
            Path(config['output_dir']) / 'models' / f'{model_name}.onnx',
            verification_images, train_dataset.classes, device
        )
        save_verification_results(
            predictions,
            Path(config['output_dir']) / 'models' / f'verification_predictions_{model_name}.txt'
        )
    aug_strategy.save_strategy(Path(config['output_dir']) / 'logs' / 'augmentation_stats.json')
    aug_strategy.visualize_augmentation(train_dataset, config['output_dir'])
    print("✅ Pipeline complete!")

if __name__ == "__main__":
    main()