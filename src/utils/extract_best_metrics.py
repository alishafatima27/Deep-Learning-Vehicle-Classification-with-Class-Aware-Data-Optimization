import json
import pandas as pd
from pathlib import Path
import glob
import shutil
from datetime import datetime
import matplotlib.pyplot as plt

def find_best_epoch(csv_path):
    """Read performance CSV and return the epoch with highest val_accuracy."""
    df = pd.read_csv(csv_path)
    best_row = df.loc[df['val_accuracy'].idxmax()]
    return {
        'epoch': int(best_row['epoch']),
        'train_loss': best_row['train_loss'],
        'val_loss': best_row['val_loss'],
        'train_accuracy': best_row['train_accuracy'],
        'val_accuracy': best_row['val_accuracy']
    }

def find_best_graph(graph_dir, model_name, best_epoch):
    """Find the learning curve graph for the best epoch, skipping the first (blank) graph."""
    graph_files = sorted(glob.glob(str(graph_dir / f'{model_name}_learning_curve_*.png')))
    if len(graph_files) < best_epoch:  # Ensure enough files exist
        print(f"⚠️ Not enough learning curve graphs for {model_name} (need {best_epoch}, found {len(graph_files)})")
        return None
    # Skip the first graph (Epoch 1, likely blank) and select the best_epoch index (0-based)
    return graph_files[best_epoch - 1] if best_epoch <= len(graph_files) else None

def plot_learning_curve(df, model_name, best_epoch, output_dir):
    """Regenerate learning curve plot for all epochs up to the best epoch."""
    plt.figure(figsize=(12, 5))
    
    # Loss subplot
    plt.subplot(1, 2, 1)
    plt.plot(df['epoch'][:best_epoch], df['train_loss'][:best_epoch], label='Train Loss', color='#1f77b4')
    plt.plot(df['epoch'][:best_epoch], df['val_loss'][:best_epoch], label='Val Loss', color='#ff7f0e')
    plt.title(f'{model_name} Loss Curve (Best Epoch: {best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy subplot
    plt.subplot(1, 2, 2)
    plt.plot(df['epoch'][:best_epoch], df['train_accuracy'][:best_epoch], label='Train Accuracy', color='#1f77b4')
    plt.plot(df['epoch'][:best_epoch], df['val_accuracy'][:best_epoch], label='Val Accuracy', color='#ff7f0e')
    plt.title(f'{model_name} Accuracy Curve (Best Epoch: {best_epoch})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_path = output_dir / f'{model_name}_best_learning_curve_regenerated.png'
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Regenerated learning curve for {model_name} saved to {output_path}")
    return output_path

def main():
    # Load config
    config_path = r"C:\Users\HP\Desktop\vehicle_classification\config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    output_dir = Path(config['output_dir'])
    models_dir = output_dir / 'models'
    viz_dir = output_dir / 'visualizations' / 'training'
    aug_dir = output_dir / 'visualizations' / 'augmentation'
    best_metrics_dir = output_dir / 'best_metrics'
    best_metrics_dir.mkdir(parents=True, exist_ok=True)
    
    models = ['resnet50', 'efficientnet_b0', 'vit_b_16']
    
    for model_name in models:
        # Read performance CSV
        csv_path = models_dir / f'{model_name}_performance.csv'
        if not csv_path.exists():
            print(f"⚠️ Performance CSV not found for {model_name} at {csv_path}")
            continue
        
        # Find best epoch
        metrics = find_best_epoch(csv_path)
        best_epoch = metrics['epoch']
        print(f"Best epoch for {model_name}: Epoch {best_epoch} (Val Accuracy: {metrics['val_accuracy']:.4f})")
        
        # Try to find existing non-blank learning curve graph
        graph_path = find_best_graph(viz_dir, model_name, best_epoch)
        if graph_path:
            dest_path = best_metrics_dir / f'{model_name}_best_learning_curve.png'
            shutil.copy(graph_path, dest_path)
            print(f"✅ Copied learning curve for {model_name} to {dest_path}")
        else:
            # Regenerate if no valid graph found
            print(f"⚠️ No valid learning curve found for {model_name}, regenerating...")
            df = pd.read_csv(csv_path)
            plot_learning_curve(df, model_name, best_epoch, best_metrics_dir)
    
    # List augmentation visualizations
    aug_images = list(aug_dir.glob('*.jpg'))
    print("\nAugmentation Visualization Images:")
    for img in aug_images:
        print(f"- {img}")
    
    # Note about metrics
    print("\nNote: Check or update 'output/best_metrics/best_metrics.json' for metrics.")
    print("Add Precision, Recall, F1, and mAP from console logs (e.g., ViT-B/16 Epoch 15: Precision=0.8911, Recall=0.8913, F1=0.8864, mAP=0.9510).")
    print(f"Confusion matrices are in {best_metrics_dir} (e.g., {model_name}_best_confusion_matrix.png).")

if __name__ == "__main__":
    main()