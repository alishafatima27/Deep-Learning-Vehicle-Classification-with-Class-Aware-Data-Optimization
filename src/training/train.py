import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from augmentation import AugmentationStrategy

def train_model(model, model_name, train_loader, val_loader, class_weights, class_names, config, aug_strategy):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights = class_weights.to(device)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5)
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    best_val_acc = 0.0
    output_dir = Path(config['output_dir']) / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = Path(config['output_dir']) / 'visualizations' / 'training'
    viz_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / f"best_{model_name}.pth"
    accum_steps = 4 if model_name == 'vit_b_16' else 1
    for epoch in range(config['num_epochs']):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        optimizer.zero_grad()
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}")):
            images, labels = images.to(device), labels.to(device)
            if random.random() < 0.3:
                images, labels_a, labels_b, lam = aug_strategy.mixup(images, labels)
                outputs = model(images)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                images = aug_strategy.cutout(images)
                outputs = model(images)
                loss = criterion(outputs, labels)
            loss = loss / accum_steps
            loss.backward()
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            train_loss += loss.item() * images.size(0) * accum_steps
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)
        train_loss /= train_total
        train_acc = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        scheduler.step(val_acc)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        mAP = average_precision_score(np.eye(len(class_names))[all_labels], all_probs, average='weighted')
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, "
              f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, mAP: {mAP:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{model_name} Confusion Matrix')
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.savefig(viz_dir / f'{model_name}_confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title(f'{model_name} Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.title(f'{model_name} Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_dir / f'{model_name}_learning_curve_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.close()
    metrics_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_accuracy': train_accuracies,
        'val_accuracy': val_accuracies
    })
    metrics_df.to_csv(output_dir / f'{model_name}_performance.csv', index=False)
    return {
        'val_acc': val_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mAP': mAP,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_model_path': best_model_path
    }