# Vehicle Classification Pipeline

## Overview
This repository implements a state-of-the-art pipeline for multi-class vehicle classification across 0 categories using deep learning models: ResNet50, EfficientNet-B0, and Vision Transformer B/16 (ViT-B/16). The pipeline integrates **Exploratory Data Analysis (EDA)**, **CleanVision-based data cleaning**, **class-aware data augmentation**, **feature extraction**, **transfer learning**, **model optimization**, and **ONNX serialization**. The dataset comprises training images 3124 (722 synthetic) and  validation images, processed on an NVIDIA GTX 1650 using PyTorch 2.4.0+cu124 in a Windows Conda environment. Performance is evaluated with **macro-averaged precision**, **recall**, **F1 score**, and **mean Average Precision (mAP)** for robust generalization.

## Methodology

### 1. Data Import
- **Purpose**: Import raw data from source (e.g., filesystem, database).
- **Implementation**: `src/data/import.py` loads data into the project directory.

### 2. Exploratory Data Analysis (EDA)
- **Purpose**: Analyze class distribution to identify imbalances and guide augmentation.
- **Implementation**: `src/data/eda.py` generates statistics, saved in `output/logs/dataset_stats.json`.
- **Outcome**: Visualized class frequencies to mitigate **data skew**.

### 3. Data Cleaning with CleanVision
- **Tool**: CleanVision (`src/data/cleaning.py`) resolves:
  - **Near-duplicates**: Removes redundant images to prevent overfitting.
  - **Blurry images**: Excludes low-resolution or out-of-focus samples.
  - **Corrupt images**: Discards unreadable files.
  - **Low-contrast images**: Eliminates poor-contrast samples.
- **Output**: Cleaned dataset in `output/new_train` and `output/new_val`.
- **Reports**: Logs detail images removed (check `src/data/cleaning.py` execution).

### 4. Dataset Splitting
- **Purpose**: Split data into training and validation sets.
- **Implementation**: `src/data/resplit_dataset.py` ensures balanced class splits.
- **Outcome**: Created `output/new_train` and `output/new_val`.

### 5. Class-Aware Data Augmentation
- **Purpose**: Generate synthetic samples to ensure ≥200 samples per class.
- **Transformations** (`src/data/augmentation.py`):
  - **RandomCrop**: Crops to 224x224 for spatial robustness.
  - **RandomHorizontalFlip**: Flips images (p=0.5) for mirror invariance.
  - **ColorJitter**: Adjusts brightness, contrast, saturation, hue for color robustness.
  - **RandomPerspective**: Applies distortion for viewpoint variation.
  - **RandomRotation**: Rotates ±15° for orientation invariance.
  - **GaussianBlur**: Applies blur for noise resilience.
  - **RandomAffine**: Combines translation, scaling, shearing for geometric diversity.
- **Output**: Visualizations in `output/visualizations/augmentation/<class_name>_augmented.jpg`, validating **domain adaptation**.

### 6. Data Preprocessing
- **Normalization**: ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), resized to 224x224 (`src/data/preprocessing.py`, `src/data/dataset.py`).
- **Sampling**: **Weighted random sampling** (`src/data/dataset.py`) balances classes.

### 7. Feature Extraction and Transfer Learning
- **Models**: Pre-trained ResNet50, EfficientNet-B0, ViT-B/16 (`src/models/model_utils.py`).
- **Approach**: Fine-tuned with 0-class heads, freezing early layers initially.

### 8. Model Training and Optimization
- **Parameters**: 15 epochs, learning rate 0.0001, weight decay 0.01, batch size 16, Adam optimizer, **cross-entropy loss** (`src/training/train.py`).
- **Evaluation**: **Macro-averaged precision**, **recall**, **F1 score**, **mAP**.
- **Output**: Best weights in `output/models/best_<model_name>.pth`.

### 9. Model Serialization
- **ONNX Export**:
  - ResNet50, EfficientNet-B0: Opset 13 (`src/export/export_onnx.py`).
  - ViT-B/16: Opset 14 for `aten::scaled_dot_product_attention` (`src/export/export_vit_onnx.py`).
- **Verification**: Validated on `verification_images/` (`src/export/verify.py`).

### 10. Performance Visualization
- **Confusion Matrices**: Per-class performance in `output/best_metrics/<model_name>_best_confusion_matrix.png`.
- **Learning Curves**: Loss and accuracy in `output/best_metrics/<model_name>_best_learning_curve.png`.
- **Augmentation Visuals**: Transformation effects in `output/visualizations/augmentation/`.

## Setup
- **Hardware**: NVIDIA GTX 1650
- **Software**: Python 3.8+, PyTorch 2.4.0+cu124, Windows, Conda (`myenv`)
- **Dependencies**:
  ```bash
  conda create -n myenv python=3.8
  conda activate myenv
  pip install torch==2.4.0 torchvision==0.19.0 onnx pandas matplotlib numpy pillow cleanvision
  ```

## Project Structure
- `src/data/`
  - `import.py`: Imports raw data.
  - `cleaning.py`: CleanVision data cleaning.
  - `eda.py`: Generates dataset statistics.
  - `resplit_dataset.py`: Splits data into training/validation.
  - `dataset.py`: Loads data, applies sampling.
  - `augmentation.py`: Applies transformations.
  - `preprocessing.py`: Normalizes images.
- `src/training/`
  - `train.py`: Trains models, generates visualizations.
  - `vehicle_classification_pipeline.py`: Orchestrates pipeline.
- `src/models/`
  - `model_utils.py`: Defines architectures, ONNX utilities.
- `src/export/`
  - `export_onnx.py`: Exports ResNet50, EfficientNet-B0.
  - `export_vit_onnx.py`: Exports ViT-B/16.
  - `verify.py`: Validates ONNX models.
- `src/utils/`
  - `extract_best_metrics.py`: Extracts best epoch metrics.
- `config/`
  - `config.json`: Paths and hyperparameters.
- `output/`: Models, metrics, visualizations, logs.
- `verification_images/`: Verification dataset.
- `README.md`: Project documentation.


## Execution
1. **Activate Environment**:
   ```bash
   conda activate myenv
   cd C:/Users/HP/Desktop/vehicle_classification
   ```

2. **Run Pipeline**:
   ```bash
   python src/training/vehicle_classification_pipeline.py
   ```

3. **Export ONNX Models**:
   - ResNet50, EfficientNet-B0:
     ```bash
     python src/export/export_onnx.py
     ```
   - ViT-B/16:
     ```bash
     python src/export/export_vit_onnx.py
     ```

## Results
### Dataset Statistics
- **Training Images**: 
- **Validation Images**: 
- **Synthetic Images**: 722
- **Classes**: 12
- **Class Distribution** (from `output/logs/dataset_stats.json`):
  - Not available (run `src/data/eda.py`)

### Performance Metrics
Best epoch metrics for **generalization performance**:
#### RESNET50
- **Best Epoch**: 13
- **Training Loss**: 0.2651
- **Validation Loss**: 0.4337
- **Training Accuracy**: 0.846
- **Validation Accuracy**: 0.8744
- **Additional Metrics**: Update `output/best_metrics/best_metrics.json` with Precision, Recall, F1, mAP from `src/training/vehicle_classification_pipeline.py` logs.

#### EFFICIENTNET-B0
- **Best Epoch**: 15
- **Training Loss**: 0.2077
- **Validation Loss**: 0.3994
- **Training Accuracy**: 0.8516
- **Validation Accuracy**: 0.8744
- **Additional Metrics**: Update `output/best_metrics/best_metrics.json` with Precision, Recall, F1, mAP from `src/training/vehicle_classification_pipeline.py` logs.

#### VIT-B-16
- **Best Epoch**: 7
- **Training Loss**: 0.146
- **Validation Loss**: 0.3331
- **Training Accuracy**: 0.885
- **Validation Accuracy**: 0.9058
- **Additional Metrics**: Update `output/best_metrics/best_metrics.json` with Precision, Recall, F1, mAP from `src/training/vehicle_classification_pipeline.py` logs.

**Update Metrics**: Add ResNet50/EfficientNet-B0 metrics to `output/best_metrics/best_metrics.json` using logs. Example:
```json
{
  "resnet50": {
    "epoch": <epoch>,
    "train_loss": <value>,
    "val_loss": <value>,
    "train_accuracy": <value>,
    "val_accuracy": <value>,
    "precision": <value>,
    "recall": <value>,
    "f1": <value>,
    "map": <value>
  }
}
```

### Visualizations
- **Confusion Matrices**: `output/best_metrics/<model_name>_best_confusion_matrix.png`
- **Learning Curves**: `output/best_metrics/<model_name>_best_learning_curve.png`
- **Augmentation Visuals**: `output/visualizations/augmentation/<class_name>_augmented.jpg`

### Cleaning Reports
- CleanVision logs from `src/data/cleaning.py` detail duplicates, blurry, corrupt, and low-contrast images removed. Check execution logs for specifics.

## Outputs
- **Model Weights**: `output/models/best_<model_name>.pth`
- **Metrics**: `output/models/<model_name>_performance.csv`
- **Visualizations**: `output/best_metrics/` and `output/visualizations/`
- **ONNX Models**: `output/models/<model_name>.onnx`
- **Verification Results**: `output/models/verification_predictions_<model_name>.txt`
- **Dataset Statistics**: `output/logs/dataset_stats.json`

## Notes
- **Data Quality**: CleanVision ensures robust training data.
- **Class Imbalance**: Mitigated with synthetic samples and weighted sampling.
- **ONNX Export**: ViT-B/16 uses `opset_version=14`; others use opset 13.
- **Windows Compatibility**: `num_workers={config.get('num_workers', 'unknown')}` for stable data loading.
- **Traceability**: Logs in `output/logs/` provide transparency.

## Future Work
- **Model Ensembling**: Combine predictions via **stacked generalization**.
- **Hyperparameter Optimization**: Grid search or Bayesian methods.
- **Advanced Augmentation**: **Mixup**, **cutmix**, **auto-augmentation**.
- **Data Drift Mitigation**: Monitor distribution shifts.
- **Adversarial Robustness**: Test against adversarial examples.
- **Real-Time Inference**: Optimize ONNX models with quantization.

## Troubleshooting
- **Missing Metrics**: Check `output/models/<model_name>_performance.csv` and update `output/best_metrics/best_metrics.json`.
- **ONNX Issues**: Verify opset versions and `output/models/verification_predictions_<model_name>.txt`.
- **Data Issues**: Run `src/data/eda.py` for `output/logs/dataset_stats.json`.

This pipeline delivers a scalable, high-performance vehicle classification solution with robust preprocessing and deployment capabilities.
