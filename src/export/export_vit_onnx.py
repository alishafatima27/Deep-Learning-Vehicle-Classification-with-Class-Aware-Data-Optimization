import json
import torch
import torchvision.models as models
import onnx
from pathlib import Path
from verify import verify_onnx_model, save_verification_results
from preprocessing import load_verification_images

def initialize_vit_model(num_classes):
    model = models.vit_b_16(weights=None)
    num_ftrs = model.heads.head.in_features
    model.heads = torch.nn.Linear(num_ftrs, num_classes)
    return model

def export_to_onnx(model, model_name, output_dir, input_shape=(1, 3, 224, 224)):
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    output_path = Path(output_dir) / 'models' / f'{model_name}.onnx'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=14,  # Updated to support aten::scaled_dot_product_attention
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ Exported {model_name} to {output_path}")
    return output_path

def main():
    # Load config
    with open(r"C:\Users\HP\Desktop\vehicle_classification\config.json") as f:
        config = json.load(f)
    
    # Load class names
    classes_path = Path(config['output_dir']) / 'classes.txt'
    if not classes_path.exists():
        raise FileNotFoundError(f"Classes file not found at {classes_path}")
    with open(classes_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    # Initialize ViT-B/16 model
    model_name = 'vit_b_16'
    model = initialize_vit_model(len(class_names))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load weights
    model_path = Path(config['output_dir']) / 'models' / f'best_{model_name}.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    print(f"Loading weights for {model_name} from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    
    # Export to ONNX
    onnx_path = export_to_onnx(model, model_name, config['output_dir'])
    
    # Load verification images
    verification_images = load_verification_images(config['verification_dir'], config['image_size'])
    
    # Verify ONNX model
    predictions = verify_onnx_model(onnx_path, verification_images, class_names, device)
    save_verification_results(
        predictions,
        Path(config['output_dir']) / 'models' / f'verification_predictions_{model_name}.txt'
    )
    
    print("✅ ViT-B/16 ONNX export and verification complete!")

if __name__ == "__main__":
    main()