import torch
import torchvision.models as models
import onnx
from pathlib import Path

def initialize_models(num_classes):
    models_dict = {}
    model = models.resnet50(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    models_dict['resnet50'] = model
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    num_ftrs = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(num_ftrs, num_classes)
    )
    models_dict['efficientnet_b0'] = model
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    num_ftrs = model.heads.head.in_features
    model.heads = torch.nn.Linear(num_ftrs, num_classes)
    models_dict['vit_b_16'] = model
    return models_dict

def export_to_onnx(model, model_name, output_dir, input_shape=(1, 3, 224, 224)):
    model.eval()
    dummy_input = torch.randn(input_shape).to(next(model.parameters()).device)
    output_path = Path(output_dir) / 'models' / f'{model_name}.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ Exported {model_name} to {output_path}")