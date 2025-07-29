import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
from preprocessing import preprocess_image
import torch

def verify_onnx_model(onnx_path, verification_images, class_names, device='cpu'):
    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    predictions = []
    for image, image_path in verification_images:
        image = image.to(device)
        ort_inputs = {input_name: image.cpu().numpy()}
        ort_outputs = session.run(None, ort_inputs)[0]
        probs = torch.softmax(torch.tensor(ort_outputs), dim=1).numpy()
        pred_class_idx = np.argmax(probs, axis=1)[0]
        pred_class = class_names[pred_class_idx]
        predictions.append((image_path.name, pred_class, probs[0][pred_class_idx]))
    return predictions

def save_verification_results(predictions, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for image_name, pred_class, confidence in predictions:
            f.write(f"{image_name}: {pred_class} ({confidence:.4f})\n")
    print(f"✅ Verification results saved to {output_path}")