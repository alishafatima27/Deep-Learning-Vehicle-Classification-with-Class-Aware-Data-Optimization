from PIL import Image
from torchvision import transforms as T
from pathlib import Path
import torch

def preprocess_image(image_path, image_size=224):
    try:
        image = Image.open(image_path).convert('RGB')
        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image).unsqueeze(0)
        return image
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def load_verification_images(verification_dir, image_size=224):
    verification_dir = Path(verification_dir)
    images = []
    image_paths = [p for p in verification_dir.glob("*.*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']]
    for image_path in image_paths:
        image = preprocess_image(image_path, image_size)
        if image is not None:
            images.append((image, image_path))
    return images