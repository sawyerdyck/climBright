import os
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, convnext_tiny
from PIL import Image
import timm

# Configuration matching train.py
NUM_CLASSES = 6
CLASS_ID_TO_NAME = {
    0: "Jug",
    1: "Crimp",
    2: "Pinch",
    3: "Sloper",
    4: "Pocket",
    5: "Volume",
}

# Normalization: ImageNet stats (matching train.py)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Inference transform (no augmentation, just normalization)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


def detect_model_type(state_dict):
    """Detect whether checkpoint is ResNet18 or ConvNeXt."""
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith('stem.') or first_key.startswith('stages.'):
        return 'convnext'
    elif first_key.startswith('conv1.') or first_key.startswith('layer'):
        return 'resnet18'
    else:
        return 'unknown'


def load_model(checkpoint_path, device):
    """Load the trained model from checkpoint (auto-detects architecture)."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Auto-detect model type
    model_type = detect_model_type(state_dict)
    
    if model_type == 'resnet18':
        print(f"✓ Detected ResNet18 architecture")
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    elif model_type == 'convnext':
        print(f"✓ Detected ConvNeXt architecture")
        # Use timm for ConvNeXt to ensure compatibility
        import timm
        model = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=False, num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model architecture. First key: {next(iter(state_dict.keys()))}")
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"✓ Loaded model from: {checkpoint_path}")
    return model


def predict_image(model, image_path, device):
    """Predict the climbing hold class for a single image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    print(f"✓ Loaded image: {image_path}")
    print(f"  Image size: {img.size[0]}x{img.size[1]}")
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        predicted_class_id = logits.argmax(dim=1).item()
        confidence = probabilities[predicted_class_id].item()
    
    return predicted_class_id, confidence, probabilities


def main():
    parser = argparse.ArgumentParser(
        description="Test climbing hold classifier on a single image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py --model checkpoints/resnet18_best.pt --image test_image.jpg
  python predict.py -m model.pt -i image.jpg
        """
    )
    parser.add_argument("-m", "--model", type=str, required=True,
                        help="Path to the model checkpoint (.pt file)")
    parser.add_argument("-i", "--image", type=str, required=True,
                        help="Path to the image file to classify")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu). Default: auto-detect")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    print()
    
    try:
        # Load model
        model = load_model(args.model, device)
        print()
        
        # Run prediction
        predicted_id, confidence, probabilities = predict_image(model, args.image, device)
        predicted_class = CLASS_ID_TO_NAME[predicted_id]
        
        # Display results
        print()
        print("=" * 60)
        print("PREDICTION RESULTS")
        print("=" * 60)
        print(f"Predicted Class: {predicted_class}")
        print(f"Confidence:      {confidence:.2%}")
        print()
        print("All Class Probabilities:")
        print("-" * 60)
        
        # Sort by probability (highest first)
        sorted_probs = sorted(
            [(CLASS_ID_TO_NAME[i], probabilities[i].item()) for i in range(NUM_CLASSES)],
            key=lambda x: x[1],
            reverse=True
        )
        
        for class_name, prob in sorted_probs:
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            marker = " ← PREDICTED" if class_name == predicted_class else ""
            print(f"  {class_name:10s} {bar} {prob:6.2%}{marker}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
