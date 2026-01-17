"""
Two-stage inference: YOLO detection → ConvNeXt classification
Uses YOLO to find holds, then classifies each detected box with ConvNeXt.
"""
import os
import json
import argparse
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import timm
from ultralytics import YOLO

# =========================
# CONFIGURATION
# =========================
YOLO_MODEL = "runs/detect/train2/weights/best.pt"
CONVNEXT_MODEL = "best_convnext_two_phase.pt"
NUM_CLASSES = 6
CLASS_NAMES = ["Jug", "Crimp", "Pinch", "Pocket", "Sloper", "Volume"]
YOLO_CONF_THRESHOLD = 0.25  # Lower threshold since we're re-classifying
BOX_PADDING = 0.15  # 15% padding around detected boxes
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ConvNeXt preprocessing (matches training)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

classify_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])


def load_classifier(checkpoint_path, device):
    """Load the ConvNeXt classifier."""
    print(f"Loading classifier from {checkpoint_path}...")
    model = timm.create_model(
        "convnext_tiny.in12k_ft_in1k",
        pretrained=False,
        num_classes=NUM_CLASSES
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✓ Classifier loaded")
    return model


def load_detector(model_path):
    """Load the YOLO detector."""
    print(f"Loading YOLO detector from {model_path}...")
    detector = YOLO(model_path)
    print("✓ Detector loaded")
    return detector


def pad_box(x1, y1, x2, y2, img_w, img_h, padding=0.15):
    """Add padding around a bounding box (percentage of box size)."""
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = box_w * padding
    pad_y = box_h * padding
    
    x1_new = max(0, int(x1 - pad_x))
    y1_new = max(0, int(y1 - pad_y))
    x2_new = min(img_w, int(x2 + pad_x))
    y2_new = min(img_h, int(y2 + pad_y))
    
    return x1_new, y1_new, x2_new, y2_new


def classify_crop(classifier, crop_pil, device):
    """Run classifier on a cropped region."""
    img_tensor = classify_transform(crop_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = classifier(img_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        class_id = logits.argmax(dim=1).item()
        confidence = probs[class_id].item()
    return class_id, confidence, probs


def detect_and_classify(detector, classifier, image_path, device, save_output=True):
    """
    Run YOLO detection, then classify each detected box with ConvNeXt.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {image_path}")
    print(f"{'='*60}")
    
    # Read image
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_bgr.shape[:2]
    print(f"Image size: {img_w}x{img_h}")
    
    # Run YOLO detection
    print("\n[1] Running YOLO detection...")
    results = detector.predict(
        source=img_rgb,
        conf=YOLO_CONF_THRESHOLD,
        verbose=False,
        device=DEVICE
    )
    
    detections = results[0].boxes
    num_detections = len(detections)
    print(f"✓ Found {num_detections} detections")
    
    if num_detections == 0:
        print("⚠ No holds detected!")
        return []
    
    # Classify each detection
    print(f"\n[2] Classifying {num_detections} detected boxes with ConvNeXt...")
    classified_results = []
    
    for i, box in enumerate(detections):
        # Get box coordinates
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        yolo_conf = box.conf[0].item()
        yolo_class = int(box.cls[0].item()) if hasattr(box, 'cls') else None
        
        # Pad box
        x1_pad, y1_pad, x2_pad, y2_pad = pad_box(x1, y1, x2, y2, img_w, img_h, BOX_PADDING)
        
        # Crop and classify
        crop_bgr = img_bgr[y1_pad:y2_pad, x1_pad:x2_pad]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        
        class_id, confidence, probs = classify_crop(classifier, crop_pil, device)
        class_name = CLASS_NAMES[class_id]
        
        classified_results.append({
            'box': (x1, y1, x2, y2),
            'padded_box': (x1_pad, y1_pad, x2_pad, y2_pad),
            'yolo_conf': yolo_conf,
            'yolo_class': yolo_class,
            'class_id': class_id,
            'class_name': class_name,
            'confidence': confidence,
            'probs': probs.cpu().tolist()
        })
        
        print(f"  Box {i+1}: {class_name} ({confidence:.2%}) | YOLO conf: {yolo_conf:.2%}")
    
    # Visualize results
    if save_output:
        output_path = image_path.rsplit('.', 1)[0] + '_classified.jpg'
        vis_img = img_bgr.copy()
        
        for i, det in enumerate(classified_results):
            x1, y1, x2, y2 = det['box']
            class_name = det['class_name']
            conf = det['confidence']
            
            # Draw box
            color = (0, 255, 0)  # Green
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Background rectangle for text
            cv2.rectangle(vis_img, (x1, y1 - text_h - baseline - 5), 
                         (x1 + text_w, y1), color, -1)
            cv2.putText(vis_img, label, (x1, y1 - baseline - 2), 
                       font, font_scale, (0, 0, 0), thickness)
        
        cv2.imwrite(output_path, vis_img)
        print(f"\n✓ Saved visualization to: {output_path}")
    
    return classified_results


def save_results_json(classified_results, image_path, output_path, yolo_model=None, classifier_model=None):
    """Persist detections and classifications to a JSON file for downstream routing."""
    holds_payload = []
    for idx, det in enumerate(classified_results):
        x1, y1, x2, y2 = det['box']
        px1, py1, px2, py2 = det['padded_box']
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)

        holds_payload.append({
            'id': idx,
            'cx': float(cx),
            'cy': float(cy),
            'class_id': int(det['class_id']),
            'class_name': det['class_name'],
            'confidence': float(det['confidence']),
            'yolo_conf': float(det['yolo_conf']),
            'yolo_class': int(det['yolo_class']) if det['yolo_class'] is not None else None,
            'box': [int(x1), int(y1), int(x2), int(y2)],
            'padded_box': [int(px1), int(py1), int(px2), int(py2)],
            'probs': [float(p) for p in det['probs']],
        })

    payload = {
        'image': image_path,
        'yolo_model': yolo_model,
        'classifier_model': classifier_model,
        'class_names': CLASS_NAMES,
        'num_classes': NUM_CLASSES,
        'holds': holds_payload,
    }

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as fp:
        json.dump(payload, fp, indent=2)
    print(f"\n✓ Saved JSON results to: {output_path}")


def main():
    global YOLO_CONF_THRESHOLD, BOX_PADDING
    parser = argparse.ArgumentParser(
        description="Two-stage detection + classification for climbing holds"
    )
    parser.add_argument(
        '-i', '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '-y', '--yolo',
        type=str,
        default=YOLO_MODEL,
        help='Path to YOLO model'
    )
    parser.add_argument(
        '-c', '--classifier',
        type=str,
        default=CONVNEXT_MODEL,
        help='Path to ConvNeXt classifier'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save visualization'
    )
    parser.add_argument(
        '--json-out',
        type=str,
        help='Path to write detections and classifications JSON'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=YOLO_CONF_THRESHOLD,
        help='YOLO confidence threshold'
    )
    parser.add_argument(
        '--padding',
        type=float,
        default=BOX_PADDING,
        help='Box padding fraction (0.15 = 15%%)'
    )
    
    args = parser.parse_args()
    
    # Update globals after parsing arguments
    YOLO_CONF_THRESHOLD = args.conf
    BOX_PADDING = args.padding
    
   
    # Load models
    detector = load_detector(args.yolo)
    classifier = load_classifier(args.classifier, DEVICE)
    
    # Run inference
    results = detect_and_classify(
        detector,
        classifier,
        args.image,
        DEVICE,
        save_output=not args.no_save
    )

    if args.json_out:
        save_results_json(
            results,
            args.image,
            args.json_out,
            yolo_model=args.yolo,
            classifier_model=args.classifier
        )
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total detections: {len(results)}")
    if results:
        for i, det in enumerate(results, 1):
            print(f"{i}. {det['class_name']} - {det['confidence']:.2%}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
