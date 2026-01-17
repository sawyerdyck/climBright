import os
from pathlib import Path
from PIL import Image

SRC = Path(r"C:\Users\sunna\Code\uottahacks\indoor-climbing-gym-hold-classification-dataset\Final_Dataset")   #C:\Users\Admin\Desktop\climBright\cr_data\Final_Dataset       # has train/valid/test
DST = Path("holds_cls")              # output classification dataset
SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

# Safety: skip tiny boxes (often junk/noise)
MIN_PIXELS = 24  # minimum width/height of crop in pixels

def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = (xc - w / 2) * W
    y1 = (yc - h / 2) * H
    x2 = (xc + w / 2) * W
    y2 = (yc + h / 2) * H
    return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def process_split(split):
    img_dir = SRC / split / "images"
    lbl_dir = SRC / split / "labels"
    out_split = SPLIT_MAP[split]

    for img_path in img_dir.glob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp"]:
            continue

        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        lines = label_path.read_text().strip().splitlines()
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:])

            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
            x1 = clamp(x1, 0, W - 1)
            y1 = clamp(y1, 0, H - 1)
            x2 = clamp(x2, 0, W - 1)
            y2 = clamp(y2, 0, H - 1)

            if x2 <= x1 or y2 <= y1:
                continue

            crop_w, crop_h = (x2 - x1), (y2 - y1)
            if crop_w < MIN_PIXELS or crop_h < MIN_PIXELS:
                continue

            crop = img.crop((x1, y1, x2, y2))

            out_dir = DST / out_split / str(cls)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_name = f"{img_path.stem}_box{i}.jpg"
            crop.save(out_dir / out_name, quality=95)

def main():
    for split in ["train", "valid", "test"]:
        process_split(split)
    print("Done. Classification crops saved to:", DST)

if __name__ == "__main__":
    main()
