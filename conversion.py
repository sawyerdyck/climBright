import os
import shutil
from pathlib import Path

# ====== EDIT THESE ======
LARGE_ROOT = Path("holds_cls")        # your big classification dataset root
SMALL_ROOT = Path("holds_cls_small")  # your small test dataset root
N_PER_CLASS = 20

# Which splits to sample (only include the ones you have)
SPLITS = ["train", "val", "test"]

# Image extensions to consider
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
# ========================

def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def copy_first_n(src_dir: Path, dst_dir: Path, n: int) -> int:
    if not src_dir.exists():
        print(f"Skip (missing): {src_dir}")
        return 0

    ensure_dir(dst_dir)

    # "First" = sorted by filename for deterministic behavior
    imgs = sorted([p for p in src_dir.iterdir() if is_image(p)])
    picked = imgs[:n]

    count = 0
    for p in picked:
        # Avoid overwriting: add suffix if needed
        dst_path = dst_dir / p.name
        if dst_path.exists():
            stem, ext = p.stem, p.suffix
            k = 1
            while True:
                candidate = dst_dir / f"{stem}__copy{k}{ext}"
                if not candidate.exists():
                    dst_path = candidate
                    break
                k += 1

        shutil.copy2(p, dst_path)
        count += 1

    return count

def main():
    total_copied = 0

    for split in SPLITS:
        split_src = LARGE_ROOT / split
        if not split_src.exists():
            continue

        # Detect class folders (e.g., 0..5)
        class_folders = sorted([d for d in split_src.iterdir() if d.is_dir()])
        if not class_folders:
            print(f"No class folders found in: {split_src}")
            continue

        for cls_dir in class_folders:
            cls_name = cls_dir.name
            src_dir = cls_dir
            dst_dir = SMALL_ROOT / split / cls_name

            copied = copy_first_n(src_dir, dst_dir, N_PER_CLASS)
            total_copied += copied
            print(f"{split}/{cls_name}: copied {copied} images")

    print(f"\nDone. Total copied: {total_copied}")
    print(f"Small dataset at: {SMALL_ROOT.resolve()}")

if __name__ == "__main__":
    main()
