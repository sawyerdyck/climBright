import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

# =========================
# SETTINGS (EDIT THESE)
# =========================
DATA_DIR = "holds_cls"   # <-- change if your folder name is different
NUM_CLASSES = 6
BATCH_SIZE = 32
NUM_WORKERS = 0          # Windows safe. After it works, try 2.
PHASE_A_EPOCHS = 2
PHASE_B_EPOCHS = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "convnext_tiny.in12k_ft_in1k"
BEST_PATH = "best_convnext_two_phase.pt"
# =========================

def get_transforms():
    # Good default transforms for your task
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return train_tf, eval_tf

def make_loaders():
    train_tf, eval_tf = get_transforms()

    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=eval_tf)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda")
    )
    return train_ds, train_loader, val_loader

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total, correct = 0, 0
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)

def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler=None):
    model.train()
    total, correct = 0, 0
    total_loss = 0.0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision on GPU
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()  # cosine decay per batch

        total_loss += loss.item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)

def freeze_backbone_train_head(model):
    # Freeze all params
    for p in model.parameters():
        p.requires_grad = False

    # Unfreeze classifier head only
    for name, p in model.named_parameters():
        if name.startswith("head"):
            p.requires_grad = True

def unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True

def main():
    train_ds, train_loader, val_loader = make_loaders()
    print("Class mapping:", train_ds.class_to_idx)

    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)

    # Loss with label smoothing (good for similar-looking classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AMP scaler (enabled only on CUDA)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_val_acc = 0.0

    # =========================
    # PHASE A: head-only
    # =========================
    print("\n=== Phase A: train head only ===")
    freeze_backbone_train_head(model)

    optA = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=3e-4,
        weight_decay=0.05
    )

    for epoch in range(PHASE_A_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optA, criterion, scaler, scheduler=None)
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        print(f"[A {epoch+1}/{PHASE_A_EPOCHS}] train acc={tr_acc:.3f} loss={tr_loss:.4f} | val acc={va_acc:.3f} loss={va_loss:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), BEST_PATH)
            print(f"Saved best so far (val acc={best_val_acc:.3f})")

    # =========================
    # PHASE B: fine-tune all
    # =========================
    print("\n=== Phase B: fine-tune all layers ===")
    unfreeze_all(model)

    optB = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.05
    )

    # Cosine decay over ALL training steps in Phase B
    total_steps = PHASE_B_EPOCHS * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optB, T_max=total_steps)

    for epoch in range(PHASE_B_EPOCHS):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optB, criterion, scaler, scheduler=scheduler)
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        lr_now = optB.param_groups[0]["lr"]

        print(f"[B {epoch+1}/{PHASE_B_EPOCHS}] train acc={tr_acc:.3f} loss={tr_loss:.4f} | val acc={va_acc:.3f} loss={va_loss:.4f} | lr={lr_now:.2e}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), BEST_PATH)
            print(f"Saved best so far (val acc={best_val_acc:.3f})")

    print("\nDone.")
    print("Best model saved to:", BEST_PATH)
    print("Best validation accuracy:", round(best_val_acc, 4))

if __name__ == "__main__":
    main()
