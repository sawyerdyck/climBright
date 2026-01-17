import os, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

DATA = "holds_cls"      # change if needed
NUM_CLASSES = 6
BATCH = 32
EPOCHS = 3              # short on purpose
MAX_TRAIN_SAMPLES = 200 # tiny subset for smoke test
MAX_VAL_SAMPLES = 200

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_subset(ds, n):
    if n is None or n >= len(ds):
        return ds
    idx = list(range(len(ds)))
    random.shuffle(idx)
    return Subset(ds, idx[:n])

def main():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225)),
    ])

    train_ds_full = datasets.ImageFolder(os.path.join(DATA, "train"), transform=train_tf)
    val_ds_full   = datasets.ImageFolder(os.path.join(DATA, "val"),   transform=eval_tf)

    print("Class mapping:", train_ds_full.class_to_idx)

    train_ds = make_subset(train_ds_full, MAX_TRAIN_SAMPLES)
    val_ds   = make_subset(val_ds_full, MAX_VAL_SAMPLES)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=2)

    model = timm.create_model("convnext_tiny.in12k_ft_in1k", pretrained=True, num_classes=NUM_CLASSES).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    for epoch in range(EPOCHS):
        model.train()
        correct = total = 0
        pbar = tqdm(train_loader, desc=f"Train {epoch+1}/{EPOCHS}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            pbar.set_postfix(loss=float(loss.item()), acc=correct/max(total,1))

        model.eval()
        v_correct = v_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                pred = logits.argmax(1)
                v_correct += (pred == y).sum().item()
                v_total += y.size(0)
        print(f"Val acc: {v_correct/max(v_total,1):.3f}")

if __name__ == "__main__":
    main()
