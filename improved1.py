import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from dataset import get_dataloaders

# ---- Paths ----
TRAIN_IMG_DIR  = "data/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input"
TRAIN_MASK_DIR = "data/ISIC2018_Task1_Training_GroundTruth"
TEST_IMG_DIR   = "data/ISIC2018_Task1-2_Test_Input/ISIC2018_Task1-2_Test_Input"
TEST_MASK_DIR  = "data/ISIC2018_Task1_Test_GroundTruth"
RESULTS_DIR    = "Results"

# ---- Config ----
EPOCHS      = 20
BATCH_SIZE  = 8
LR          = 1e-4
NUM_WORKERS = 4

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ---- cuDNN Benchmark ----
    cudnn.benchmark = True

    # ---- Data ----
    train_loader, test_loader = get_dataloaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        TEST_IMG_DIR, TEST_MASK_DIR,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # ---- Model ----
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)
    print("✅ Improved1 U-Net loaded")

    # ---- Loss, Optimizer & Scaler ----
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler()

    # ---- Metrics ----
    def iou_score(pred, target, threshold=0.5):
        pred         = (torch.sigmoid(pred) > threshold).float()
        intersection = (pred * target).sum()
        union        = pred.sum() + target.sum() - intersection
        return (intersection + 1e-6) / (union + 1e-6)

    def dice_score(pred, target, threshold=0.5):
        pred         = (torch.sigmoid(pred) > threshold).float()
        intersection = (pred * target).sum()
        return (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

    # ---- Training Loop ----
    train_losses = []
    best_iou     = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(imgs)
                loss    = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ---- Per-Epoch Evaluate ----
        model.eval()
        total_iou, total_dice, count = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast():
                    outputs = model(imgs)
                for i in range(outputs.size(0)):
                    total_iou  += iou_score(outputs[i], masks[i]).item()
                    total_dice += dice_score(outputs[i], masks[i]).item()
                    count += 1

        avg_iou  = total_iou / count
        avg_dice = total_dice / count
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f}")

        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "improved1_best.pth"))
            print(f"  ✅ Best model saved (IoU: {best_iou:.4f})")

    # ---- Save Final Model ----
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "improved1_final.pth"))
    print("✅ Final model saved")

    # ---- Plot Training Loss ----
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label='Improved1 Loss')
    plt.title("Improved1 U-Net Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "improved1_training_loss.png"))
    plt.close()
    print("✅ Training loss saved")

    # ============================================================
    # FINAL TEST SESSION — loads best model and evaluates
    # ============================================================
    print("\n🔍 Running final test session with best saved model...")

    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "improved1_best.pth")))
    model.eval()

    total_iou, total_dice, count = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Final Testing"):
            imgs, masks = imgs.to(device), masks.to(device)
            with autocast():
                outputs = model(imgs)
            for i in range(outputs.size(0)):
                total_iou  += iou_score(outputs[i], masks[i]).item()
                total_dice += dice_score(outputs[i], masks[i]).item()
                count += 1

    final_iou  = total_iou / count
    final_dice = total_dice / count
    print(f"🏁 Improved1 Final Test IoU: {final_iou:.4f} | Dice: {final_dice:.4f}")

    # ---- Side-by-side figure: image | prediction | ground truth ----
    print("🖼️  Saving side-by-side predictions...")

    # Unnormalize helper
    def unnormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        return (tensor.cpu() * std + mean).clamp(0, 1)

    imgs, masks = next(iter(test_loader))
    imgs, masks = imgs.to(device), masks.to(device)
    with torch.no_grad():
        with autocast():
            outputs = model(imgs)
    preds = (torch.sigmoid(outputs) > 0.5).float()

    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    for i in range(5):
        img_np  = unnormalize(imgs[i]).permute(1, 2, 0).numpy()
        pred_np = preds[i].cpu().squeeze().numpy()
        mask_np = masks[i].cpu().squeeze().numpy()

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_np, cmap="gray")
        axes[i, 1].set_title(f"Prediction")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mask_np, cmap="gray")
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis("off")

    plt.suptitle(f"Improved1 U-Net | IoU: {final_iou:.4f} | Dice: {final_dice:.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "improved1_predictions.png"))
    plt.close()
    print("✅ Predictions saved to Results/improved1_predictions.png")
    