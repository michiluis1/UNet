import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
import kornia.augmentation as K
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from dataset2 import get_dataloaders

# ---- Paths ----
TRAIN_IMG_DIR  = "data/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input"
TRAIN_MASK_DIR = "data/ISIC2018_Task1_Training_GroundTruth"
TEST_IMG_DIR   = "data/ISIC2018_Task1-2_Test_Input/ISIC2018_Task1-2_Test_Input"
TEST_MASK_DIR  = "data/ISIC2018_Task1_Test_GroundTruth"
RESULTS_DIR    = "Results"

# ---- Config ----
EPOCHS      = 20
BATCH_SIZE  = 4
LR          = 1e-4
NUM_WORKERS = 8

# ============================================================
# GPU AUGMENTATION
# ============================================================
def build_gpu_aug(device):
    return nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=90.0),
        K.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1),
        K.RandomAffine(degrees=0, translate=(0.1, 0.1),
                       scale=(0.9, 1.1))
    ).to(device)


# ============================================================
# ATTENTION GATE
# ============================================================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ============================================================
# IMPROVED U-NET ARCHITECTURE
# ============================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # ---- Encoder ----
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(
            DoubleConv(512, 1024),
            nn.Dropout2d(p=0.5)
        )

        # ---- Decoder ----
        self.up4  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.ag4  = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.dec4 = DoubleConv(1024, 512, dropout=0.3)

        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.ag3  = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec3 = DoubleConv(512, 256, dropout=0.3)

        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.ag2  = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec2 = DoubleConv(256, 128, dropout=0.2)

        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.ag1  = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec1 = DoubleConv(128, 64, dropout=0.1)

        # ---- Output ----
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        self.pool  = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))

        up4 = self.up4(b)
        up4 = torch.nn.functional.pad(up4, [0, e4.shape[3] - up4.shape[3],
                                             0, e4.shape[2] - up4.shape[2]])
        e4  = self.ag4(g=up4, x=e4)
        d4  = self.dec4(torch.cat([up4, e4], dim=1))

        up3 = self.up3(d4)
        up3 = torch.nn.functional.pad(up3, [0, e3.shape[3] - up3.shape[3],
                                             0, e3.shape[2] - up3.shape[2]])
        e3  = self.ag3(g=up3, x=e3)
        d3  = self.dec3(torch.cat([up3, e3], dim=1))

        up2 = self.up2(d3)
        up2 = torch.nn.functional.pad(up2, [0, e2.shape[3] - up2.shape[3],
                                             0, e2.shape[2] - up2.shape[2]])
        e2  = self.ag2(g=up2, x=e2)
        d2  = self.dec2(torch.cat([up2, e2], dim=1))

        up1 = self.up1(d2)
        up1 = torch.nn.functional.pad(up1, [0, e1.shape[3] - up1.shape[3],
                                             0, e1.shape[2] - up1.shape[2]])
        e1  = self.ag1(g=up1, x=e1)
        d1  = self.dec1(torch.cat([up1, e1], dim=1))

        return self.final(d1)


# ============================================================
# COMBINED DICE + BCE LOSS
# ============================================================
class DiceBCELoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce  = weight_bce
        self.bce         = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss     = self.bce(pred, target)
        pred_sig     = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum(dim=(2, 3))
        dice_loss    = 1 - (2 * intersection + 1e-6) / (
            pred_sig.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6
        )
        dice_loss = dice_loss.mean()
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


# ============================================================
# METRICS
# ============================================================
def iou_score(pred, target, threshold=0.5):
    pred         = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

def dice_score(pred, target, threshold=0.5):
    pred         = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum()
    return (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    cudnn.benchmark = True

    # ---- Data ----
    train_loader, test_loader = get_dataloaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        TEST_IMG_DIR,  TEST_MASK_DIR,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # ---- GPU Augmentation ----
    gpu_aug = build_gpu_aug(device)

    # ---- Model ----
    model = ImprovedUNet(in_channels=3, out_channels=1).to(device)
    print("✅ Improved2 U-Net loaded")

    # ---- Loss, Optimizer & Scaler ----
    criterion = DiceBCELoss(weight_dice=0.5, weight_bce=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler('cuda')

    # ---- LR Scheduler ----
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ---- Training Loop ----
    train_losses = []
    best_iou     = 0.0

    for epoch in range(EPOCHS):
        model.train()
        gpu_aug.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)

            # ---- GPU Augmentation ----
            with torch.no_grad():
                imgs  = gpu_aug(imgs)

            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(imgs)
                loss    = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        # ---- Per Epoch Evaluation ----
        model.eval()
        gpu_aug.eval()
        total_iou, total_dice, count = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast('cuda'):
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
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "improved2_best.pth"))
            print(f"  ✅ Best model saved (IoU: {best_iou:.4f})")

    # ---- Save Final Model ----
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "improved2_final.pth"))
    print("✅ Final model saved")

    # ---- Plot Training Loss ----
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label='Improved2 Loss')
    plt.title("Improved2 U-Net Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "improved2_training_loss.png"))
    plt.close()
    print("✅ Training loss saved")

    # ============================================================
    # FINAL TEST SESSION
    # ============================================================
    print("\n🔍 Running final test session with best saved model...")

    model.load_state_dict(torch.load(os.path.join(RESULTS_DIR, "improved2_best.pth")))
    model.eval()

    total_iou, total_dice, count = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Final Testing"):
            imgs, masks = imgs.to(device), masks.to(device)
            with autocast('cuda'):
                outputs = model(imgs)
            for i in range(outputs.size(0)):
                total_iou  += iou_score(outputs[i], masks[i]).item()
                total_dice += dice_score(outputs[i], masks[i]).item()
                count += 1

    final_iou  = total_iou / count
    final_dice = total_dice / count
    print(f"🏁 Improved2 Final Test IoU:  {final_iou:.4f}")
    print(f"🏁 Improved2 Final Test Dice: {final_dice:.4f}")

    # ---- Side-by-side figure ----
    print("\n🖼️  Saving side-by-side predictions...")

    def unnormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
        return (tensor.cpu() * std + mean).clamp(0, 1)

    imgs, masks = next(iter(test_loader))
    imgs, masks = imgs.to(device), masks.to(device)
    with torch.no_grad():
        with autocast('cuda'):
            outputs = model(imgs)
    preds = (torch.sigmoid(outputs) > 0.5).float()

    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    for i in range(min(5, imgs.size(0))):
        img_np  = unnormalize(imgs[i]).permute(1, 2, 0).numpy()
        pred_np = preds[i].cpu().squeeze().numpy()
        mask_np = masks[i].cpu().squeeze().numpy()

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Input Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_np, cmap="gray")
        axes[i, 1].set_title("Prediction")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mask_np, cmap="gray")
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis("off")

    plt.suptitle(f"Improved2 U-Net | IoU: {final_iou:.4f} | Dice: {final_dice:.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "improved2_predictions.png"))
    plt.close()
    print("✅ Saved to Results/improved2_predictions.png")
    