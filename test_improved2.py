import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
import kornia.augmentation as K
import kornia.geometry.transform as KGT
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
# GPU AUGMENTATION (image + mask together)
# ============================================================
class PairedAug(nn.Module):
    """Applies geometric augmentations to both image and mask with the same params."""
    def __init__(self):
        super().__init__()
        self.hflip   = K.RandomHorizontalFlip(p=0.5)
        self.vflip   = K.RandomVerticalFlip(p=0.5)
        self.rotate  = K.RandomRotation(degrees=90.0)
        self.affine  = K.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1))
        # Color jitter only on image, not mask
        self.color   = K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

    def forward(self, imgs, masks):
        # ---- Geometric (same transform for both) ----
        imgs  = self.hflip(imgs);   masks = self.hflip(masks,  self.hflip._params)
        imgs  = self.vflip(imgs);   masks = self.vflip(masks,  self.vflip._params)
        imgs  = self.rotate(imgs);  masks = self.rotate(masks, self.rotate._params)
        imgs  = self.affine(imgs);  masks = self.affine(masks, self.affine._params)
        # ---- Color (image only) ----
        imgs  = self.color(imgs)
        return imgs, masks


# ============================================================
# PRETRAINED RESNET34 ENCODER + ATTENTION U-NET DECODER
# ============================================================
import torchvision.models as tvm

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        )
    def forward(self, x): return self.conv(x)


class ImprovedUNet(nn.Module):
    """Attention U-Net with pretrained ResNet34 encoder."""
    def __init__(self, out_channels=1):
        super().__init__()

        # ---- Pretrained ResNet34 Encoder ----
        resnet = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # 64,  H/2
        self.pool = resnet.maxpool                                          # 64,  H/4
        self.enc1 = resnet.layer1   # 64,  H/4
        self.enc2 = resnet.layer2   # 128, H/8
        self.enc3 = resnet.layer3   # 256, H/16
        self.enc4 = resnet.layer4   # 512, H/32

        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(DoubleConv(512, 1024), nn.Dropout2d(0.5))

        # ---- Decoder ----
        self.up4  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.ag4  = AttentionGate(512, 512, 256)
        self.dec4 = DoubleConv(1024, 512, dropout=0.3)

        self.up3  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ag3  = AttentionGate(256, 256, 128)
        self.dec3 = DoubleConv(512, 256, dropout=0.3)

        self.up2  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ag2  = AttentionGate(128, 128, 64)
        self.dec2 = DoubleConv(256, 128, dropout=0.2)

        self.up1  = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ag1  = AttentionGate(64, 64, 32)
        self.dec1 = DoubleConv(128, 64, dropout=0.1)

        self.up0  = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec0 = DoubleConv(32, 32)

        self.final = nn.Conv2d(32, out_channels, 1)

    def _pad(self, x, ref):
        return torch.nn.functional.pad(x, [0, ref.shape[3]-x.shape[3], 0, ref.shape[2]-x.shape[2]])

    def forward(self, x):
        e0 = self.enc0(x)           # 64,  H/2
        e1 = self.enc1(self.pool(e0))  # 64,  H/4
        e2 = self.enc2(e1)          # 128, H/8
        e3 = self.enc3(e2)          # 256, H/16
        e4 = self.enc4(e3)          # 512, H/32
        b  = self.bottleneck(e4)    # 1024,H/32

        up4 = self._pad(self.up4(b),  e4); d4 = self.dec4(torch.cat([up4, self.ag4(up4, e4)], 1))
        up3 = self._pad(self.up3(d4), e3); d3 = self.dec3(torch.cat([up3, self.ag3(up3, e3)], 1))
        up2 = self._pad(self.up2(d3), e2); d2 = self.dec2(torch.cat([up2, self.ag2(up2, e2)], 1))
        up1 = self._pad(self.up1(d2), e1); d1 = self.dec1(torch.cat([up1, self.ag1(up1, e1)], 1))
        up0 = self._pad(self.up0(d1), e0); d0 = self.dec0(up0)

        return self.final(d0)


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
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss.mean()


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
    paired_aug = PairedAug().to(device)

    # ---- Model ----
    model = ImprovedUNet(out_channels=1).to(device)
    print("✅ Improved2 U-Net (ResNet34 encoder) loaded")

    # ---- Loss, Optimizer, Scaler ----
    criterion = DiceBCELoss(weight_dice=0.5, weight_bce=0.5)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scaler    = GradScaler('cuda')

    # ---- LR Scheduler (more patient) ----
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # ---- Training Loop ----
    train_losses = []
    best_iou     = 0.0

    for epoch in range(EPOCHS):
        model.train()
        paired_aug.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for imgs, masks in loop:
            imgs, masks = imgs.to(device), masks.to(device)

            with torch.no_grad():
                imgs, masks = paired_aug(imgs, masks)

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

    model.load_state_dict(torch.load(
        os.path.join(RESULTS_DIR, "improved2_best.pth"),
        map_location=device, weights_only=True
    ))
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

    num_show = min(5, imgs.size(0))
    fig, axes = plt.subplots(num_show, 3, figsize=(12, num_show * 4))
    for i in range(num_show):
        img_np  = unnormalize(imgs[i]).permute(1, 2, 0).numpy()
        pred_np = preds[i].cpu().squeeze().numpy()
        mask_np = masks[i].cpu().squeeze().numpy()

        axes[i, 0].imshow(img_np);               axes[i, 0].set_title("Input Image");    axes[i, 0].axis("off")
        axes[i, 1].imshow(pred_np, cmap="gray"); axes[i, 1].set_title("Prediction");     axes[i, 1].axis("off")
        axes[i, 2].imshow(mask_np, cmap="gray"); axes[i, 2].set_title("Ground Truth");   axes[i, 2].axis("off")

    plt.suptitle(f"Improved2 U-Net | IoU: {final_iou:.4f} | Dice: {final_dice:.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "improved2_predictions.png"))
    plt.close()
    print("✅ Saved to Results/improved2_predictions.png")
    