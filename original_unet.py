import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image

# ---- Paths ----
TRAIN_IMG_DIR  = "data/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input"
TRAIN_MASK_DIR = "data/ISIC2018_Task1_Training_GroundTruth"
TEST_IMG_DIR   = "data/ISIC2018_Task1-2_Test_Input/ISIC2018_Task1-2_Test_Input"
TEST_MASK_DIR  = "data/ISIC2018_Task1_Test_GroundTruth"
RESULTS_DIR    = "Results"

# ---- Config ----
EPOCHS      = 20
BATCH_SIZE  = 2
LR          = 0.01       # Original paper uses SGD with lr=0.01
MOMENTUM    = 0.99       # Original paper momentum
NUM_WORKERS = 4

# ============================================================
# DATASET (with original paper augmentations)
# ============================================================
class ISIC2018Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, augment=False):
        self.img_dir   = img_dir
        self.mask_dir  = mask_dir
        self.augment   = augment

        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])

        # Base transforms (original paper used no normalization)
        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        # Augmentation transforms (original paper: flips + rotations)
        self.aug_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        img_id    = os.path.splitext(img_name)[0]
        img_path  = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_id + '_segmentation.png')

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')

        if self.augment:
            # Apply same random transform to both image and mask
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image = self.aug_transform(image)
            torch.manual_seed(seed)
            mask  = self.aug_transform(mask)

        image = self.img_transform(image)
        mask  = self.mask_transform(mask)
        mask  = (mask > 0.5).float()

        return image, mask


# ============================================================
# ORIGINAL U-NET ARCHITECTURE (Ronneberger et al. 2015)
# ============================================================
class DoubleConv(nn.Module):
    """Two consecutive Conv2d -> BatchNorm -> ReLU blocks"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
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
            nn.Dropout2d(p=0.5)   # Original paper has dropout in bottleneck
        )

        # ---- Decoder ----
        self.up4    = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4   = DoubleConv(1024, 512)

        self.up3    = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3   = DoubleConv(512, 256)

        self.up2    = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2   = DoubleConv(256, 128)

        self.up1    = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1   = DoubleConv(128, 64)

        # ---- Output ----
        self.final  = nn.Conv2d(64, out_channels, kernel_size=1)

        self.pool   = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b  = self.bottleneck(self.pool(e4))

        # Decoder with skip connections + padding fix
        up4 = self.up4(b)
        up4 = torch.nn.functional.pad(up4, [0, e4.shape[3] - up4.shape[3],
                                            0, e4.shape[2] - up4.shape[2]])
        d4  = self.dec4(torch.cat([up4, e4], dim=1))

        up3 = self.up3(d4)
        up3 = torch.nn.functional.pad(up3, [0, e3.shape[3] - up3.shape[3],
                                            0, e3.shape[2] - up3.shape[2]])
        d3  = self.dec3(torch.cat([up3, e3], dim=1))

        up2 = self.up2(d3)
        up2 = torch.nn.functional.pad(up2, [0, e2.shape[3] - up2.shape[3],
                                            0, e2.shape[2] - up2.shape[2]])
        d2  = self.dec2(torch.cat([up2, e2], dim=1))

        up1 = self.up1(d2)
        up1 = torch.nn.functional.pad(up1, [0, e1.shape[3] - up1.shape[3],
                                            0, e1.shape[2] - up1.shape[2]])
        d1  = self.dec1(torch.cat([up1, e1], dim=1))

        return self.final(d1)


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

    # ---- Data ----
    train_dataset = ISIC2018Dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, augment=True)
    test_dataset  = ISIC2018Dataset(TEST_IMG_DIR,  TEST_MASK_DIR,  augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=2
    )

    print(f"✅ Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # ---- Model ----
    model = UNet(in_channels=3, out_channels=1).to(device)
    print("✅ Original U-Net loaded (from scratch)")

    # ---- Loss & Optimizer (original paper: SGD + momentum) ----
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

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
            outputs = model(imgs)
            loss    = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # ---- Evaluate ----
        model.eval()
        total_iou, total_dice, count = 0.0, 0.0, 0
        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs, masks = imgs.to(device), masks.to(device)
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
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "original_unet_best.pth"))
            print(f"  ✅ Best model saved (IoU: {best_iou:.4f})")

    # ---- Save Final Model ----
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "original_unet_final.pth"))
    print("✅ Final model saved")

    # ---- Plot Training Loss ----
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label='Original U-Net Loss')
    plt.title("Original U-Net Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "original_unet_training_loss.png"))
    plt.close()
    print("✅ Training loss saved")
    print(f"🏁 Original U-Net Final IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f}")
    