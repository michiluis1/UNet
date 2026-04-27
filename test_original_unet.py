import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ---- Paths ----
TEST_IMG_DIR  = "data/ISIC2018_Task1-2_Test_Input/ISIC2018_Task1-2_Test_Input"
TEST_MASK_DIR = "data/ISIC2018_Task1_Test_GroundTruth"
RESULTS_DIR   = "Results"
MODEL_PATH    = "Results/original_unet_best.pth"

# ---- Config ----
BATCH_SIZE  = 8
NUM_WORKERS = 4

# ---- Dataset ----
class ISIC2018Dataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])

        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
        ])

        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
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

        image = self.img_transform(image)
        mask  = self.mask_transform(mask)
        mask  = (mask > 0.5).float()

        return image, mask


# ---- U-Net Architecture (must match original_unet.py) ----
class DoubleConv(nn.Module):
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

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.bottleneck = nn.Sequential(
            DoubleConv(512, 1024),
            nn.Dropout2d(p=0.5)
        )

        self.up4  = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3  = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2  = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1  = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

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


if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ---- Data ----
    test_dataset = ISIC2018Dataset(TEST_IMG_DIR, TEST_MASK_DIR)
    test_loader  = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=2
    )
    print(f"✅ Test samples: {len(test_dataset)}")

    # ---- Load Model ----
    model = UNet(in_channels=3, out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Model loaded from {MODEL_PATH}")

    # ---- Final Test Evaluation ----
    total_iou, total_dice, count = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Testing"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            for i in range(outputs.size(0)):
                total_iou  += iou_score(outputs[i], masks[i]).item()
                total_dice += dice_score(outputs[i], masks[i]).item()
                count += 1

    final_iou  = total_iou / count
    final_dice = total_dice / count
    print(f"\n🏁 Original U-Net Final Test IoU:  {final_iou:.4f}")
    print(f"🏁 Original U-Net Final Test Dice: {final_dice:.4f}")

    # ---- Side-by-side figure ----
    print("\n🖼️  Saving side-by-side predictions...")

    imgs, masks = next(iter(test_loader))
    imgs, masks = imgs.to(device), masks.to(device)
    with torch.no_grad():
        outputs = model(imgs)
    preds = (torch.sigmoid(outputs) > 0.5).float()

    fig, axes = plt.subplots(5, 3, figsize=(12, 20))
    for i in range(5):
        img_np  = imgs[i].cpu().permute(1, 2, 0).numpy()
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

    plt.suptitle(f"Original U-Net | IoU: {final_iou:.4f} | Dice: {final_dice:.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "original_unet_predictions.png"))
    plt.close()
    print("✅ Saved to Results/original_unet_predictions.png")
    