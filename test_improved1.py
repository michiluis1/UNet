import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from dataset import get_dataloaders

# ---- Paths ----
TEST_IMG_DIR  = "data/ISIC2018_Task1-2_Test_Input/ISIC2018_Task1-2_Test_Input"
TEST_MASK_DIR = "data/ISIC2018_Task1_Test_GroundTruth"
RESULTS_DIR   = "Results"
MODEL_PATH    = "Results/improved1_best.pth"  # ← trained weights

# ---- Config ----
BATCH_SIZE  = 8
NUM_WORKERS = 4

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

# ---- Unnormalize helper ----
def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (tensor.cpu() * std + mean).clamp(0, 1)

if __name__ == '__main__':

    os.makedirs(RESULTS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Using device: {device}")

    # ---- Data (test only) ----
    _, test_loader = get_dataloaders(
        "data/ISIC2018_Task1-2_Training_Input/ISIC2018_Task1-2_Training_Input",
        "data/ISIC2018_Task1_Training_GroundTruth",
        TEST_IMG_DIR, TEST_MASK_DIR,
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # ---- Load Model ----
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,   # ← no pretrained weights, loading our own
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Model loaded from {MODEL_PATH}")

    # ---- Final Test Evaluation ----
    total_iou, total_dice, count = 0.0, 0.0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Testing"):
            imgs, masks = imgs.to(device), masks.to(device)
            with autocast():
                outputs = model(imgs)
            for i in range(outputs.size(0)):
                total_iou  += iou_score(outputs[i], masks[i]).item()
                total_dice += dice_score(outputs[i], masks[i]).item()
                count += 1

    final_iou  = total_iou / count
    final_dice = total_dice / count
    print(f"\n🏁 Improved1 Final Test IoU:  {final_iou:.4f}")
    print(f"🏁 Improved1 Final Test Dice: {final_dice:.4f}")

    # ---- Side-by-side figure ----
    print("\n🖼️  Saving side-by-side predictions...")

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
        axes[i, 1].set_title("Prediction")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(mask_np, cmap="gray")
        axes[i, 2].set_title("Ground Truth")
        axes[i, 2].axis("off")

    plt.suptitle(f"Improved1 U-Net | IoU: {final_iou:.4f} | Dice: {final_dice:.4f}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "improved1_predictions.png"))
    plt.close()
    print("✅ Saved to Results/improved1_predictions.png")