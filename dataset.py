import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# ---- Dataset Class ----
class ISIC2018Dataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, mask_transform=None):
        self.img_dir       = img_dir
        self.mask_dir      = mask_dir
        self.transform     = transform
        self.mask_transform = mask_transform

        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith('.jpg') or f.endswith('.png')
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

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = (mask > 0.5).float()
        return image, mask


# ---- Transforms ----
def get_transforms():
    img_transform = transforms.Compose([
        transforms.Resize((767, 1022)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((767, 1022)),
        transforms.ToTensor()
    ])

    return img_transform, mask_transform


# ---- DataLoaders ----
def get_dataloaders(train_img_dir, train_mask_dir,
                    test_img_dir, test_mask_dir,
                    batch_size=8, num_workers=4):

    img_transform, mask_transform = get_transforms()

    train_dataset = ISIC2018Dataset(
        train_img_dir, train_mask_dir,
        transform=img_transform,
        mask_transform=mask_transform
    )

    test_dataset = ISIC2018Dataset(
        test_img_dir, test_mask_dir,
        transform=img_transform,
        mask_transform=mask_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=4,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=2
    )

    print(f"✅ Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    return train_loader, test_loader