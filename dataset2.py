import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from PIL import Image

# ---- Dataset Class ----
class ISIC2018Dataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir  = img_dir
        self.mask_dir = mask_dir

        self.images = sorted([
            f for f in os.listdir(img_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])

        # ---- Image transform ----
        self.img_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # ---- Mask transform ----
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
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


# ---- DataLoaders ----
def get_dataloaders(train_img_dir, train_mask_dir,
                    test_img_dir, test_mask_dir,
                    batch_size=4, num_workers=8):

    train_dataset = ISIC2018Dataset(train_img_dir, train_mask_dir)
    test_dataset  = ISIC2018Dataset(test_img_dir,  test_mask_dir)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=4
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True, persistent_workers=True,
        prefetch_factor=2
    )

    print(f"✅ Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")
    return train_loader, test_loader