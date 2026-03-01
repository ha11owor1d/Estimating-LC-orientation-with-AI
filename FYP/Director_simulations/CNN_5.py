# -*- coding: utf-8 -*-
"""
ZERO-VALIDATION VERSION - Validation and test sets are completely untouched during training
"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pyvista as pv
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, Normalize
from matplotlib.colors import LinearSegmentedColormap
from torch.optim.lr_scheduler import CosineAnnealingLR

# ─── Visualization ──────────────────────────────────────────────────────────
def visualize_prediction(img_tensor, pred_tensor, target_tensor=None, title="", save_path=None):
    img = img_tensor.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    img = np.clip(img, 0, 1)

    pred = pred_tensor.cpu().permute(1, 2, 0).numpy()
    angle_pred = np.arctan2(pred[...,1], pred[...,0]) % np.pi
    hue_pred = angle_pred / np.pi
    rgb_pred = hsv_to_rgb(np.stack([hue_pred, np.ones_like(hue_pred), np.ones_like(hue_pred)], -1))

    theta = angle_pred - 0.0
    I = np.sin(2 * theta) ** 2
    I = np.clip(I, 0, 1)
    I = 0.2 + 0.8 * I
    fake_texture = np.stack([I, I, I], axis=-1)

    has_gt = target_tensor is not None
    num_panels = 4 if has_gt else 3

    fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 5))

    if has_gt:
        tgt = target_tensor.cpu().permute(1, 2, 0).numpy()
        angle_tgt = np.arctan2(tgt[...,1], tgt[...,0]) % np.pi
        hue_tgt = angle_tgt / np.pi
        rgb_tgt = hsv_to_rgb(np.stack([hue_tgt, np.ones_like(hue_tgt), np.ones_like(hue_tgt)], -1))
        axes[0].imshow(rgb_tgt)
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')
        panel_idx = 1
    else:
        panel_idx = 0

    axes[panel_idx].imshow(img)
    axes[panel_idx].set_title("Input patch")
    axes[panel_idx].axis('off')
    panel_idx += 1

    axes[panel_idx].imshow(rgb_pred)
    axes[panel_idx].set_title("Prediction")
    axes[panel_idx].axis('off')
    panel_idx += 1

    axes[panel_idx].imshow(fake_texture)
    axes[panel_idx].set_title("Fake texture\nfrom prediction")
    axes[panel_idx].axis('off')

    colors = [(1,0,0),(1,1,0),(0,1,0),(0,1,1),(0,0,1),(1,0,1)]
    cmap = LinearSegmentedColormap.from_list("director", colors, N=256)
    norm = Normalize(0, np.pi)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=axes[panel_idx-1], orientation='vertical',
                        fraction=0.046, pad=0.04, shrink=0.7,
                        ticks=[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    cbar.ax.set_yticklabels(['0°', '45°', '90°', '135°', '180°'])
    cbar.set_label('Angle (0°–180°)')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)


# ─── Dataset (same as before) ───────────────────────────────────────────────
class PatchDirectorDataset(Dataset):
    def __init__(self, root_dir, patch_size=40, patches_per_image=16,
                 random_crop=True, transform=None, image_files=None):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.random_crop = random_crop
        self.transform = transform

        if image_files is not None:
            self.image_files = image_files
        else:
            bmp_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.bmp')]
            bmp_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
            self.image_files = bmp_files

        print(f"Dataset loaded: {len(self.image_files)} images")

    def __len__(self):
        return len(self.image_files) * self.patches_per_image

    def get_patch_position(self, patch_idx):
        row = patch_idx // 4
        col = patch_idx % 4
        return row, col

    def get_vti_name(self, base_name, patch_idx):
        return f"{base_name}_{patch_idx}.vti"

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        patch_subidx = idx % self.patches_per_image

        img_name = self.image_files[img_idx]
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.root_dir, img_name)

        image_full = Image.open(img_path).convert('RGB')

        row, col = self.get_patch_position(patch_subidx)
        left = col * self.patch_size
        upper = row * self.patch_size

        if self.random_crop:
            offset = random.randint(-4, 4)
            left = max(0, min(160 - self.patch_size, left + offset))
            upper = max(0, min(160 - self.patch_size, upper + offset))

        image_crop = image_full.crop((left, upper, left + self.patch_size, upper + self.patch_size))
        image_crop = image_crop.transpose(Image.FLIP_TOP_BOTTOM)   # match PyVista

        if self.transform:
            image = self.transform(image_crop)
        else:
            image = transforms.ToTensor()(image_crop)

        # Director field loading (same as before)
        vti_name = self.get_vti_name(base, patch_subidx)
        vti_path = os.path.join(self.root_dir, vti_name)

        mesh = pv.read(vti_path)
        raw_data = mesh.point_data.get('n') or mesh.cell_data.get('n')

        if raw_data.ndim == 1:
            side = int(np.sqrt(len(raw_data) // 3))
            data = raw_data.reshape(side, side, 3)
        else:
            data = raw_data

        target = torch.from_numpy(data[..., :2].astype(np.float32)).permute(2, 0, 1)
        norm = torch.norm(target, dim=0, keepdim=True).clamp_min(1e-8)
        target = target / norm

        return image, target


# ─── Transforms, Model, Loss (same as before) ───────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),
    transforms.RandomRotation(degrees=(-45, 45)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15), shear=15),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5]*3, [0.5]*3)])

# ─── Model ──────────────────────────────────────────────────────────────────
class DirectorUNet(nn.Module):
    """ 4-level deep U-Net (128-256-512-1024) — FIXED for 40×40 """
    def __init__(self, in_channels=3, out_channels=2):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        # Encoder
        self.enc1 = conv_block(in_channels, 128)
        self.enc2 = conv_block(128, 256)
        self.enc3 = conv_block(256, 512)
        self.enc4 = conv_block(512, 1024)

        # Bottleneck
        self.bottleneck = conv_block(1024, 1024)

        # Decoder (CORRECTED channel counts)
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1536, 512)          # ← 1024 + 512 = 1536

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(768, 256)           # 512 + 256

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(384, 128)           # 256 + 128

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = conv_block(256, 128)           # 128 + 128

        self.final = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        b = self.bottleneck(F.max_pool2d(e4, 2))

        d4 = self.up4(b)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return F.normalize(out, p=2, dim=1)

# ─── Loss & Metrics ─────────────────────────────────────────────────────────
def director_loss(pred, targ):
    cos = (pred * targ).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos).mean()

def mae_deg(pred, targ):
    cos = (pred * targ).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
    return torch.acos(cos).mean().item() * (180 / np.pi)


# ─── MAIN - ZERO VALIDATION VERSION ─────────────────────────────────────────
def main():
    root_dir = r"C:/Users/user/Downloads/FYP/Director_simulations/Test"

    # Delete old images
    for f in os.listdir("visualizations"):
        if f.endswith(".png"):
            os.remove(os.path.join("visualizations", f))

    # ── Split once ─────────────────────────────────────────────────────
    all_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.bmp')]
    all_files.sort(key=lambda x: int(os.path.splitext(x)[0]))

    random.seed(42)
    indices = list(range(len(all_files)))
    random.shuffle(indices)

    n_train = int(0.70 * len(indices))
    n_val   = int(0.15 * len(indices))
    # n_test  = len(indices) - n_train - n_val

    train_files = [all_files[i] for i in indices[:n_train]]
    val_files   = [all_files[i] for i in indices[n_train:n_train+n_val]]
    test_files  = [all_files[i] for i in indices[n_train+n_val:]]

    train_ds = PatchDirectorDataset(root_dir, patches_per_image=16, random_crop=True,  transform=train_transform, image_files=train_files)
    val_ds   = PatchDirectorDataset(root_dir, patches_per_image=4,  random_crop=False, transform=val_transform, image_files=val_files)
    test_ds  = PatchDirectorDataset(root_dir, patches_per_image=4,  random_crop=False, transform=val_transform, image_files=test_files)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DirectorUNet().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=5e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)

    best_train_mae = float('inf')

    for epoch in range(600):
        model.train()
        train_loss = train_mae = 0
        n = 0

        for imgs, tgts in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, tgts = imgs.to(device), tgts.to(device)
            optimizer.zero_grad()
            pred = model(imgs)
            loss = director_loss(pred, tgts)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_mae += mae_deg(pred, tgts)
            n += 1

        train_loss /= n
        train_mae /= n

        scheduler.step()

        print(f"[{epoch+1:3d}] train loss {train_loss:.4f}  mae {train_mae:5.2f}°")

        if train_mae < best_train_mae:
            best_train_mae = train_mae
            torch.save(model.state_dict(), "best_director_model.pth")

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1:04d}.pth")

    print("Training finished - validation was NEVER used.")

    # ── Final evaluation on val and test (only now) ─────────────────────
    model.load_state_dict(torch.load("best_director_model.pth"))
    model.eval()

    for name, loader in [("Validation", val_loader), ("Test", test_loader)]:
        loss = mae = 0
        n = 0
        with torch.no_grad():
            for imgs, tgts in loader:
                imgs, tgts = imgs.to(device), tgts.to(device)
                pred = model(imgs)
                loss += director_loss(pred, tgts).item()
                mae += mae_deg(pred, tgts)
                n += 1
        loss /= n
        mae /= n
        print(f"Final {name} MAE: {mae:.2f}°")
    with torch.no_grad():
        imgs, tgts = next(iter(val_loader))
        preds = model(imgs.to(device))

        # Create save directory if needed
        save_dir = "visualizations"
        os.makedirs(save_dir, exist_ok=True)

        num_save = min(32, len(imgs))  # Save first 4 samples (adjust as needed)

        for i in range(num_save):
            mae = mae_deg(preds[i:i+1], tgts[i:i+1])
            title = f"Epoch {epoch+1} val sample {i} mae {mae:.1f}°"
            filename = f"epoch_{epoch+1:03d}_sample_{i:02d}_mae_{mae:05.1f}.png"
            save_path = os.path.join(save_dir, filename)

            visualize_prediction(
                imgs[i].cpu(), preds[i].cpu(), tgts[i].cpu(),
                title=title,
                save_path=save_path  # This triggers saving
            )

if __name__ == '__main__':
    main()