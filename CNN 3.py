# -*- coding: utf-8 -*-
"""
Director field estimator - multi-patch version
Assumes: 160×160 .bmp = 4×4 stitched 40×40 patches
         Corresponding 16 small .vti files per image, named like base_0.vti ... base_15.vti
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

# ─── Visualization ──────────────────────────────────────────────────────────
def visualize_prediction(img_tensor, pred_tensor, target_tensor=None, title=""):
    img = img_tensor.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    img = np.clip(img, 0, 1)

    pred = pred_tensor.cpu().permute(1, 2, 0).numpy()
    angle_pred = np.arctan2(pred[...,1], pred[...,0]) % np.pi
    hue_pred = angle_pred / np.pi
    rgb_pred = hsv_to_rgb(np.stack([hue_pred, np.ones_like(hue_pred), np.ones_like(hue_pred)], -1))

    fig, axes = plt.subplots(1, 3 if target_tensor is not None else 2, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title("Input patch")
    axes[0].axis('off')

    axes[1].imshow(rgb_pred)
    axes[1].set_title("Prediction")
    axes[1].axis('off')

    if target_tensor is not None:
        tgt = target_tensor.cpu().permute(1, 2, 0).numpy()
        angle_tgt = np.arctan2(tgt[...,1], tgt[...,0]) % np.pi
        hue_tgt = angle_tgt / np.pi
        rgb_tgt = hsv_to_rgb(np.stack([hue_tgt, np.ones_like(hue_tgt), np.ones_like(hue_tgt)], -1))
        
        axes[2].imshow(rgb_tgt)
        axes[2].set_title("Ground Truth")
        axes[2].axis('off')

    # Colorbar (shared)
    colors = [(1,0,0),(1,1,0),(0,1,0),(0,1,1),(0,0,1),(1,0,1)]
    cmap = LinearSegmentedColormap.from_list("director", colors, N=256)
    norm = Normalize(0, np.pi)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', 
                 fraction=0.03, pad=0.04, shrink=0.7,
                 ticks=[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
                 label='Angle (0°–180°)')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


# ─── Device & Seeds ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─── Dataset ────────────────────────────────────────────────────────────────
class PatchDirectorDataset(Dataset):
    def __init__(self, root_dir, patch_size=40, patches_per_image=16,
                 random_crop=True, transform=None):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.random_crop = random_crop
        self.transform = transform

        bmp_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.bmp')]
        bmp_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        self.image_files = bmp_files

        print(f"Found {len(self.image_files)} large 160×160 images → "
              f"{len(self)} total patches")

    def __len__(self):
        return len(self.image_files) * self.patches_per_image

    def get_patch_position(self, patch_idx):
        """ 0..15 → row, col (4×4 grid, row-major) """
        row = patch_idx // 4
        col = patch_idx % 4
        return row, col

    def get_vti_name(self, base_name, patch_idx):
        """ Customize this according to your actual .vti naming """
        # Examples:
        # return f"{base_name}_{patch_idx:02d}.vti"          # 001_00.vti, 001_01.vti...
        # return f"{base_name}_p{patch_idx}.vti"
        # return f"patch_{base_name}_{patch_idx}.vti"
        return f"{base_name}.vti"   # ← change here to match your files

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        patch_subidx = idx % self.patches_per_image

        img_name = self.image_files[img_idx]
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.root_dir, img_name)

        # ── Load image patch ────────────────────────────────────────
        image_full = Image.open(img_path).convert('RGB')

        row, col = self.get_patch_position(patch_subidx)
        left = col * self.patch_size
        upper = row * self.patch_size

        if self.random_crop:
            offset = random.randint(-4, 4)   # small jitter
            left = max(0, min(160 - self.patch_size, left + offset)
)
            upper = max(0, min(160 - self.patch_size, upper + offset))

        image_crop = image_full.crop((left, upper, left + self.patch_size, upper + self.patch_size))

        if self.transform:
            image = self.transform(image_crop)
        else:
            image = transforms.ToTensor()(image_crop)

        # ── Load high-resolution director field ──────────────────────────
        vti_name = self.get_vti_name(base, patch_subidx)
        vti_path = os.path.join(self.root_dir, vti_name)

        if not os.path.exists(vti_path):
            raise FileNotFoundError(f"Missing .vti for patch {patch_subidx}: {vti_path}")

        mesh = pv.read(vti_path)
        array_name = 'n'

        if array_name in mesh.point_data:
            raw_data = mesh.point_data[array_name]
            is_point = True
        elif array_name in mesh.cell_data:
            raw_data = mesh.cell_data[array_name]
            is_point = False
        else:
            available = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
            raise KeyError(f"No '{array_name}' array found. Available: {available}")

        # ── Reshape raw data to grid ─────────────────────────────────────
        dims = mesh.dimensions
        if is_point:
            resh_shape = (dims[2], dims[1], dims[0], -1)
            n_elements = mesh.n_points
        else:
            resh_shape = (max(dims[2]-1,1), max(dims[1]-1,1), max(dims[0]-1,1), -1)
            n_elements = mesh.n_cells

        if raw_data.size != n_elements * raw_data.shape[1] if raw_data.ndim == 2 else n_elements:
            raise ValueError(f"Array size mismatch: {raw_data.shape} vs expected {n_elements}")

        data = raw_data.reshape(resh_shape)

        if data.shape[0] > 1:
            data = data[data.shape[0] // 2]  # take mid slice if 3D
        else:
            data = data[0]  # squeeze Z=1

        # data now (ny, nx, n_comp)

        if data.shape[0] != data.shape[1]:
            raise ValueError(f"Non-square director field: {data.shape[:2]}")

        # ── Downsample if higher resolution ──────────────────────────────
        h = data.shape[0]
        if h % self.patch_size != 0:
            raise ValueError(f"Director resolution {h} not divisible by patch size {self.patch_size}")

        target_high = torch.from_numpy(data[..., :2].astype(np.float32)).permute(2, 0, 1)  # (2, h, h)

        if h == self.patch_size:
            target = target_high
        else:
            kernel = h // self.patch_size
            # Avg pool each channel
            target = F.avg_pool2d(target_high.unsqueeze(0), kernel_size=kernel, stride=kernel).squeeze(0)  # (2, 40, 40)

        # Normalize to unit length
        norm = torch.norm(target, dim=0, keepdim=True).clamp_min(1e-8)
        target = target / norm

        return image, target


# ─── Transforms ─────────────────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(180),          # 180° symmetry for non-polar directors
    transforms.ColorJitter(0.12, 0.12, 0.08),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# ─── Model ──────────────────────────────────────────────────────────────────
class MiniUNetDirector(nn.Module):
    def __init__(self, in_ch=3, out_ch=2):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True)
            )

        self.enc1 = block(in_ch, 64)
        self.enc2 = block(64, 128)
        self.enc3 = block(128, 256)
        self.bottleneck = block(256, 512)

        self.dec3 = block(512+256, 256)
        self.dec2 = block(256+128, 128)
        self.dec1 = block(128+64, 64)

        self.final = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        b  = self.bottleneck(F.max_pool2d(e3, 2))

        d3 = F.interpolate(b, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], 1)
        d3 = self.dec3(d3)

        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], 1)
        d2 = self.dec2(d2)

        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], 1)
        d1 = self.dec1(d1)

        out = self.final(d1)
        return F.normalize(out, p=2, dim=1)


# ─── Loss & Metrics ─────────────────────────────────────────────────────────
def director_loss(pred, targ):
    cos = (pred * targ).sum(dim=1).clamp(-0.9999, 0.9999)
    return (1 - cos).mean() + 0.15 * F.mse_loss(pred, targ)


def mae_deg(pred, targ):
    cos = (pred * targ).sum(dim=1).clamp(-0.9999, 0.9999)
    return torch.acos(cos).mean().item() * (180 / np.pi)


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    folder = r"C:/Users/user/Downloads/FYP/Director_simulations/Test"   # ← update path

    train_ds = PatchDirectorDataset(folder, patches_per_image=16, random_crop=True,  transform=train_transform)
    val_ds   = PatchDirectorDataset(folder, patches_per_image=4,  random_crop=False, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = MiniUNetDirector().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=6)

    best_val_mae = 999
    for epoch in range(5):
        model.train()
        train_loss = train_mae = 0
        n = 0

        for imgs, tgts in tqdm(train_loader, desc=f"Epoch {epoch+1} train"):
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

        # ── Validation ──────────────────────────────────────────────
        model.eval()
        val_loss = val_mae = 0
        n = 0
        with torch.no_grad():
            for imgs, tgts in val_loader:
                imgs, tgts = imgs.to(device), tgts.to(device)
                pred = model(imgs)
                val_loss += director_loss(pred, tgts).item()
                val_mae += mae_deg(pred, tgts)
                n += 1

        val_loss /= n
        val_mae /= n

        print(f"[{epoch+1:3d}]  train loss {train_loss:.4f}  mae {train_mae:5.2f}°   "
              f" val loss {val_loss:.4f}  mae {val_mae:5.2f}°")

        scheduler.step(val_loss)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "best_director_model.pth")
            print(f"  → saved better model  (val mae = {val_mae:.2f}°)")

        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                imgs, tgts = next(iter(val_loader))
                preds = model(imgs.to(device))
                for i in range(min(4, len(imgs))):
                    visualize_prediction(
                        imgs[i], preds[i].cpu(), tgts[i],
                        title=f"Epoch {epoch+1}  val sample {i}   mae {mae_deg(preds[i:i+1], tgts[i:i+1]):.1f}°"
                    )

    print("Training finished.")

if __name__ == '__main__':
    main()