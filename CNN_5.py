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
from torch.optim.lr_scheduler import CosineAnnealingLR
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# ─── Visualization ──────────────────────────────────────────────────────────
def visualize_prediction(img_tensor, pred_tensor, target_tensor=None, title="", save_path=None):
    """
    4-panel version:
    1. Ground Truth (if available)
    2. Input patch
    3. Prediction (HSV hue encoding)
    4. Synthesized texture from prediction (simple crossed-polarizer style)
    """
    # ── Prepare data ────────────────────────────────────────────────────────
    img = img_tensor.cpu().permute(1, 2, 0).numpy() * 0.5 + 0.5
    img = np.clip(img, 0, 1)

    pred = pred_tensor.cpu().permute(1, 2, 0).numpy()
    angle_pred = np.arctan2(pred[...,1], pred[...,0]) % np.pi
    hue_pred = angle_pred / np.pi
    rgb_pred = hsv_to_rgb(np.stack([hue_pred, np.ones_like(hue_pred), np.ones_like(hue_pred)], -1))

    # Synthesized fake texture from prediction (simple crossed pol)
    theta = angle_pred - 0.0  # polarizer at 0°
    I = np.sin(2 * theta) ** 2
    I = np.clip(I, 0, 1)
    # Optional: slight contrast boost
    I = 0.2 + 0.8 * I
    fake_texture = np.stack([I, I, I], axis=-1)  # grayscale

    # ── Create figure ───────────────────────────────────────────────────────
    has_gt = target_tensor is not None
    num_panels = 4 if has_gt else 3

    fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 5))

    # Ground Truth (leftmost, if present)
    if has_gt:
        tgt = target_tensor.cpu().permute(1, 2, 0).numpy()
        angle_tgt = np.arctan2(tgt[...,1], tgt[...,0]) % np.pi
        hue_tgt = angle_tgt / np.pi
        rgb_tgt = hsv_to_rgb(np.stack([hue_tgt, np.ones_like(hue_tgt), np.ones_like(hue_tgt)], -1))
        
        axes[0].imshow(rgb_tgt, origin='lower')
        axes[0].set_title("Ground Truth")
        #axes[0].axis('off')
        panel_idx = 1
    else:
        panel_idx = 0

    # Input patch
    axes[panel_idx].imshow(img, origin='lower')
    axes[panel_idx].set_title("Input patch")
    #axes[panel_idx].axis('off')
    
    panel_idx += 1

    # Prediction (HSV)
    axes[panel_idx].imshow(rgb_pred, origin='lower')
    axes[panel_idx].set_title("Prediction")
    #axes[panel_idx].axis('off')
    panel_idx += 1

    # Fake synthesized texture
    axes[panel_idx].imshow(fake_texture, origin='lower')
    axes[panel_idx].set_title("Fake texture\nfrom prediction")
    axes[panel_idx].axis('off')

    # Colorbar (attached to prediction panel)
    colors = [(1,0,0),(1,1,0),(0,1,0),(0,1,1),(0,0,1),(1,0,1)]
    cmap = LinearSegmentedColormap.from_list("director", colors, N=256)
    norm = Normalize(0, np.pi)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=axes[panel_idx-1], orientation='vertical', 
                 fraction=0.046, pad=0.04, shrink=0.7,
                 ticks=[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi]
                 )
    cbar.ax.set_yticklabels(['0°', '45°', '90°', '135°', '180°'])
    cbar.set_label('Angle (0°–180°)')

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)


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
    
# ─── Director-specific post-correction ──────────────────────────────────────
def correct_director_after_aug(director: np.ndarray, transform_params: dict) -> np.ndarray:
    """
    Apply vector-specific corrections after Albumentations geometric transforms.
    director shape: (H, W, 2)  ──  [nx, ny]
    """
    # 1. Horizontal flip → negate nx
    if transform_params.get('horizontal_flip', False):
        director[..., 0] = -director[..., 0]

    # 2. Vertical flip → negate ny
    if transform_params.get('vertical_flip', False):
        director[..., 1] = -director[..., 1]

    # 3. Rotation or arbitrary affine → rotate vectors
    #    Albumentations stores the rotation angle in degrees (positive = CCW)
    angle_deg = transform_params.get('angle', 0.0)
    if abs(angle_deg) > 1e-6:
        theta = np.deg2rad(-angle_deg)          # inverse rotation for vectors!
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        nx = director[..., 0]
        ny = director[..., 1]
        director[..., 0] =  cos_t * nx + sin_t * ny
        director[..., 1] = -sin_t * nx + cos_t * ny

    # For full Affine (shift, scale, rotate, shear) you would need the full matrix.
    # Here we handle the most common case (Rotate + Flip). For full support see note below.

    return director


# ─── Albumentations pipelines ──────────────────────────────────────────────
# ── Geometric transforms ── (flips, rotations, affine → safe for both image & director)
geometric_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Rotate(limit=(-30, 30), p=0.7, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.15,
        rotate_limit=0,
        p=0.6,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
    ),
], additional_targets={'director': 'image'})

# ── Color-only transforms ── (only apply to RGB image)
color_transform = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
    A.ToFloat(max_value=255),                      # ← KEY FIX: uint8 → float32 [0,1]
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
])


val_transform = A.Compose([
    A.ToFloat(max_value=255),                      # ← KEY FIX: uint8 → float32 [0,1]
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2()
], additional_targets={'director': 'image'})


# ─── Updated Dataset ────────────────────────────────────────────────────────
class PatchDirectorDataset(Dataset):
    def __init__(self, root_dir, patch_size=40, patches_per_image=16,
                 is_train=True, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.is_train = is_train
        self.transform = transform

        bmp_files = [f for f in os.listdir(root_dir) if f.lower().endswith('.bmp')]
        bmp_files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        self.image_files = bmp_files

        print(f"Found {len(self.image_files)} images → {len(self)} patches "
              f"({'train' if is_train else 'val/test'})")

    def __len__(self):
        return len(self.image_files) * self.patches_per_image

    def get_patch_position(self, patch_idx):
        row = patch_idx // 4
        col = patch_idx % 4
        return row, col

    def get_vti_name(self, base_name, patch_idx):
        # Update this to match your actual naming convention!
        return f"{base_name}.vti"          # e.g. image001_0.vti ... image001_15.vti

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        patch_subidx = idx % self.patches_per_image

        img_name = self.image_files[img_idx]
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.root_dir, img_name)

        # ── Load RGB patch ───────────────────────────────────────
        image_full = Image.open(img_path).convert('RGB')
        image_full = np.array(image_full)   # Albumentations works with numpy

        row, col = self.get_patch_position(patch_subidx)
        left = col * self.patch_size
        upper = row * self.patch_size

        # Small random jitter only during training
        if self.is_train:
            offset = np.random.randint(-4, 5)
            left = max(0, min(160 - self.patch_size, left + offset))
            upper = max(0, min(160 - self.patch_size, upper + offset))

        image_crop = image_full[upper:upper+self.patch_size, left:left+self.patch_size]

        # ── Load director field ──────────────────────────────────
        vti_path = os.path.join(self.root_dir, self.get_vti_name(base, patch_subidx))
        if not os.path.exists(vti_path):
            raise FileNotFoundError(vti_path)

        mesh = pv.read(vti_path)
        array_name = 'n'

        if array_name in mesh.point_data:
            raw = mesh.point_data[array_name]
            is_point = True
        elif array_name in mesh.cell_data:
            raw = mesh.cell_data[array_name]
            is_point = False
        else:
            raise KeyError("No 'n' array found")

        dims = mesh.dimensions
        if is_point:
            resh = (dims[2], dims[1], dims[0], -1)
        else:
            resh = (max(dims[2]-1,1), max(dims[1]-1,1), max(dims[0]-1,1), -1)

        data = raw.reshape(resh)
        if data.shape[0] > 1:
            data = data[data.shape[0] // 2]   # mid slice
        else:
            data = data[0]

        # Assume director is (H, W, 3) → take first two components
        director_high = data[..., :2].astype(np.float32)   # (H, W, 2)

        # Downsample if needed (average pooling style)
        h = director_high.shape[0]
        if h != self.patch_size:
            factor = h // self.patch_size
            director_high = director_high.reshape(
                self.patch_size, factor, self.patch_size, factor, 2
            ).mean(axis=(1,3))   # crude but ok for starters

        # Normalize to unit vectors (very important!)
        norm = np.linalg.norm(director_high, axis=-1, keepdims=True)
        director_high = np.divide(director_high, norm, where=norm > 1e-8)

        # ── Apply Albumentations ────────────────────────────────
        if self.transform and self.is_train:
            # 1. Geometric to BOTH
            geo = geometric_transform(image=image_crop, director=director_high)
            image_crop_aug = geo['image']      # numpy uint8 most likely
            director_aug   = geo['director']

            # 2. Color + normalize pipeline
            color = color_transform(image=image_crop_aug)
            image = color['image']             # should be float32, but force anyway

            # ── FORCE FLOAT32 + correct range if needed ────────────────
            if image.dtype == torch.uint8:
                image = image.float() / 255.0                  # uint8 → [0,1] float32

            # Better: always force (safe even if already float)
            # image = image.float() / 255.0 if image.max() > 1.5 else image.float()

            # Vector correction
            director_aug = correct_director_after_aug(director_aug, geo)
            director = torch.from_numpy(director_aug).permute(2, 0, 1).float()

        else:
            # val/test path — IMPORTANT: add proper conversion here too!
            color = val_transform(image=image_crop)
            image = color['image']

            if image.dtype == torch.uint8:
                image = image.float() / 255.0

            director = torch.from_numpy(director_high).permute(2, 0, 1).float()

        return image, director

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
def director_loss(pred, target):
    # pred, target: (B, 2, H, W) unit vectors
    cos_sim = torch.sum(pred * target, dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angle_rad = torch.acos(cos_sim)                     # in [0, π]
    return angle_rad.mean()


def mae_deg(pred, targ):
    cos = (pred * targ).sum(dim=1).clamp(-0.9999, 0.9999)
    return torch.acos(cos).mean().item() * (180 / np.pi)


# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    for f in os.listdir("visualizations_5"):
        os.remove(os.path.join("visualizations_5", f)) if f.endswith(".png") else None
    
    folder = r"C:/Users/user/Downloads/FYP/Director_simulations/Test"   # ← update path
    
    train_ds = PatchDirectorDataset(
    root_dir=folder,
    patch_size=40,
    patches_per_image=16,
    is_train=True
    )

    img, director = train_ds[0]

    print("Image dtype:", img.dtype)          # should be torch.float32
    print("Image shape:", img.shape)          # should be [3,40,40]
    print("Image min/max:", img.min(), img.max())  # should be around -1 .. +1 after Normalize(0.5,0.5)
    print("Director dtype:", director.dtype)  # should be torch.float32
    
    val_ds = PatchDirectorDataset(
    root_dir=folder,
    patch_size=40,
    patches_per_image=16,       # usually better to use full 16
    is_train=False
    )
    
    test_ds = PatchDirectorDataset(
    root_dir=folder,
    patch_size=40,
    patches_per_image=16,       # usually better to use full 16
    is_train=False
    )

   # train_ds = PatchDirectorDataset(folder, is_train=True, transform=train_transform, patches_per_image=16)
  #  val_ds   = PatchDirectorDataset(folder, is_train=False, transform=val_transform, patches_per_image=16)
   # test_ds  = PatchDirectorDataset(folder, is_train=False, transform=val_transform, patches_per_image=16)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,   batch_size=32, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    model = DirectorUNet(in_channels=3, out_channels=2).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=300, eta_min=1e-6)

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

        scheduler.step()
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), "best_director_model_5.pth")
            print(f"  → saved better model  (val mae = {val_mae:.2f}°)")
            
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                imgs, tgts = next(iter(val_loader))
                preds = model(imgs.to(device))
                tgts = tgts.to(device)

                # Create save directory if needed
                save_dir = "visualizations_5"
                os.makedirs(save_dir, exist_ok=True)

                num_save = min(4, len(imgs))  # Save first 4 samples (adjust as needed)

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

    print(f"Saved {num_save} visualization images to {save_dir} for epoch {epoch+1}")

    print("Training finished.")
    
    # ── Final evaluation on TEST set ────────────────────────────────────────
    print("\n" + "="*70)
    print(" Final evaluation on held-out TEST set ".center(70, "="))
    print("="*70)

    model.load_state_dict(torch.load("best_director_model.pth"))
    model.eval()

    test_loss = test_mae = 0.0
    n = 0

    with torch.no_grad():
        for imgs, tgts in tqdm(test_loader, desc="Test evaluation"):
            imgs, tgts = imgs.to(device), tgts.to(device)
            pred = model(imgs)
            test_loss += director_loss(pred, tgts).item()
            test_mae  += mae_deg(pred, tgts)
            n += 1

    test_loss /= n
    test_mae  /= n

    print("TEST SET RESULTS (best model):")
    print(f" → loss:     {test_loss:.4f}")
    print(f" → MAE:      {test_mae:.2f}°")

    # ── Final visualizations on TEST set ────────────────────────────────────
    print("\nGenerating final test set visualizations...")
    model.load_state_dict(torch.load("best_director_model.pth"))
    model.eval()

    with torch.no_grad():
        num_show = 12   # how many you want
        indices = torch.randperm(len(test_ds))[:num_show].tolist()

        imgs_list = []
        tgts_list = []
        for idx in indices:
            img, tgt = test_ds[idx]
            imgs_list.append(img)
            tgts_list.append(tgt)

        imgs = torch.stack(imgs_list).to(device)
        tgts = torch.stack(tgts_list)
        preds = model(imgs)

        save_dir = os.path.join("visualizations_5", "final")
        os.makedirs(save_dir, exist_ok=True)

        for i in range(num_show):
            mae = mae_deg(preds[i:i+1], tgts[i:i+1])
            title = f"Final test sample {i+1}/{num_show}  (idx {indices[i]})  mae {mae:.1f}°"
            filename = f"test_{i+1:02d}_mae_{mae:05.1f}_idx{indices[i]:04d}.png"
            save_path = os.path.join(save_dir, filename)

            visualize_prediction(
                imgs[i].cpu(),
                preds[i].cpu(),
                tgts[i].cpu(),
                title=title,
                save_path=save_path
                )

    print(f"Saved {num_show} final test visualizations to {save_dir}")

if __name__ == '__main__':
    main()