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


def save_prediction_as_vti(pred_tensor, save_path, num_layers=10,
                           total_twist_degrees=90, 
                           tilt_start_degrees=0, 
                           tilt_end_degrees=45):
    """
    Saves a (2, H, W) PyTorch tensor as a 3D VTK file.
    Applies both an in-plane twist and an out-of-plane tilt across the Z-axis.
    Includes a vertical Y-axis flip to correct VTK orientation.
    """
    # Convert PyTorch tensor to NumPy array
    pred_np = pred_tensor.detach().cpu().numpy()
    
    # 1. Flip vertically to correct the Y-axis mismatch
    pred_np = np.flip(pred_np, axis=1) 
    pred_hw = np.transpose(pred_np, (1, 2, 0)) # Shape: (H, W, 2)
    h, w = pred_hw.shape[:2]
    
    # Ensure vectors are perfectly normalized to length 1
    norm = np.linalg.norm(pred_hw, axis=-1, keepdims=True)
    pred_hw = np.divide(pred_hw, norm, out=np.zeros_like(pred_hw), where=norm!=0)
    
    volume_layers = []
    
    # Convert degrees to radians
    twist_rad_total = np.radians(total_twist_degrees)
    tilt_start_rad = np.radians(tilt_start_degrees)
    tilt_end_rad = np.radians(tilt_end_degrees)
    
    for z in range(num_layers):
        fraction = z / max(1, (num_layers - 1))
        
        # --- 1. IN-PLANE TWIST (Azimuthal) ---
        theta_twist = fraction * twist_rad_total
        cos_twist = np.cos(theta_twist)
        sin_twist = np.sin(theta_twist)
        
        x_rot = pred_hw[..., 0] * cos_twist - pred_hw[..., 1] * sin_twist
        y_rot = pred_hw[..., 0] * sin_twist + pred_hw[..., 1] * cos_twist
        
        # --- 2. OUT-OF-PLANE TILT (Polar) ---
        phi_tilt = tilt_start_rad + fraction * (tilt_end_rad - tilt_start_rad)
        cos_tilt = np.cos(phi_tilt)
        sin_tilt = np.sin(phi_tilt)
        
        x_final = x_rot * cos_tilt
        y_final = y_rot * cos_tilt
        z_final = np.full((h, w), sin_tilt, dtype=pred_hw.dtype)
        
        layer_3d = np.stack([x_final, y_final, z_final], axis=-1)
        volume_layers.append(layer_3d)

    # Stack all unique layers into a volume: Shape (Z, H, W, 3)
    pred_volume = np.stack(volume_layers, axis=0) 

    # Save to VTK
    grid = pv.ImageData()
    grid.dimensions = (w, h, num_layers)
    grid.point_data['n'] = pred_volume.reshape(-1, 3)
    
    structured_grid = grid.cast_to_structured_grid()
    structured_grid.save(save_path)

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
    
    
    theta = angle_pred
    I = np.sin(2 * theta) ** 2
    I = 0.2 + 0.8 * np.clip(I, 0, 1)

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
    axes[panel_idx].imshow(I, origin='lower')
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
            left = max(0, min(160 - self.patch_size, left + offset))
            upper = max(0, min(160 - self.patch_size, upper + offset))

        image_crop = image_full.crop((left, upper, left + self.patch_size, upper + self.patch_size))
        image_crop = image_crop.transpose(Image.FLIP_TOP_BOTTOM)   # match PyVista

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

        # ── Multi-slice sampling for 3D volumes ─────────────────────────────
        if data.shape[0] > 1:
            # For 3D-like samples (varying across z), sample multiple slices
            num_z = data.shape[0]
            if num_z <= 3:
                z_indices = list(range(num_z))                    # take all
            else:
                # Take 4 representative slices: top, 25%, 75%, bottom
                z_indices = [0, num_z//4, 3*num_z//4, num_z-1]
            
            # Randomly pick one slice for this patch (adds variety)
            z = random.choice(z_indices)
            data = data[z]
        else:
            data = data[0]

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
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=180),  # 180° is safe for non-polar directors
    transforms.RandomRotation(degrees=(-30, 30)),  # small continuous rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# ─── Model ──────────────────────────────────────────────────────────────────
class DirectorUNet(nn.Module):
    """ Simple wider 4-level U-Net (256-512-1024-2048) - No ConvNeXt, no extra complexity """
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

        # Wider channels
        self.enc1 = conv_block(in_channels, 256)
        self.enc2 = conv_block(256, 512)
        self.enc3 = conv_block(512, 1024)
        self.enc4 = conv_block(1024, 2048)

        self.bottleneck = conv_block(2048, 2048)

        # Decoder with original concat sizes adjusted for wider channels
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec4 = conv_block(3072, 1024)   # 2048 + 1024

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = conv_block(1536, 512)    # 1024 + 512

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = conv_block(768, 256)     # 512 + 256

        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)
        self.dec1 = conv_block(512, 256)     # 256 + 256

        self.final = nn.Conv2d(256, out_channels, kernel_size=1)

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
    # for f in os.listdir("predictions_mk4"):
    #     os.remove(os.path.join("predictions_mk2", f)) if f.endswith(".png") else None
    
    folder = r"C:/Users/user/Downloads/FYP/Director_simulations/nmtest"   # ← update path

    train_ds = PatchDirectorDataset(
    root_dir=os.path.join(folder, "train"),
    patches_per_image=16, random_crop=True, transform=train_transform
    )
    
    val_ds = PatchDirectorDataset(
    root_dir=os.path.join(folder, "val"),
    patches_per_image=16, random_crop=False, transform=val_transform   # ← use 16 now
    )
    
    test_ds = PatchDirectorDataset(
    root_dir=os.path.join(folder, "test"),
    patches_per_image=16, random_crop=False, transform=val_transform
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,   batch_size=32, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    model = DirectorUNet(in_channels=3, out_channels=2).to(device)
    model.load_state_dict(torch.load("best_director_model_mk4t.pth"))
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

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
            torch.save(model.state_dict(), "best_director_model_mk4t.pth")
            print(f"  → saved better model  (val mae = {val_mae:.2f}°)")
            
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                imgs, tgts = next(iter(val_loader))
                preds = model(imgs.to(device))
                tgts = tgts.to(device)

                # Create save directory if needed
                save_dir = "predictions_mk49"
                os.makedirs(save_dir, exist_ok=True)

                num_save = min(1, len(imgs))  # Save first 4 samples (adjust as needed)

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

    model.load_state_dict(torch.load("best_director_model_mk4t.pth"))
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
    print("\nSaving final TEST visualizations (best model)...")
    with torch.no_grad():
        test_iter = iter(test_loader)
        imgs, tgts = next(test_iter)
        preds = model(imgs.to(device))
        tgts = tgts.to(device)

        save_dir = os.path.join("predictions_mk49", "final")
        os.makedirs(save_dir, exist_ok=True)

        num_save = min(1, len(imgs))
        
        for i in range(num_save):
            mae = mae_deg(preds[i:i+1], tgts[i:i+1])
            title = f"FINAL best model — TEST sample {i}   mae {mae:.1f}°"
            filename = f"final_TEST_{i:02d}_mae_{mae:05.1f}.png"
            save_path = os.path.join(save_dir, filename)

            visualize_prediction(
                imgs[i], preds[i], tgts[i],
                title=title,
                save_path=save_path
            )
            
            # --- Save VTI File ---
            vti_filename = f"final_TEST_{i:02d}_mae_{mae:05.1f}.vtk"
            vti_save_path = os.path.join(save_dir, vti_filename)
            
            save_prediction_as_vti(
                preds[i], 
                vti_save_path,
                num_layers=10, 
                total_twist_degrees=90,  # Set your desired twist here
                tilt_start_degrees=0,    # Lays flat at the bottom
                tilt_end_degrees=45      # Tilts out of plane at the top
            )

    print(f"Saved final test visualizations to: {save_dir}")

if __name__ == '__main__':
    main()
