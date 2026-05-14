# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:31:10 2026
@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from matplotlib.colors import hsv_to_rgb, Normalize, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import pyvista as pv


def save_stitched_director_as_vtk_3d(pred_np, save_path, num_layers=20, 
                                     total_twist_degrees=90, 
                                     tilt_start_degrees=0, 
                                     tilt_end_degrees=180):
    """
    Saves a (2, H, W) NumPy array as a 3D VTK file.
    Applies both an in-plane twist and an out-of-plane tilt across the Z-axis.
    """
    # 1. Flip vertically to correct the Y-axis mismatch for VTK
    pred_np = np.flip(pred_np, axis=1) 
    pred_hw = np.transpose(pred_np, (1, 2, 0)) # Shape: (H, W, 2)
    h, w = pred_hw.shape[:2]
    
    # Optional: Ensure the 2D input vectors are perfectly normalized to length 1
    # This prevents vectors from shrinking/growing unexpectedly when we tilt them
    norm = np.linalg.norm(pred_hw, axis=-1, keepdims=True)
    pred_hw = np.divide(pred_hw, norm, out=np.zeros_like(pred_hw), where=norm!=0)
    
    volume_layers = []
    
    # Convert degrees to radians for numpy
    twist_rad_total = np.radians(total_twist_degrees)
    tilt_start_rad = np.radians(tilt_start_degrees)
    tilt_end_rad = np.radians(tilt_end_degrees)
    
    for z in range(num_layers):
        # Calculate how far along the Z-axis we are (0.0 at bottom, 1.0 at top)
        fraction = z / max(1, (num_layers - 1))
        
        # --- 1. IN-PLANE TWIST (Azimuthal) ---
        theta_twist = fraction * twist_rad_total
        cos_twist = np.cos(theta_twist)
        sin_twist = np.sin(theta_twist)
        
        # Apply 2D rotation matrix
        x_rot = pred_hw[..., 0] * cos_twist - pred_hw[..., 1] * sin_twist
        y_rot = pred_hw[..., 0] * sin_twist + pred_hw[..., 1] * cos_twist
        
        # --- 2. OUT-OF-PLANE TILT (Polar) ---
        # Calculate the tilt angle for this specific layer
        phi_tilt = tilt_start_rad + fraction * (tilt_end_rad - tilt_start_rad)
        cos_tilt = np.cos(phi_tilt)
        sin_tilt = np.sin(phi_tilt)
        
        # Scale the XY components by cos(tilt) so the total vector length stays 1
        x_final = x_rot * cos_tilt
        y_final = y_rot * cos_tilt
        
        # The Z component becomes sin(tilt)
        # We use np.full to create a Z-array of the same shape (H, W)
        z_final = np.full((h, w), sin_tilt, dtype=pred_hw.dtype)
        
        # Stack into a 3D vector (H, W, 3)
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

# ── Your DirectorUNet model (unchanged) ──
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

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = conv_block(1536, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = conv_block(768, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = conv_block(384, 128)

        self.up1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)
        self.dec1 = conv_block(256, 128)

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


# ─── Device & Paths ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MODEL_PATH = "best_director_model_x.pth"
IMAGE_PATH = "5.png"

# Output directory for saved director fields
SAVE_DIR = "director_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Load model ─────────────────────────────────────────────────────────────
model = DirectorUNet(in_channels=3, out_channels=2).to(device)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded model from: {MODEL_PATH}")

# ─── Load & prepare image ───────────────────────────────────────────────────
img_pil = Image.open(IMAGE_PATH).convert("RGB")
w, h = img_pil.size
print(f"Input image size: {w} × {h}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# ─── Tiling parameters ──────────────────────────────────────────────────────
PATCH_SIZE = 18
STRIDE     = 3

# ─── Extract overlapping patches ────────────────────────────────────────────
patches = []
positions = []

y_start = range(0, max(h - PATCH_SIZE + 1, 1), STRIDE)
x_start = range(0, max(w - PATCH_SIZE + 1, 1), STRIDE)

for y in y_start:
    for x in x_start:
        patch = img_pil.crop((x, y, x + PATCH_SIZE, y + PATCH_SIZE))
        patch_t = transform(patch).unsqueeze(0).to(device)
        patches.append(patch_t)
        positions.append((x, y))

print(f"Number of patches to predict: {len(patches)}")

# ─── Inference ──────────────────────────────────────────────────────────────
preds = []
with torch.no_grad():
    for patch_t in tqdm(patches, desc="Predicting patches"):
        out = model(patch_t)
        preds.append(out.cpu().squeeze(0))

print("Inference complete.")

# ─── Stitch with averaging ──────────────────────────────────────────────────
director = np.zeros((2, h, w), dtype=np.float32)
count    = np.zeros((h, w), dtype=np.float32)

for (x, y), pred in zip(positions, preds):
    pred_np = pred.numpy()
    director[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE] += pred_np
    count[  y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1

count = np.maximum(count, 1e-6)
director /= count[None, :, :]

# Re-normalize vectors to unit length
norm = np.sqrt(np.sum(director**2, axis=0, keepdims=True))
director /= np.maximum(norm, 1e-6)

print("Stitching complete.")

# ─── Compute angle & HSV visualization ──────────────────────────────────────
angle = np.arctan2(director[1], director[0]) % np.pi
hue   = angle / np.pi
hsv   = np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1)
rgb   = hsv_to_rgb(hsv)

# Light smoothing
angle_smoothed = gaussian_filter(angle, sigma=1.5)
hue_smoothed   = angle_smoothed / np.pi
rgb_smoothed   = hsv_to_rgb(np.stack([hue_smoothed, np.ones_like(hue_smoothed), np.ones_like(hue_smoothed)], -1))

# ─── Fake crossed-polarizer texture ─────────────────────────────────────────
theta = angle_smoothed
I = 0.2 + 0.8 * np.sin(2 * theta) ** 2
I = np.clip(I, 0, 1)

# ─── Save full stitched director to VTK ─────────────────────────────────────
save_dir = os.path.join("visual")
os.makedirs(save_dir, exist_ok=True)
vtk_filename = "4.vtk"  # Ensure the extension matches what you want
vtk_save_path = os.path.join(save_dir, vtk_filename)

# Pass the stitched `director` array here!
save_stitched_director_as_vtk_3d(director, vtk_save_path, num_layers=10)

print(f"Saved full stitched director to: {vtk_save_path}")

# ─── Plot results (optional) ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 6))

axes[0].imshow(img_pil)
axes[0].set_title("Input Image")
axes[0].axis('off')

axes[1].imshow(rgb)
axes[1].set_title("Predicted Director (HSV) - raw")
axes[1].axis('off')

axes[2].imshow(rgb_smoothed)
axes[2].set_title("Predicted Director (HSV) - smoothed")
axes[2].axis('off')

axes[3].imshow(I, cmap='gray')
axes[3].set_title("Synthesized crossed-polarizer texture")
axes[3].axis('off')

plt.tight_layout()
plt.show()