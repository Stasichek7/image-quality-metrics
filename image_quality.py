import numpy as np
import matplotlib.pyplot as plt
import os

output_dir = "student_output"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------------------------
# 1. SETUP THE IMAGES
# ------------------------------------------------------------------------------
size = 200
x = np.linspace(-10, 10, size)

# Original image
original = np.sin(x[:, None]**2 + x[None, :]**2) / (x[:, None]**2 + x[None, :]**2 + 1)
original = np.interp(original, (original.min(), original.max()), (0, 255)).astype(np.float64)

# Compressed (noisy) image
np.random.seed(42)
compressed = original + np.random.normal(0, 15, original.shape)
compressed = np.clip(compressed, 0, 255)

# ------------------------------------------------------------------------------
# PSNR CALCULATION
# ------------------------------------------------------------------------------

# Step 1: MSE
mse_value = np.mean((original - compressed) ** 2)

# Step 2: MAX value
max_pixel_value = 255.0

# Step 3: PSNR
psnr_value = 10 * np.log10((max_pixel_value ** 2) / mse_value)

# ------------------------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------------------------

print(f"CALCULATED MSE:  {mse_value:.2f}")
print(f"CALCULATED PSNR: {psnr_value:.2f} dB")

# ------------------------------------------------------------------------------
# VISUALIZATION
# ------------------------------------------------------------------------------

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle(f"Quality Report: PSNR = {psnr_value:.2f} dB", fontsize=16)

axs[0].imshow(original, cmap='gray', vmin=0, vmax=255)
axs[0].set_title("Original Frame")
axs[0].axis('off')

axs[1].imshow(compressed, cmap='gray', vmin=0, vmax=255)
axs[1].set_title("Compressed Frame")
axs[1].axis('off')

plt.tight_layout()
plt.show()