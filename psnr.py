import os
import numpy as np
from PIL import Image

def psnr(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).convert('RGB'))
    img2 = np.array(Image.open(img2_path).convert('RGB'))
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # identical images
    
    PIXEL_MAX = 255.0
    psnr_value = 10 * np.log10((PIXEL_MAX ** 2) / mse)
    return psnr_value

# Folders
real_folder = './test_real'
output_folder = './test_output'

# Get sorted list of images
real_images = sorted([f for f in os.listdir(real_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
output_images = sorted([f for f in os.listdir(output_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

psnr_values = []

# Loop through image pairs
for real_img, output_img in zip(real_images, output_images):
    real_path = os.path.join(real_folder, real_img)
    output_path = os.path.join(output_folder, output_img)
    
    psnr_val = psnr(real_path, output_path)
    psnr_values.append(psnr_val)
    print(f"{real_img} <-> {output_img} : PSNR = {psnr_val:.2f} dB")

# Average PSNR
if psnr_values:
    avg_psnr = sum(psnr_values) / len(psnr_values)
    print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
else:
    print("No images found to compare.")
