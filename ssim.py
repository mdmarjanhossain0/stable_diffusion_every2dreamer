import os
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1_path, img2_path):
    img1 = np.array(Image.open(img1_path).convert('RGB'))
    img2 = np.array(Image.open(img2_path).convert('RGB'))
    
    # Convert to grayscale for SSIM calculation
    img1_gray = np.mean(img1, axis=2)
    img2_gray = np.mean(img2, axis=2)
    
    ssim_value = ssim(img1_gray, img2_gray, data_range=img2_gray.max() - img2_gray.min())
    return ssim_value

# Folders
real_folder = './test_real'
output_folder = './test_output'

# Get sorted list of images
real_images = sorted([f for f in os.listdir(real_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
output_images = sorted([f for f in os.listdir(output_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

ssim_values = []

# Loop through image pairs
for real_img, output_img in zip(real_images, output_images):
    real_path = os.path.join(real_folder, real_img)
    output_path = os.path.join(output_folder, output_img)
    
    ssim_val = calculate_ssim(real_path, output_path)
    ssim_values.append(ssim_val)
    print(f"{real_img} <-> {output_img} : SSIM = {ssim_val:.4f}")

# Average SSIM
if ssim_values:
    avg_ssim = sum(ssim_values) / len(ssim_values)
    print(f"\nAverage SSIM: {avg_ssim:.4f}")
else:
    print("No images found to compare.")
