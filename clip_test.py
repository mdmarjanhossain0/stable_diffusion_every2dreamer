import torch
import clip
from PIL import Image
import os
import numpy as np
from tqdm import tqdm

# ---------- Load CLIP model ----------
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# ---------- Example Data ----------
# Suppose you have a list of captions corresponding to your generated images
captions = [
    "A bottle of 17 cm height", "A bottle of 14 cm height",
    "A bottle of 17 cm height", "A bottle of 18 cm height", "A bottle of 19.2 cm height"
]

generated_folder = "./test_output"

# ---------- Compute CLIPScore ----------
clip_scores = []

image_files = sorted([f for f in os.listdir(generated_folder) if f.endswith((".png", ".jpg"))])

for i, img_file in enumerate(image_files):
    img_path = os.path.join(generated_folder, img_file)
    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    text = clip.tokenize([captions[i]]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (image_features @ text_features.T).item()
        clip_scores.append(similarity)

# ---------- Results ----------
clip_scores = np.array(clip_scores)
print(f"CLIPScore mean: {clip_scores.mean():.4f}, std: {clip_scores.std():.4f}")
