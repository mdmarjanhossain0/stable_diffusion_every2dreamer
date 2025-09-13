import torch
from torchvision.models import inception_v3
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm

# ---------- Helper Functions ----------
def get_inception_features(img_folder, device='cuda'):
    """Extract softmax features from InceptionV3 for all images in a folder."""
    model = inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    
    features = []
    for filename in tqdm(os.listdir(img_folder)):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(os.path.join(img_folder, filename)).convert('RGB')
            img = transform(img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                preds = model(img)
                preds = torch.nn.functional.softmax(preds, dim=1)
                features.append(preds.cpu().numpy())
    
    return np.concatenate(features, axis=0)

def inception_score(img_folder, device='cuda', splits=2):
    preds = get_inception_features(img_folder, device)
    
    N = preds.shape[0]
    split_scores = []
    
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            scores.append(entropy(part[i], py))
        split_scores.append(np.exp(np.mean(scores)))
    
    return float(np.mean(split_scores)), float(np.std(split_scores))

# ---------- Example Usage ----------
if __name__ == "__main__":
    folder = "./output"  # your generated images folder
    mean_is, std_is = inception_score(folder, device='cuda')
    print(f"Inception Score: {mean_is:.4f} ± {std_is:.4f}")
