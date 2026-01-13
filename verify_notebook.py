import os
import sys
import torch
import pandas as pd
import numpy as np
from torchvision import transforms

# Correct path appending: add 'src' directly via full path to 'chest_xray_detection/src'
sys.path.append(os.path.abspath('chest_xray_detection/src'))

try:
    from data.dataset import ChestXrayDataset
    from modeling.chexnet import CheXNet
    from interpretability.gradcam import generate_heatmap
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

# Configuration parameters to match notebook
CSV_PATH = 'cleaned_data/dataset_with_paths.csv'
IMG_SIZE = 224
BATCH_SIZE = 2

def verify_logic():
    print("Verifying Data Loading...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found.")
        return

    # Define Transforms (Same as notebook/training)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_ops = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['path'])
    # Reduce size for verification speed
    df = df.head(10)
    
    dataset = ChestXrayDataset(df, transform=transform_ops)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    images, labels = next(iter(dataloader))
    print(f"Data loaded. Batch shape: {images.shape}")

    print("Verifying Model Initialization...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(dataset.classes)
    model = CheXNet(num_classes=num_classes, pretrained=False).to(device)
    print("Model initialized.")

    print("Verifying Forward Pass...")
    model.eval()
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        probs = torch.sigmoid(outputs)
    print(f"Forward pass successful. Output shape: {outputs.shape}")

    print("Verifying Grad-CAM setup...")
    # FIX: Use densenet121 instead of densenet
    target_layer = model.densenet121.features[-1]
    
    # Save a temp image for Grad-CAM
    from torchvision.utils import save_image
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    inv_img = inv_normalize(images[0].cpu())
    inv_img = torch.clamp(inv_img, 0, 1)
    temp_img_path = 'temp_verify_gradcam.png'
    save_image(inv_img, temp_img_path)

    try:
        generate_heatmap(model, images[0], temp_img_path, target_layer, device=device, save_path='test_heatmap.png')
        if os.path.exists('test_heatmap.png'):
            print("Grad-CAM generated successfully.")
            # cleanup
            if os.path.exists(temp_img_path): os.remove(temp_img_path)
            if os.path.exists('test_heatmap.png'): os.remove('test_heatmap.png')
    except Exception as e:
        print(f"Grad-CAM verification warning: {e}")
        import traceback
        traceback.print_exc()

    print("Verification complete.")

if __name__ == "__main__":
    verify_logic()
