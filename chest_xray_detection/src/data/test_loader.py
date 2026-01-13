import pandas as pd
import torch
from dataset import ChestXrayDataset
from torchvision import transforms
import os

def test_loader():
    csv_path = r"c:\Users\ASUS\Documents\Techtatva\cleaned_data\dataset_with_paths.csv"
    
    if not os.path.exists(csv_path):
        print("CSV not found. Please run path_mapper.py first.")
        return

    print("Loading CSV...")
    df = pd.read_csv(csv_path)
    
    # Filter valid paths
    df = df.dropna(subset=['path'])
    print(f"Valid images: {len(df)}")
    
    if len(df) == 0:
        print("No valid images found.")
        return

    # Basic transform
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    print("Creating Dataset...")
    dataset = ChestXrayDataset(df.head(10), transform=data_transform) # Test with 10 items
    
    print("Getting first item...")
    img, label = dataset[0]
    
    print(f"Image Shape: {img.shape}")
    print(f"Label: {label}")
    print("Dataloader test passed!")

if __name__ == "__main__":
    test_loader()
