import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from data.dataset import ChestXrayDataset
from modeling.chexnet import CheXNet
from training.trainer import Trainer
from training.evaluate import compute_auc

def main():
    parser = argparse.ArgumentParser(description='Train Chest X-Ray Model')
    parser.add_argument('--csv_path', type=str, default='cleaned_data/dataset_with_paths.csv', help='Path to dataset CSV')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--no_pretrained', action='store_true', help='Disable pretrained weights')
    args = parser.parse_args()

    print(f"Using device: {args.device}")

    # Load Data
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found at {args.csv_path}")
        return

    df = pd.read_csv(args.csv_path)
    # Filter valid paths just in case
    df = df.dropna(subset=['path'])
    
    # Use a smaller subset for quick testing if needed, or full dataset
    # For demonstration/development, we might want to restrict if dataset is huge and no GPU
    if args.device == 'cpu':
        print("Running on CPU: Limiting dataset size for speed.")
        df = df.sample(n=min(len(df), 50), random_state=42) # Limit to 50 images on CPU for TEST

    print(f"Total samples: {len(df)}")

    # Transforms
    # ImageNet normalization
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    # Dataset & Split
    # Split properly with DataFrame
    # If dataset is too small (e.g. 1), split might fail.
    if len(df) < 5:
        print("Dataset too small for train/val split. Using same data for both.")
        train_df = df
        val_df = df
    else:
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)
    
    train_dataset = ChestXrayDataset(train_df, transform=train_transform)
    val_dataset = ChestXrayDataset(val_df, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0) # num_workers=0 for Windows compatibility
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = CheXNet(num_classes=14, pretrained=not args.no_pretrained)
    
    # Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Train
    trainer = Trainer(model, criterion, optimizer, scheduler, device=args.device, save_dir='checkpoints')
    model = trainer.train(train_loader, val_loader, num_epochs=args.epochs)

    # Evaluate
    compute_auc(model, val_loader, device=args.device)

if __name__ == "__main__":
    main()
