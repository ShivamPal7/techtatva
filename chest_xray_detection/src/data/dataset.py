import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import pandas as pd
import os

class ChestXrayDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        Args:
            df (pd.DataFrame): DataFrame containing 'path' and 'finding_labels'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.df = df
        self.transform = transform
        
        # Standard NIH Chest X-ray 14 classes
        self.classes = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
            'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
            'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
            'Pleural_Thickening', 'Hernia'
        ]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['path']
        labels_str = row['finding_labels']
        
        # Load Image
        # Convert to RGB (NIH images are Grayscale but models usually expect 3 channels)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy black image in case of error to avoid crashing
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)
            
        # Process Labels
        label_vec = np.zeros(len(self.classes), dtype=np.float32)
        
        if labels_str != "No Finding":
            for label in labels_str.split('|'):
                if label in self.class_to_idx:
                    label_vec[self.class_to_idx[label]] = 1.0
        
        return image, torch.tensor(label_vec)
