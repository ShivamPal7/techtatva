import torch
import torch.nn as nn
import time
import copy
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device='cuda', save_dir='checkpoints'):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        
        # Use simple print if tqdm causes issues in some environments, but tqdm is preferred
        pbar = tqdm(dataloader, desc="Training")
        
        for inputs, labels in pbar:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            pbar.set_postfix({'loss': loss.item()})

        if self.scheduler:
            self.scheduler.step()
            
        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Validation"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        return epoch_loss

    def train(self, train_loader, val_loader, num_epochs=10):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            print('-' * 10)

            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            print(f'Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}')

            # Deep copy the model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'best_model.pth'))
                print("Saved best model.")
        
        print('Training complete.')
        print(f'Best Val Loss: {best_loss:.4f}')
        
        # Load best model weights
        self.model.load_state_dict(best_model_wts)
        return self.model
