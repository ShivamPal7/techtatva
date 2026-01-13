import torch
import time
from torchvision import models
import torch.nn as nn

def benchmark():
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Benchmarking on device: {device}")
    
    if device == 'cpu':
        print("WARNING: You are running on CPU. This will be slow.")
        print("If you have an RTX 4050, you need to install PyTorch with CUDA support.")
    
    # Model
    model = models.densenet121()
    model.classifier = nn.Linear(1024, 14)
    model.to(device)
    model.train()
    
    # Dummy Data (Batch size 16)
    batch_size = 16
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
    target = torch.randn(batch_size, 14).to(device) # Random floats for BCEWithLogits
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Warmup
    print("Warming up...")
    for _ in range(3):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    # Benchmark
    num_batches = 5
    print(f"Running {num_batches} batches...")
    start_time = time.time()
    
    for _ in range(num_batches):
        optimizer.zero_grad()
        output = model(input_tensor)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_batches
    
    # Extrapolation
    total_images = 55000
    images_per_epoch = total_images
    batches_per_epoch = total_images / batch_size
    
    time_per_epoch_sec = batches_per_epoch * avg_time_per_batch
    time_per_epoch_min = time_per_epoch_sec / 60
    time_per_epoch_hr = time_per_epoch_min / 60
    
    print("-" * 20)
    print(f"Average time per batch ({batch_size} imgs): {avg_time_per_batch:.4f} seconds")
    print(f"Estimated time per epoch ({total_images} images): {time_per_epoch_min:.2f} minutes ({time_per_epoch_hr:.2f} hours)")
    print(f"Total time for 10 epochs: {time_per_epoch_hr * 10:.2f} hours")
    print("-" * 20)

if __name__ == "__main__":
    benchmark()
