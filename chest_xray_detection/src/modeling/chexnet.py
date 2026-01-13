import torch
import torch.nn as nn
from torchvision import models

class CheXNet(nn.Module):
    def __init__(self, num_classes=14, pretrained=True):
        super(CheXNet, self).__init__()
        
        # Load DenseNet121
        # weights='DEFAULT' is the new way to specify pretrained, equivalent to pretrained=True
        try:
            self.densenet121 = models.densenet121(weights='DEFAULT' if pretrained else None)
        except:
            # Fallback for older torchvision versions
            self.densenet121 = models.densenet121(pretrained=pretrained)
        
        # Get input features of the classifier
        num_features = self.densenet121.classifier.in_features
        
        # Replace the classifier
        # We output logits (raw scores). Sigmoid will be applied during loss calculation (BCEWithLogitsLoss)
        # or during inference.
        self.densenet121.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.densenet121(x)

if __name__ == "__main__":
    # Test model
    model = CheXNet(pretrained=False)
    print("Model initialized successfully.")
    
    # Dummy input
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")
