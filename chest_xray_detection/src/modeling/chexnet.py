import torch
import torch.nn as nn
from torchvision import models

import torch.nn.functional as F

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
        
        # FIX for Grad-CAM: Disable in-place ReLU
        # in-place operations can cause "RuntimeError: Output 0 of BackwardHookFunctionBackward is a view..."
        for module in self.densenet121.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    def forward(self, x):
        # Manually run DenseNet121 forward pass to avoid functional inplace ReLU
        # Original: return self.densenet121(x)
        features = self.densenet121.features(x)
        out = F.relu(features, inplace=False)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.densenet121.classifier(out)
        return out

if __name__ == "__main__":
    # Test model
    model = CheXNet(pretrained=False)
    print("Model initialized successfully.")
    
    # Dummy input
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(f"Output shape: {y.shape}")
