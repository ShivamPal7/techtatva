
import sys
import os
import torch
import torch.nn as nn

# Add src to path
sys.path.append(os.path.abspath("chest_xray_detection/src"))

from modeling.chexnet import CheXNet

def verify():
    print("Initializing CheXNet...")
    model = CheXNet(pretrained=False)
    
    print("\nChecking ReLU layers...")
    inplace_count = 0
    non_inplace_count = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU):
            if module.inplace:
                print(f"[FAIL] Layer {name} has inplace=True")
                inplace_count += 1
            else:
                print(f"[PASS] Layer {name} has inplace=False")
                non_inplace_count += 1
                
    print(f"\nSummary: {non_inplace_count} ReLUs fixed, {inplace_count} ReLUs still inplace.")
    
    if inplace_count > 0:
        print("VERIFICATION FAILED: Some layers are still inplace=True.")
    else:
        print("VERIFICATION PASSED: All ReLU layers have inplace=False.")

if __name__ == "__main__":
    verify()
