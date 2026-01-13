import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

class TensorGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hook for gradients
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # Forward pass
        output = self.model(x)
        
        if class_idx is None:
            # Default to class with highest score
            class_idx = torch.argmax(output, dim=1).item()
            
        # Zero grads
        self.model.zero_grad()
        
        # Backward pass for target class
        target = output[0, class_idx]
        target.backward()
        
        # Pool gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight activations by gradients
        activations = self.activations.detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        # Average the channels (heatmap)
        heatmap = torch.mean(activations, dim=1).squeeze()
        
        # ReLU
        heatmap = F.relu(heatmap)
        
        # Normalize
        if torch.max(heatmap) != 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.cpu().numpy()

def show_cam_on_image(img_path, heatmap, alpha=0.5, save_path=None):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = np.float32(img) / 255
    
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (224, 224))
    
    # Colorize
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Overlay
    cam = heatmap * alpha + img * (1 - alpha)
    cam = cam / np.max(cam)
    
    vis = np.uint8(255 * cam)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    
    plt.imshow(vis)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

# Helper for CheXNet
def generate_heatmap(model, img_tensor, original_img_path, target_layer, device='cuda', save_path='heatmap.png'):
    model.eval()
    grad_cam = TensorGradCAM(model, target_layer)
    
    img_tensor = img_tensor.to(device)
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
        
    heatmap = grad_cam(img_tensor)
    show_cam_on_image(original_img_path, heatmap, save_path=save_path)
    print(f"Heatmap saved to {save_path}")
