import torch
from sklearn.metrics import roc_auc_score
import numpy as np

def compute_auc(model, dataloader, device='cuda', num_classes=14):
    model.eval()
    y_true = []
    y_pred = []
    
    # Store predictions
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            # Apply sigmoid because model outputs logits
            probs = torch.sigmoid(outputs)
            
            y_true.append(labels.cpu().numpy())
            y_pred.append(probs.cpu().numpy())
    
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    
    aucs = []
    classes = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
        'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 
        'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 
        'Pleural_Thickening', 'Hernia'
    ]
    
    print("\nAUC Scores per Class:")
    for i in range(num_classes):
        try:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(score)
            print(f"{classes[i]}: {score:.4f}")
        except ValueError:
            # Can happen if a class is not present in the validation set
            print(f"{classes[i]}: N/A (No positive samples)")
            
    mean_auc = np.mean([x for x in aucs if x is not None])
    print(f"\nMean AUC: {mean_auc:.4f}")
    return mean_auc
