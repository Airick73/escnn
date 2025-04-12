# trainers/metrics.py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, test_loader, device):
    """
    Evaluate model performance on a test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to use
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_targets = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            
            # Get probabilities and predictions
            probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
            preds = output.argmax(dim=1).cpu().numpy()
            
            # Store for later
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(target.cpu().numpy())
    
    # Calculate metrics
    auc = roc_auc_score(all_targets, all_probs)
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    
    return {
        'auc': auc,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }