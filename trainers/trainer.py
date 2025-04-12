# trainers/trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

class DSFTrainer:
    def __init__(self, model, device, train_loader, val_loader, 
                 lr=0.001, weight_decay=1e-4):
        """
        Trainer for DSF-CNN model
        
        Args:
            model: The ESCNN_DSF model
            device: Device to use (cuda/cpu)
            train_loader: Training data loader
            val_loader: Validation data loader
            lr: Learning rate
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5, verbose=True
        )
        
        # Keep track of metrics
        self.train_losses = []
        self.val_losses = []
        self.val_aucs = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        train_loss = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            # Move data to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Calculate loss
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            
            # Update stats
            train_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss
        train_loss /= len(self.train_loader)
        self.train_losses.append(train_loss)
        
        return train_loss
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                
                # Calculate loss
                loss = self.criterion(output, target)
                val_loss += loss.item()
                
                # Store predictions and targets for AUC calculation
                probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_targets.extend(target.cpu().numpy())
        
        # Calculate average loss
        val_loss /= len(self.val_loader)
        self.val_losses.append(val_loss)
        
        # Calculate AUC
        val_auc = roc_auc_score(all_targets, all_probs)
        self.val_aucs.append(val_auc)
        
        # Update learning rate
        self.scheduler.step(val_auc)
        
        return val_loss, val_auc
    
    def train(self, num_epochs=100, patience=15):
        """
        Train the model for multiple epochs
        
        Args:
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
        """
        best_auc = 0
        no_improvement = 0
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_auc = self.validate()
            
            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")
            
            # Check for improvement
            if val_auc > best_auc:
                best_auc = val_auc
                no_improvement = 0
                
                # Save the best model
                torch.save(self.model.state_dict(), 'best_model.pth')
                print("Model saved as the new best!")
            else:
                no_improvement += 1
                print(f"No improvement for {no_improvement} epochs")
            
            # Early stopping
            if no_improvement >= patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
        
        # Load the best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_aucs': self.val_aucs,
            'best_auc': best_auc
        }

# trainers/trainer.py
def get_dataloaders(x_train_path, y_train_path, x_val_path, y_val_path, 
                    batch_size=32, num_workers=4):
    """Create dataloaders for training and validation"""
    
    # Define datasets
    train_dataset = PCamDataset(x_train_path, y_train_path)
    val_dataset = PCamDataset(x_val_path, y_val_path)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader