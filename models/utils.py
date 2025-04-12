# models/utils.py
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class PCamDataset(Dataset):
    """PCam dataset handler compatible with escnn models"""
    def __init__(self, x_path, y_path, transform=None):
        """
        Args:
            x_path: Path to the x.h5 file
            y_path: Path to the y.h5 file
            transform: Optional transform to apply
        """
        self.x_file = h5py.File(x_path, 'r')
        self.y_file = h5py.File(y_path, 'r')
        self.transform = transform
        self.x_data = self.x_file['x']
        self.y_data = self.y_file['y']
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        # H5 files store images in a different format than PyTorch expects
        # Convert from [H, W, C] to [C, H, W]
        img = self.x_data[idx].astype('float32')
        img = img.transpose(2, 0, 1) / 255.0  # Normalize to [0, 1]
        label = self.y_data[idx][0]
        
        # Convert to PyTorch tensors
        img_tensor = torch.from_numpy(img)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, label_tensor