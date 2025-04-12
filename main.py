# main.py
import os
import torch
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from models.escnn_dsf import ESCNN_DSF
from models.utils import PCamDataset
from trainers.trainer import DSFTrainer, get_dataloaders
from trainers.metrics import evaluate_model

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = get_dataloaders(
        args.x_train_path,
        args.y_train_path,
        args.x_val_path,
        args.y_val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create test loader if available
    test_loader = None
    if args.x_test_path and args.y_test_path:
        test_dataset = PCamDataset(args.x_test_path, args.y_test_path)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        )
    
    # Create model
    print("Creating model...")
    model = ESCNN_DSF(
        num_classes=2,
        growth_rate=args.growth_rate,
        block_config=args.block_config,
        group_order=args.group_order,
        dropout_rate=args.dropout_rate,
        compression=args.compression
    )
    model = model.to(device)
    
    # Create trainer
    trainer = DSFTrainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Train the model
    print("Starting training...")
    results = trainer.train(
        num_epochs=args.epochs,
        patience=args.patience
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['train_losses'], label='Train Loss')
    plt.plot(results['val_losses'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(results['val_aucs'], label='Val AUC')
    plt.axhline(y=results['best_auc'], color='r', linestyle='--', label=f'Best AUC: {results["best_auc"]:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('Validation AUC')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    
    # Evaluate on test set if available
    if test_loader is not None:
        print("Evaluating on test set...")
        test_metrics = evaluate_model(model, test_loader, device)
        
        # Save metrics
        pd.DataFrame([test_metrics]).to_csv(os.path.join(args.output_dir, 'test_metrics.csv'), index=False)
        print(f"Test metrics: {test_metrics}")
    
    # Save the final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_model.pth'))
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DSF-CNN with escnn on PCam')
    
    # Data paths
    parser.add_argument('--x_train_path', type=str, required=True, help='Path to training x.h5 file')
    parser.add_argument('--y_train_path', type=str, required=True, help='Path to training y.h5 file')
    parser.add_argument('--x_val_path', type=str, required=True, help='Path to validation x.h5 file')
    parser.add_argument('--y_val_path', type=str, required=True, help='Path to validation y.h5 file')
    parser.add_argument('--x_test_path', type=str, default=None, help='Path to test x.h5 file')
    parser.add_argument('--y_test_path', type=str, default=None, help='Path to test y.h5 file')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory')
    
    # Model hyperparameters
    parser.add_argument('--growth_rate', type=int, default=32, help='Growth rate in dense blocks')
    parser.add_argument('--block_config', type=int, nargs='+', default=[6, 12, 24, 16], 
                        help='Number of layers in each dense block')
    parser.add_argument('--group_order', type=int, default=8, help='Order of rotation group (8=C8)')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--compression', type=float, default=0.5, help='Compression in transition layers')
    
    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    main(args)