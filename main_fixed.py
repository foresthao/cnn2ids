from sklearn.metrics import classification_report, f1_score
import pandas as pd
import numpy as np
import argparse
import logging
import os
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torch.optim as optim
from models.CNN2ID_Fixed import CNN2ID_Fixed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from logger import setup_logging
from utils import (
    dataset_fixed,
    test,
    utils,
    visualisation,
)


LOG_CONFIG_PATH = os.path.join(os.path.abspath("."), "logger", "logger_config.json")
LOG_DIR   = os.path.join(os.path.abspath("."), "logs")
DATA_DIR  = os.path.join(os.path.abspath('.'), "data")
IMAGE_DIR = os.path.join(os.path.abspath("."), "images_fixed")
MODEL_DIR = os.path.join(os.path.abspath("."), "checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure that all operations are deterministic for reproducibility
utils.set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='CNN2ID Training - Fixed Version')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    return parser.parse_args()

def train_fixed(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim,
    scheduler,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    device: torch.device,
    patience: int = 10
):
    """Train the network with proper validation and early stopping."""

    model.to(device)
    
    history = {
        'train': {'loss': [], 'accuracy': []},
        'valid': {'loss': [], 'accuracy': []}
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        logging.info(f"Epoch {epoch}/{num_epochs}:")
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1)
            inputs = inputs.unsqueeze(1)  # Add channel dimension
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Log progress every 100 batches
            if batch_idx % 100 == 0:
                logging.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.view(-1)
                inputs = inputs.unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Store history
        history['train']['loss'].append(avg_train_loss)
        history['train']['accuracy'].append(train_acc)
        history['valid']['loss'].append(avg_val_loss)
        history['valid']['accuracy'].append(val_acc)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(avg_val_loss)
        
        logging.info(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                    f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            logging.info(f'New best validation loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            logging.info(f'No improvement. Patience: {patience_counter}/{patience}')
            
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch} epochs')
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        logging.info('Loaded best model state')
    
    return history

def main():
    args = parse_args()

    # Configure logging module
    utils.mkdir(LOG_DIR)
    utils.mkdir(IMAGE_DIR)
    setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)

    logging.info(f'######## Training the CNN2ID model - FIXED VERSION ########')
    logging.info(f'Arguments: {args}')
    logging.info(f'Device: {DEVICE}')

    logging.info("Loading fixed dataset...")

    # Get the datasets with fixed preprocessing
    train_loader, valid_loader, test_loader = dataset_fixed.load_data_fixed(
        data_path=DATA_DIR, 
        batch_size=args.batch_size
    )
    
    logging.info("Fixed dataset loaded!")

    # Get a sample to determine input size and number of classes
    sample_features, sample_labels = next(iter(train_loader))
    input_features = sample_features.shape[-1]
    
    # Determine number of classes dynamically
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.tolist())
    num_classes = len(set(all_labels))
    
    logging.info(f"Input features: {input_features}, Number of classes: {num_classes}")

    # Create improved model
    model = CNN2ID_Fixed(
        num_classes=num_classes, 
        input_features=input_features,
        dropout_rate=args.dropout
    )
    model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")

    # Dataset sizes
    train_size = len(train_loader.dataset)
    val_size = len(valid_loader.dataset)
    test_size = len(test_loader.dataset)
    
    print(f'# instances in training set: {train_size}')
    print(f'# instances in validation set: {val_size}')
    print(f'# instances in testing set: {test_size}')
    print(f'# epochs: {args.epochs}')
    print(f'# batch size: {args.batch_size}')
    print(f'# learning rate: {args.lr}')
    print(f'# weight decay: {args.weight_decay}')
    print(f'# dropout rate: {args.dropout}')

    # Loss function with class weights for imbalanced data
    criterion = nn.CrossEntropyLoss()

    # Optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

    # Training
    logging.info('Starting training...')
    history = train_fixed(
        model, criterion, optimizer, scheduler,
        train_loader, valid_loader, 
        args.epochs, DEVICE, args.patience
    )

    # Plot training curves
    logging.info('Plotting training curves...')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train']['loss'], label='Training Loss', linewidth=2)
    ax1.plot(history['valid']['loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train']['accuracy'], label='Training Accuracy', linewidth=2)
    ax2.plot(history['valid']['accuracy'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'training_curves.pdf'), dpi=300, bbox_inches='tight')
    plt.close()

    # Testing
    logging.info('Testing model...')
    test_history = test.test(model, criterion, test_loader, DEVICE)

    test_output_true = np.array(test_history['test']['output_true'])
    test_output_pred = np.array(test_history['test']['output_pred'])
    test_output_pred_prob = np.array(test_history['test']['output_pred_prob'])

    # Classification report
    class_labels = [str(i) for i in range(num_classes)]
    print("\\n=== Test Results ===")
    print("Classification Report:")
    print(classification_report(test_output_true, test_output_pred, target_names=class_labels))

    # Plot confusion matrix
    visualisation.plot_confusion_matrix(
        y_true=test_output_true,
        y_pred=test_output_pred,
        labels=class_labels,
        save=True,
        save_dir=IMAGE_DIR,
        filename="confusion_matrix_fixed.pdf"
    )

    # Plot ROC curve
    if num_classes > 2:
        y_test_one_hot = np.eye(num_classes)[test_output_true]
        visualisation.plot_roc_curve(
            y_test=y_test_one_hot,
            y_score=test_output_pred_prob,
            labels=class_labels,
            save=True,
            save_dir=IMAGE_DIR,
            filename="roc_curve_fixed.pdf"
        )

    # Save model
    model_path = os.path.join(MODEL_DIR, 'cnn2id_fixed_best.pth')
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logging.info(f'Model saved to {model_path}')

    logging.info('Training completed successfully!')


if __name__ == "__main__":
    main()