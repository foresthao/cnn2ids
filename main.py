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
from models import CNN2ID
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from logger import setup_logging
from utils import (
    dataset,
    models,
    test,
    train,
    utils,
    visualisation,
)


LOG_CONFIG_PATH = os.path.join(os.path.abspath("."), "logger", "logger_config.json")
LOG_DIR   = os.path.join(os.path.abspath("."), "logs")
DATA_DIR  = os.path.join(os.path.abspath('.'), "data")
IMAGE_DIR = os.path.join(os.path.abspath("."), "images")
MODEL_DIR = os.path.join(os.path.abspath("."), "checkpoints")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Ensure that all operations are deterministic for reproducibility, even on GPU (if used)
utils.set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser(description='CNN2ID Training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization)')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--balanced', action='store_true', help='Use balanced dataset')
    parser.add_argument('--val_ratio', type=float, default=0.3, help='Ratio of validation subset')
    parser.add_argument('--test_ratio', type=float, default=0.3, help='Ratio of test subset')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--use_scheduler', action='store_true', help='Use learning rate scheduler')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping value')
    return parser.parse_args()

def train_improved(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim,
    scheduler,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    device: torch.device,
    patience: int = 10,
    grad_clip: float = 1.0
):
    """Train the network with improved stability features."""

    model.to(device)
    
    history = {
        'train': {
            'total': 0,
            'loss': [],
            'accuracy': [],
            'output_pred': [],
            'output_true': []
        },
        'valid': {
            'total': 0,
            'loss': [],
            'accuracy': [],
            'output_pred': [],
            'output_true': []
        }
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, num_epochs + 1):
        
        ########################################
        ##             TRAIN LOOP             ##
        ########################################
        model.train()

        train_loss = 0.0
        train_steps = 0
        train_total = 0
        train_correct = 0

        train_output_pred = []
        train_output_true = []

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            
            optimizer.step()

            train_loss += loss.cpu().item()
            train_steps += 1

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_output_pred += outputs.argmax(1).cpu().tolist()
            train_output_true += labels.cpu().tolist()
            
            # Log progress every 100 batches
            if batch_idx % 100 == 0:
                logging.info(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

        ########################################
        ##             VALID LOOP             ##
        ########################################
        model.eval()

        val_loss = 0.0
        val_steps = 0
        val_total = 0
        val_correct = 0

        val_output_pred = []
        val_output_true = []

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.view(-1)
                inputs = inputs.unsqueeze(1)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.cpu().item()
                val_steps += 1

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                val_output_pred += outputs.argmax(1).cpu().tolist()
                val_output_true += labels.cpu().tolist()

        # Calculate averages
        avg_train_loss = train_loss / train_steps
        avg_val_loss = val_loss / val_steps
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        # Store history
        history['train']['total'] = train_total
        history['train']['loss'].append(avg_train_loss)
        history['train']['accuracy'].append(train_acc)
        history['train']['output_pred'] = train_output_pred
        history['train']['output_true'] = train_output_true

        history['valid']['total'] = val_total
        history['valid']['loss'].append(avg_val_loss)
        history['valid']['accuracy'].append(val_acc)
        history['valid']['output_pred'] = val_output_pred
        history['valid']['output_true'] = val_output_true
        
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
    
    logging.info(f"Finished Training")
    return history

def main():
    args = parse_args()

    # Configure logging module
    utils.mkdir(LOG_DIR)
    utils.mkdir(IMAGE_DIR)
    setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)

    logging.info(f'######## Training the CNN2ID model ########')
    logging.info(f'Arguments: {args}')
    logging.info(f'Using device: {DEVICE}')

    logging.info("Loading dataset...")

    # Get the datasets
    train_data, val_data, test_data = dataset.get_dataset(data_path=DATA_DIR, balanced=args.balanced)
    logging.info("Dataset loaded!")

    # Determine number of classes dynamically
    num_classes = int(train_data.labels.iloc[:, 0].nunique())
    class_labels = [str(i) for i in range(num_classes)]

    # Create model after knowing num_classes
    model = CNN2ID(num_classes=num_classes)
    model.to(DEVICE)

    #选择一部分作为呀征集采样
    utils.set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #选择一部分作为呀征集采样
    # Use subset sizes relative to respective datasets and guard for small sets
    val_subset_size = max(1, int(args.val_ratio * len(val_data)))
    test_subset_size = max(1, int(args.test_ratio * len(test_data)))
    val_subset_indices = np.random.choice(len(val_data), size=val_subset_size, replace=False)
    test_subset_indices = np.random.choice(len(test_data), size=test_subset_size, replace=False)

    val_subset = Subset(val_data, val_subset_indices)
    test_subset = Subset(test_data, test_subset_indices)
    # How many instances have we got?
    print('# instances in training set: ', len(train_data))
    print('# instances in validation set: ', len(val_data))
    print('# instances in testing set: ', len(test_data))
    print('# instances in validation subset: ', len(val_subset))
    print('# instances in testing subset: ', len(test_subset))
    print(f'# epochs: {args.epochs}')
    print(f'# batch size: {args.batch_size}')

    # Create the dataloaders - for training, validation and testing
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=val_subset, batch_size=args.batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    # Out loss function
    criterion = nn.CrossEntropyLoss()

    # Our optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
    
    # Print training parameters
    print(f'# learning rate: {args.lr}')
    print(f'# weight decay: {args.weight_decay}')
    print(f'# gradient clipping: {args.grad_clip}')
    print(f'# early stopping patience: {args.patience}')
    print(f'# use scheduler: {args.use_scheduler}')
    
    # Epochs
    logging.info('start to train')
    history = train_improved(model, criterion, optimizer, scheduler, train_loader, valid_loader, args.epochs, DEVICE, args.patience, args.grad_clip)

    training_loss = history['train']['loss']
    training_accuracy = history['train']['accuracy']
    train_output_true = history['train']['output_true']
    train_output_pred = history['train']['output_pred']

    validation_loss = history['valid']['loss']
    validation_accuracy = history['valid']['accuracy']
    valid_output_true = history['valid']['output_true']
    valid_output_pred = history['valid']['output_pred']

    logging.info('start to Plot loss vs iterations')
    # Set font sizes for better readability
    plt.rcParams.update({'font.size': 14})
    
    fig = plt.figure(figsize=(14, 10))
    plt.plot(training_loss, label='train - loss', linewidth=2.5)
    plt.plot(validation_loss, label='validation - loss', linewidth=2.5)
    plt.title("Train and Validation Loss", fontsize=20, fontweight='bold')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(loc="best", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'Train_and_Validation_Loss.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure(figsize=(14, 10))
    plt.plot(training_accuracy, label='train - accuracy', linewidth=2.5)
    plt.plot(validation_accuracy, label='validation - accuracy', linewidth=2.5)
    plt.title("Train and Validation Accuracy", fontsize=20, fontweight='bold')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.ylim(0, 1)
    plt.legend(loc="best", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, 'Train_and_Validation_Accuracy.pdf'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    logging.info('plot confusion matrix')
    visualisation.plot_confusion_matrix(y_true=train_output_true,
                                    y_pred=train_output_pred,
                                    labels=class_labels,
                                    save=True,
                                    save_dir=IMAGE_DIR,
                                    filename="cnn2ids_train_confusion_matrix.pdf")
    print("Training Set -- Classification Report", end="\n\n")
    print(classification_report(train_output_true, train_output_pred))

    visualisation.plot_confusion_matrix(y_true=valid_output_true,
                                    y_pred=valid_output_pred,
                                    labels=class_labels,
                                    save=True,
                                    save_dir=IMAGE_DIR,
                                    filename="cnn2ids_valid_confusion_matrix.pdf")
    print("Validation Set -- Classification Report", end="\n\n")
    print(classification_report(valid_output_true, valid_output_pred))

    logging.info('test it')
    test_history = test(model, criterion, test_loader, DEVICE)

    test_output_true = np.array(test_history['test']['output_true'])
    test_output_pred = np.array(test_history['test']['output_pred'])
    test_output_pred_prob = np.array(test_history['test']['output_pred_prob'])

    visualisation.plot_confusion_matrix(y_true=test_output_true,
                                    y_pred=test_output_pred,
                                    labels=class_labels,
                                    save=True,
                                    save_dir=IMAGE_DIR,
                                    filename="cnn2ids_test_confusion_matrix.pdf")
    print("Testing Set -- Classification Report", end="\n\n")
    print(classification_report(test_output_true, test_output_pred))
    
    logging.info('plot ROC')
    # Prepare one-hot y_test and probability scores for ROC
    y_test_one_hot = np.eye(num_classes)[test_output_true]
    y_score = test_output_pred_prob
    visualisation.plot_roc_curve(y_test=y_test_one_hot,
                             y_score=y_score,
                             labels=class_labels,
                             save=True,
                             save_dir=IMAGE_DIR,
                             filename="cnn2ids_roc_curve.pdf")


if __name__ == "__main__":
    main()
