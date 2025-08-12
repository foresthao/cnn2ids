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
    parser.add_argument('--balanced', action='store_true', help='Use balanced dataset')
    parser.add_argument('--val_ratio', type=float, default=0.3, help='Ratio of validation subset')
    parser.add_argument('--test_ratio', type=float, default=0.3, help='Ratio of test subset')
    return parser.parse_args()

def main():
    args = parse_args()

    # Configure logging module
    utils.mkdir(LOG_DIR)
    utils.mkdir(IMAGE_DIR)
    setup_logging(save_dir=LOG_DIR, log_config=LOG_CONFIG_PATH)

    logging.info(f'######## Training the CNN2ID model ########')
    logging.info(f'Arguments: {args}')

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

    # Our optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Epochs
    logging.info('start to train')
    history = train(model, criterion, optimizer, train_loader, valid_loader, args.epochs, DEVICE)

    training_loss = history['train']['loss']
    training_accuracy = history['train']['accuracy']
    train_output_true = history['train']['output_true']
    train_output_pred = history['train']['output_pred']

    validation_loss = history['valid']['loss']
    validation_accuracy = history['valid']['accuracy']
    valid_output_true = history['valid']['output_true']
    valid_output_pred = history['valid']['output_pred']

    logging.info('start to Plot loss vs iterations')
    fig = plt.figure(figsize=(12, 8))
    plt.plot(training_loss, label='train - loss')
    plt.plot(validation_loss, label='validation - loss')
    plt.title("Train and Validation Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="best")
    plt.savefig(os.path.join(IMAGE_DIR, 'Train_and_Validation_Loss.pdf'))
    plt.close(fig)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(training_accuracy, label='train - accuracy')
    plt.plot(validation_accuracy, label='validation - accuracy')
    plt.title("Train and Validation Accuracy")
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend(loc="best")
    plt.savefig(os.path.join(IMAGE_DIR, 'Train_and_Validation_Accuracy.pdf'))
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
