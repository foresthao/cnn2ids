# cnn2id-based NIDS on the CICIDS2017 and CICIDS Dataset

### Dataset from above

https://www.unb.ca/cic/datasets/ids-2017.html
https://www.unb.ca/cic/datasets/ids-2018.html

## Introduction

This is the source code for "An Intrusion Detection System via novel Convolutional Neural Network"

In this repository, we propose  CNN2ID (Convolutional Neural Network for Intrusion Detection), an intrusion detection algorithm based on the CNN architecture. In extensive experiments, we achieved a high accuracy of up to 99% with CNN2ID, while requiring reasonable training time. CNN2ID achieves state-of-the-art detection performance and outperforms traditional methods in intrusion detection. 

## Installation

Use conda environment

# This file may be used to create an environment using:

$ conda create --name <env> --file <this file>


<this file> is requirements1.txt

## Requirements

All the experiments were conducted using a 64-bit Intel(R) Core(TM) i7 CPU with 32GB RAM in Linux environment. The models have been implemented in Python v3.9.18 using the PyTorch.

## Data

We use the 'CIC-IDS-2017\MachineLearningCSV' and 'CSE-CIC-IDS2018\Processed Traffic Data for ML Algorithms' download from the above link.
for example:
Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv\\Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv\\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv

## How to use

### Quick Start

1. Install Conda env like the requirements.
2. Download dataset and use csv file, copy the csv file into the data/raw folder.
3. Run preprocessing to generate processed data:
   ```bash
   python preprocessing/cicids2017.py
   ```
4. Run training with default parameters:
   ```bash
   python main.py
   ```

### Command Line Options

The training script supports various command line arguments:

```bash
python main.py [OPTIONS]
```

#### Basic Training Parameters
- `--epochs EPOCHS`: Number of training epochs (default: 5)
- `--batch_size BATCH_SIZE`: Batch size for training (default: 128)
- `--lr LR`: Learning rate (default: 0.001)
- `--balanced`: Use balanced dataset (default: False)
- `--val_ratio VAL_RATIO`: Ratio of validation subset (default: 0.3)
- `--test_ratio TEST_RATIO`: Ratio of test subset (default: 0.3)

#### Training Stability Parameters (New)
- `--weight_decay WEIGHT_DECAY`: Weight decay (L2 regularization) coefficient (default: 1e-4)
- `--dropout DROPOUT`: Dropout rate for regularization (default: 0.3)
- `--patience PATIENCE`: Early stopping patience - stops training if validation loss doesn't improve (default: 10)
- `--use_scheduler`: Enable learning rate scheduler (ReduceLROnPlateau) (default: False)
- `--grad_clip GRAD_CLIP`: Gradient clipping value to prevent exploding gradients (default: 1.0)

### Training Stability Improvements

The latest version includes several improvements for training stability:

#### 1. Weight Decay (L2 Regularization)
- Prevents overfitting by penalizing large weights
- Improves generalization performance
- Default value: 1e-4

#### 2. Learning Rate Scheduler
- Automatically reduces learning rate when validation loss plateaus
- Helps model converge better in later training stages
- Use with `--use_scheduler` flag

#### 3. Gradient Clipping
- Prevents gradient explosion in deep networks
- Improves training stability
- Configurable via `--grad_clip` parameter

#### 4. Early Stopping
- Automatically stops training when validation loss stops improving
- Prevents overfitting and saves training time
- Configurable patience via `--patience` parameter

### Examples

#### Basic Training
Quick test with 2 epochs and larger batch size:
```bash
python main.py --epochs 2 --batch_size 256
```

Use balanced dataset with custom learning rate:
```bash
python main.py --epochs 10 --lr 0.0005 --balanced
```

Small validation/test subsets for faster training:
```bash
python main.py --epochs 3 --val_ratio 0.1 --test_ratio 0.1
```

#### Advanced Training with Stability Features
Enable learning rate scheduler and early stopping:
```bash
python main.py --epochs 50 --use_scheduler --patience 15
```

Use stronger regularization:
```bash
python main.py --epochs 50 --weight_decay 1e-3 --dropout 0.5
```

Strict gradient clipping for unstable training:
```bash
python main.py --epochs 50 --grad_clip 0.5 --use_scheduler
```

Complete configuration for stable training:
```bash
python main.py --epochs 50 --use_scheduler --weight_decay 1e-4 --grad_clip 1.0 --patience 15 --lr 0.001
```

### Parameter Recommendations

#### For Unstable Training
- Use `--use_scheduler` to enable learning rate scheduling
- Increase `--weight_decay` to 1e-3 or 1e-2
- Reduce `--grad_clip` to 0.5 or 0.1
- Increase `--patience` to 15-20

#### For Overfitting
- Increase `--weight_decay` to 1e-3
- Increase `--dropout` to 0.5
- Reduce `--epochs` and use `--patience` for early stopping

#### For Faster Training
- Reduce `--val_ratio` and `--test_ratio` to 0.1
- Increase `--batch_size` if memory allows
- Use `--patience` to stop early when converged

The balanced dataset is now ready for use. To train with balanced data, use:
python main.py --balanced --epochs [num_epochs] --batch_size [batch_size]

### Output

The experiment results are saved in:
- `images/`: Training plots, confusion matrices, and ROC curves
- `logs/`: Training logs
- Console output: Classification reports and metrics

## References

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Authors

## Citation
