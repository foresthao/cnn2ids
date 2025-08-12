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

Available options:
- `--epochs EPOCHS`: Number of training epochs (default: 5)
- `--batch_size BATCH_SIZE`: Batch size for training (default: 128)
- `--lr LR`: Learning rate (default: 0.001)
- `--balanced`: Use balanced dataset (default: False)
- `--val_ratio VAL_RATIO`: Ratio of validation subset (default: 0.3)
- `--test_ratio TEST_RATIO`: Ratio of test subset (default: 0.3)

### Examples

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
