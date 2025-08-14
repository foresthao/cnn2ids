import torch.nn as nn
import torch.nn.functional as F
import torch

class CNN2ID_Fixed(nn.Module):
    def __init__(self, num_classes: int = 7, input_features: int = 49, dropout_rate: float = 0.3):
        super(CNN2ID_Fixed, self).__init__()
        
        # Add input layer normalization
        self.input_bn = nn.BatchNorm1d(1)
        
        # First conv block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(dropout_rate * 0.5)
        
        # Second conv block  
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(dropout_rate * 0.7)
        
        # Third conv block
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        # Fully connected layers with proper regularization
        self.fc1 = nn.Linear(128, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc1_dropout = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc2_dropout = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, 64)
        self.fc3_bn = nn.BatchNorm1d(64)
        self.fc3_dropout = nn.Dropout(dropout_rate * 0.5)
        
        self.classifier = nn.Linear(64, num_classes)
        
        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input normalization
        x = self.input_bn(x)
        
        # First conv block
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        if self.training:  # Apply dropout only during training
            x = self.dropout1(x.transpose(1, 2)).transpose(1, 2)
        
        # Second conv block
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        if self.training:
            x = self.dropout2(x.transpose(1, 2)).transpose(1, 2)
        
        # Third conv block
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        if self.training:
            x = self.dropout3(x.transpose(1, 2)).transpose(1, 2)
        
        # Global pooling and flatten
        x = self.gap(x)
        x = self.flatten(x)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc1_bn(x)
        x = self.fc1_dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.fc2_bn(x)
        x = self.fc2_dropout(x)
        
        x = F.relu(self.fc3(x))
        x = self.fc3_bn(x)
        x = self.fc3_dropout(x)
        
        # Classification layer (no activation - will be handled by loss function)
        x = self.classifier(x)
        
        return x