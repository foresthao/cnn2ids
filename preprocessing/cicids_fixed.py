import pandas as pd
import numpy as np
import glob
import os

from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split


DATA_DIR = os.path.join(os.path.dirname(os.path.abspath('.')), 'data')

class CICIDS2017PreprocessorFixed(object):
    def __init__(self, data_path, training_size, validation_size, testing_size):
        self.data_path = data_path
        self.training_size = training_size
        self.validation_size = validation_size
        self.testing_size = testing_size
        
        self.data = None
        self.features = None
        self.label = None

    def read_data(self):
        """Read and combine all CSV files"""
        filenames = glob.glob(os.path.join(self.data_path, 'raw', '*.csv'))
        datasets = [pd.read_csv(filename) for filename in filenames]

        # Remove white spaces and rename the columns
        for dataset in datasets:
            dataset.columns = [self._clean_column_name(column) for column in dataset.columns]

        # Concatenate the datasets
        self.data = pd.concat(datasets, axis=0, ignore_index=True)
        self.data.drop(labels=['fwd_header_length.1'], axis=1, inplace=True)

    def _clean_column_name(self, column):
        """Clean column names"""
        column = column.strip(' ')
        column = column.replace('/', '_')
        column = column.replace(' ', '_')
        column = column.lower()
        return column

    def remove_duplicate_values(self):
        """Remove duplicate rows"""
        print(f"Before removing duplicates: {len(self.data)} rows")
        self.data.drop_duplicates(inplace=True, keep='first', ignore_index=True)
        print(f"After removing duplicates: {len(self.data)} rows")

    def remove_missing_values(self):
        """Remove missing values"""
        print(f"Before removing missing values: {len(self.data)} rows")
        self.data.dropna(axis=0, inplace=True, how="any")
        print(f"After removing missing values: {len(self.data)} rows")

    def remove_infinite_values(self):
        """Remove infinite values"""
        print(f"Before removing infinite values: {len(self.data)} rows")
        # Replace infinite values to NaN
        self.data.replace([-np.inf, np.inf], np.nan, inplace=True)
        # Remove infinite values
        self.data.dropna(axis=0, how='any', inplace=True)
        print(f"After removing infinite values: {len(self.data)} rows")

    def remove_constant_features(self, threshold=0.001):
        """Remove features with very low variance"""
        # Get only numeric columns for variance analysis
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        # Standard deviation
        data_std = numeric_data.std()
        
        # Find Features that meet the threshold
        constant_features = [column for column, std in data_std.items() if std < threshold]
        
        print(f"Removing {len(constant_features)} constant features: {constant_features}")
        
        # Drop the constant features
        self.data.drop(labels=constant_features, axis=1, inplace=True)

    def remove_correlated_features(self, threshold=0.95):
        """Remove highly correlated features"""
        # Get only numeric columns for correlation analysis
        numeric_data = self.data.select_dtypes(include=[np.number])
        
        # Correlation matrix
        data_corr = numeric_data.corr().abs()

        # Create & Apply mask
        mask = np.triu(np.ones_like(data_corr, dtype=bool), k=1)
        tri_df = data_corr.mask(~mask)

        # Find Features that meet the threshold
        correlated_features = [c for c in tri_df.columns if any(tri_df[c] > threshold)]
        
        print(f"Removing {len(correlated_features)} highly correlated features: {correlated_features}")

        # Drop the highly correlated features
        self.data.drop(labels=correlated_features, axis=1, inplace=True)

    def group_labels(self):
        """Group attack types into broader categories"""
        # Proposed Groupings - More balanced approach
        attack_group = {
            'BENIGN': 'Benign',
            'PortScan': 'Recon', 
            'DDoS': 'DoS',
            'DoS Hulk': 'DoS',
            'DoS GoldenEye': 'DoS',
            'DoS slowloris': 'DoS', 
            'DoS Slowhttptest': 'DoS',
            'Heartbleed': 'DoS',
            'FTP-Patator': 'BruteForce',
            'SSH-Patator': 'BruteForce',
            'Bot': 'Malware',
            'Web Attack � Brute Force': 'WebAttack',
            'Web Attack � Sql Injection': 'WebAttack',
            'Web Attack � XSS': 'WebAttack',
            'Infiltration': 'Advanced'
        }

        # Create grouped label column
        self.data['label_category'] = self.data['label'].map(lambda x: attack_group[x])
        
        # Print label distribution
        print("Label distribution:")
        print(self.data['label_category'].value_counts())
        
    def train_valid_test_split(self, balanced=False):
        """Split data with proper methodology"""
        self.labels = self.data['label_category']
        self.features = self.data.drop(labels=['label', 'label_category'], axis=1)

        # First split: train vs (val + test)
        X_train, X_temp, y_train, y_temp = train_test_split(
            self.features,
            self.labels,
            test_size=(self.validation_size + self.testing_size),
            random_state=42,
            stratify=self.labels
        )
        
        # Second split: val vs test
        val_ratio = self.validation_size / (self.validation_size + self.testing_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(1 - val_ratio),
            random_state=42,
            stratify=y_temp
        )
        
        if balanced:
            X_train, y_train = self._balance_training_data(X_train, y_train)
    
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _balance_training_data(self, X_train, y_train):
        """Balance training data more realistically"""
        from sklearn.utils import resample
        
        # Combine features and labels
        train_data = pd.concat([X_train, y_train], axis=1)
        
        # Get class distributions
        class_counts = y_train.value_counts()
        print(f"Original class distribution: {class_counts.to_dict()}")
        
        # Use more realistic balancing - not perfect balance
        # Oversample minorities to 50% of majority, undersample majority to 80%
        majority_class = class_counts.index[0]
        majority_count = class_counts.iloc[0]
        
        target_majority = int(majority_count * 0.8)  # Reduce majority by 20%
        target_minority = int(target_majority * 0.3)  # Minorities to 30% of reduced majority
        
        resampled_data = []
        
        for class_label in class_counts.index:
            class_data = train_data[train_data.iloc[:, -1] == class_label]
            current_count = len(class_data)
            
            if class_label == majority_class:
                # Undersample majority class
                target_count = target_majority
                resampled_class = resample(class_data, 
                                         n_samples=target_count, 
                                         random_state=42, 
                                         replace=False)
            else:
                # Oversample minority classes
                target_count = min(target_minority, current_count * 3)  # Don't oversample too much
                replace_needed = target_count > current_count
                resampled_class = resample(class_data, 
                                         n_samples=target_count, 
                                         random_state=42, 
                                         replace=replace_needed)
            
            resampled_data.append(resampled_class)
            print(f"Class {class_label}: {current_count} -> {target_count}")
        
        # Combine resampled data
        balanced_data = pd.concat(resampled_data, ignore_index=True)
        
        # Shuffle the data
        balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Separate features and labels
        X_balanced = balanced_data.iloc[:, :-1]
        y_balanced = balanced_data.iloc[:, -1]
        
        return X_balanced, y_balanced
    
    def scale(self, training_set, validation_set, testing_set):
        """Apply proper scaling without data leakage"""
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = training_set, validation_set, testing_set
        
        # Get only numeric features (categorical features should be handled separately)
        numeric_features = self.features.select_dtypes(include=[np.number]).columns
        
        # Use RobustScaler instead of QuantileTransformer to reduce overfitting
        scaler = RobustScaler()
        
        # Fit scaler ONLY on training data
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train[numeric_features]), 
            columns=numeric_features,
            index=X_train.index
        )
        
        # Transform validation and test data using the same scaler
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val[numeric_features]), 
            columns=numeric_features,
            index=X_val.index
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test[numeric_features]), 
            columns=numeric_features,
            index=X_test.index
        )
        
        # Handle categorical features if any
        categorical_features = self.features.select_dtypes(exclude=[np.number]).columns
        if len(categorical_features) > 0:
            print(f"Warning: {len(categorical_features)} categorical features found but not processed")
        
        # Encode labels
        le = LabelEncoder()
        
        y_train_encoded = pd.DataFrame(le.fit_transform(y_train), columns=["label"])
        y_val_encoded = pd.DataFrame(le.transform(y_val), columns=["label"])
        y_test_encoded = pd.DataFrame(le.transform(y_test), columns=["label"])
        
        print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        return (X_train_scaled, y_train_encoded), (X_val_scaled, y_val_encoded), (X_test_scaled, y_test_encoded)


if __name__ == "__main__":

    cicids2017 = CICIDS2017PreprocessorFixed(
        data_path=DATA_DIR,
        training_size=0.7,  # Increase training data
        validation_size=0.15,  # Reduce validation 
        testing_size=0.15
    )

    # Read datasets
    print("Reading data...")
    cicids2017.read_data()

    # Remove NaN, -Inf, +Inf, Duplicates
    print("\nCleaning data...")
    cicids2017.remove_duplicate_values()
    cicids2017.remove_missing_values()
    cicids2017.remove_infinite_values()

    # Drop constant & correlated features
    print("\nFeature selection...")
    cicids2017.remove_constant_features()
    cicids2017.remove_correlated_features()

    # Create new label category
    print("\nGrouping labels...")
    cicids2017.group_labels()

    # Split & Normalise data sets
    print("\nSplitting data...")
    training_set, validation_set, testing_set = cicids2017.train_valid_test_split(balanced=True)
    
    print("\nScaling features...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = cicids2017.scale(training_set, validation_set, testing_set)
    
    # Create directories
    os.makedirs(os.path.join(DATA_DIR, 'processed_fixed', 'train'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed_fixed', 'val'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'processed_fixed', 'test'), exist_ok=True)
    
    # Save the results
    print("\nSaving processed data...")
    X_train.to_pickle(os.path.join(DATA_DIR, 'processed_fixed', 'train/train_features.pkl'))
    X_val.to_pickle(os.path.join(DATA_DIR, 'processed_fixed', 'val/val_features.pkl'))
    X_test.to_pickle(os.path.join(DATA_DIR, 'processed_fixed', 'test/test_features.pkl'))

    y_train.to_pickle(os.path.join(DATA_DIR, 'processed_fixed', 'train/train_labels.pkl'))
    y_val.to_pickle(os.path.join(DATA_DIR, 'processed_fixed', 'val/val_labels.pkl'))
    y_test.to_pickle(os.path.join(DATA_DIR, 'processed_fixed', 'test/test_labels.pkl'))
    
    print(f"\nFinal dataset sizes:")
    print(f"Training: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples") 
    print(f"Test: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    
    print(f"\nTraining set class distribution:")
    print(y_train.value_counts().sort_index())