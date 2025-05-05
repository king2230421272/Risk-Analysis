import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class DataProcessor:
    """
    Module for data processing, cleaning, and transformation.
    """
    
    def __init__(self):
        """Initialize data processor with default settings."""
        self.scaler = None
    
    def process_data(self, data, target_column=None, handle_missing=False, 
                     missing_method='Mean imputation', remove_duplicates=False,
                     normalize_data=False, norm_method='Min-Max Scaling'):
        """
        Process data based on specified options.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data to process
        target_column : str, optional
            Name of the target column (won't be normalized if specified)
        handle_missing : bool
            Whether to handle missing values
        missing_method : str
            Method for handling missing values ('Remove rows', 'Mean imputation', 
            'Median imputation', or 'Mode imputation')
        remove_duplicates : bool
            Whether to remove duplicate rows
        normalize_data : bool
            Whether to normalize numerical features
        norm_method : str
            Normalization method ('Min-Max Scaling' or 'Standard Scaling')
            
        Returns:
        --------
        pandas.DataFrame
            Processed data
        """
        # Make a copy of the data to avoid modifying the original
        processed_data = data.copy()
        
        # Remove duplicates if specified
        if remove_duplicates:
            processed_data = processed_data.drop_duplicates().reset_index(drop=True)
        
        # Handle missing values if specified
        if handle_missing:
            if missing_method == 'Remove rows':
                # Remove rows with any missing values
                processed_data = processed_data.dropna().reset_index(drop=True)
            else:
                # Identify numeric and categorical columns
                numeric_cols = processed_data.select_dtypes(include=np.number).columns
                categorical_cols = processed_data.select_dtypes(include=['object']).columns
                
                # Handle missing values based on the specified method
                if missing_method == 'Mean imputation':
                    # Impute missing values with mean for numeric columns
                    for col in numeric_cols:
                        if processed_data[col].isnull().sum() > 0:
                            processed_data[col] = processed_data[col].fillna(processed_data[col].mean())
                    
                elif missing_method == 'Median imputation':
                    # Impute missing values with median for numeric columns
                    for col in numeric_cols:
                        if processed_data[col].isnull().sum() > 0:
                            processed_data[col] = processed_data[col].fillna(processed_data[col].median())
                
                elif missing_method == 'Mode imputation':
                    # Impute missing values with mode for all columns
                    for col in processed_data.columns:
                        if processed_data[col].isnull().sum() > 0:
                            processed_data[col] = processed_data[col].fillna(processed_data[col].mode()[0])
                
                # For categorical columns, use mode imputation if not already done
                if missing_method != 'Mode imputation':
                    for col in categorical_cols:
                        if processed_data[col].isnull().sum() > 0:
                            processed_data[col] = processed_data[col].fillna(processed_data[col].mode()[0])
        
        # Normalize numerical features if specified
        if normalize_data:
            # Get numeric columns excluding the target column
            numeric_cols = processed_data.select_dtypes(include=np.number).columns
            if target_column in numeric_cols:
                numeric_cols = [col for col in numeric_cols if col != target_column]
            
            if len(numeric_cols) > 0:
                # Create a copy of numeric data for normalization
                numeric_data = processed_data[numeric_cols].copy()
                
                # Initialize the scaler based on the specified method
                if norm_method == 'Min-Max Scaling':
                    self.scaler = MinMaxScaler()
                else:  # Standard Scaling
                    self.scaler = StandardScaler()
                
                # Normalize the data
                normalized_data = self.scaler.fit_transform(numeric_data)
                
                # Update the processed data with normalized values
                for i, col in enumerate(numeric_cols):
                    processed_data[col] = normalized_data[:, i]
        
        return processed_data
    
    def handle_categorical_features(self, data, method='one-hot'):
        """
        Handle categorical features in the dataset.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        method : str
            Method for handling categorical features ('one-hot' or 'label')
            
        Returns:
        --------
        pandas.DataFrame
            Data with processed categorical features
        """
        # Make a copy of the data
        processed_data = data.copy()
        
        # Identify categorical columns
        categorical_cols = processed_data.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            if method == 'one-hot':
                # One-hot encode categorical variables
                processed_data = pd.get_dummies(processed_data, columns=categorical_cols, drop_first=True)
            elif method == 'label':
                # Label encode categorical variables
                for col in categorical_cols:
                    processed_data[col] = pd.factorize(processed_data[col])[0]
        
        return processed_data
    
    def get_feature_names(self):
        """Get the feature names after processing."""
        if hasattr(self, 'scaler') and self.scaler is not None:
            if hasattr(self.scaler, 'get_feature_names_out'):
                return self.scaler.get_feature_names_out()
        return None
