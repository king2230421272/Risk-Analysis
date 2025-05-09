import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import json

class NeuralNetworkRegressor(nn.Module):
    """
    Neural Network for regression tasks.
    """
    def __init__(self, input_dim, hidden_dims=[64, 32], output_dim=1, dropout_rate=0.2):
        """
        Initialize the neural network model.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of the input features
        hidden_dims : list
            List of dimensions for the hidden layers
        output_dim : int
            Dimension of the output (default: 1 for regression)
        dropout_rate : float
            Dropout rate for regularization
        """
        super(NeuralNetworkRegressor, self).__init__()
        
        # Store parameters
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # Build network architecture
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_dims[0]))
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the network."""
        return self.model(x)

class LSTMRegressor(nn.Module):
    """
    LSTM Network for sequence-based regression tasks.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout_rate=0.2):
        """
        Initialize the LSTM model.
        
        Parameters:
        -----------
        input_dim : int
            Dimension of the input features
        hidden_dim : int
            Dimension of the hidden state
        num_layers : int
            Number of LSTM layers
        output_dim : int
            Dimension of the output (default: 1 for regression)
        dropout_rate : float
            Dropout rate for regularization
        """
        super(LSTMRegressor, self).__init__()
        
        # Store parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
        --------
        torch.Tensor
            Output tensor
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Get output from the last time step
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layer
        output = self.fc(lstm_out)
        
        return output

class Predictor:
    """
    Module for training prediction models and generating forecasts.
    """
    
    def __init__(self):
        """Initialize predictor with default settings."""
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_names = None
        self.scaler_X = None
        self.scaler_y = None
        self.model_type = None
        self.nn_config = None  # For neural network configurations
        self.training_history = None  # For neural network training history
        
    def predict_with_model(self, model_type, model_details, input_data):
        """
        Make predictions using a saved model.
        
        Parameters:
        -----------
        model_type : str
            Type of the model ('Linear Regression', 'Decision Tree', 
            'Random Forest', 'Gradient Boosting', 'Neural Network', 'LSTM Network')
        model_details : dict
            Dictionary containing model parameters and details
        input_data : pandas.DataFrame
            Input data for prediction with feature columns matching the model's requirements
            
        Returns:
        --------
        numpy.ndarray
            Predicted values
        """
        # Check input data
        if input_data is None or len(input_data) == 0:
            raise ValueError("Input data cannot be empty")
            
        # Get model object or weights
        if model_type == 'Linear Regression':
            model = LinearRegression()
            model.coef_ = np.array(model_details.get('weights', []))
            model.intercept_ = model_details.get('intercept', 0)
            
        elif model_type == 'Decision Tree':
            model = joblib.loads(model_details.get('model_binary', ''))
            
        elif model_type == 'Random Forest':
            model = joblib.loads(model_details.get('model_binary', ''))
            
        elif model_type == 'Gradient Boosting':
            model = joblib.loads(model_details.get('model_binary', ''))
            
        elif model_type == 'Neural Network':
            # Recreate the neural network architecture
            input_dim = model_details.get('neural_network_config', {}).get('input_dim', len(input_data.columns))
            hidden_dims = model_details.get('neural_network_config', {}).get('hidden_dims', [64, 32])
            output_dim = model_details.get('neural_network_config', {}).get('output_dim', 1)
            dropout_rate = model_details.get('neural_network_config', {}).get('dropout_rate', 0.2)
            
            # Create model with same architecture
            model = NeuralNetworkRegressor(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                dropout_rate=dropout_rate
            )
            
            # Load model weights if available
            if 'model_weights' in model_details:
                # Convert weights back to state dict
                state_dict = {}
                for key, value in model_details['model_weights'].items():
                    state_dict[key] = torch.tensor(value)
                model.load_state_dict(state_dict)
            
            # Set to evaluation mode
            model.eval()
            
        elif model_type == 'LSTM Network':
            # Recreate the LSTM network architecture
            input_dim = model_details.get('neural_network_config', {}).get('input_dim', len(input_data.columns))
            hidden_dim = model_details.get('neural_network_config', {}).get('hidden_dim', 64)
            num_layers = model_details.get('neural_network_config', {}).get('num_layers', 2)
            output_dim = model_details.get('neural_network_config', {}).get('output_dim', 1)
            dropout_rate = model_details.get('neural_network_config', {}).get('dropout_rate', 0.2)
            
            # Create model with same architecture
            model = LSTMRegressor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout_rate=dropout_rate
            )
            
            # Load model weights if available
            if 'model_weights' in model_details:
                # Convert weights back to state dict
                state_dict = {}
                for key, value in model_details['model_weights'].items():
                    state_dict[key] = torch.tensor(value)
                model.load_state_dict(state_dict)
            
            # Set to evaluation mode
            model.eval()
            
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Check if we need to apply scaling
        if 'scaler_X_mean' in model_details and 'scaler_X_scale' in model_details:
            # Create and configure the scaler with saved parameters
            scaler_X = StandardScaler()
            scaler_X.mean_ = np.array(model_details['scaler_X_mean'])
            scaler_X.scale_ = np.array(model_details['scaler_X_scale'])
            
            # Scale input data
            scaled_input = scaler_X.transform(input_data)
        else:
            # Use as-is
            scaled_input = input_data.values
            
        # Make prediction based on model type
        if model_type in ['Neural Network', 'LSTM Network']:
            # Convert to tensor
            input_tensor = torch.FloatTensor(scaled_input)
            
            # Reshape for LSTM if needed
            if model_type == 'LSTM Network':
                sequence_length = model_details.get('neural_network_config', {}).get('sequence_length', 1)
                if len(input_tensor.shape) < 3:
                    # Reshape to [batch_size, sequence_length, features]
                    input_tensor = input_tensor.unsqueeze(0)
                    if input_tensor.shape[0] > 1 and input_tensor.shape[1] == 1:
                        # Single sample but multiple features
                        input_tensor = input_tensor.unsqueeze(1)
            
            # Get prediction
            with torch.no_grad():
                prediction = model(input_tensor).numpy()
                
        else:
            # Use scikit-learn model
            prediction = model.predict(scaled_input)
            
        # Check if we need to invert scaling for the output
        if 'scaler_y_mean' in model_details and 'scaler_y_scale' in model_details:
            # Create and configure the scaler with saved parameters
            scaler_y = StandardScaler()
            scaler_y.mean_ = np.array(model_details['scaler_y_mean'])
            scaler_y.scale_ = np.array(model_details['scaler_y_scale'])
            
            # Reshape prediction for inverse transform if needed
            if len(prediction.shape) == 1:
                prediction = prediction.reshape(-1, 1)
                
            # Invert scaling
            prediction = scaler_y.inverse_transform(prediction)
            
            # Flatten if single dimension
            if prediction.shape[1] == 1:
                prediction = prediction.flatten()
        
        return prediction
    
    def train_with_multiple_datasets(self, breach_data, non_breach_data, target_column, model_type, test_size=0.2, feature_columns=None, **model_params):
        """
        Train a prediction model using both breach and non-breach datasets simultaneously.
        
        Parameters:
        -----------
        breach_data : pandas.DataFrame
            Processed breach data (dam failure data)
        non_breach_data : pandas.DataFrame
            Processed non-breach data (intact dam data)
        target_column : str or list[str]
            Name(s) of the target column(s)
        model_type : str
            Type of model to train ('Linear Regression', 'Decision Tree', 
            'Random Forest', 'Gradient Boosting', 'Neural Network', 'LSTM Network')
        test_size : float
            Proportion of data to use for testing (default: 0.2)
        feature_columns : str or list[str]
            Name(s) of the feature column(s)
        **model_params : dict
            Additional parameters for the model
            
        Returns:
        --------
        tuple
            (predictions_df, model_details, metrics)
            - predictions_df: DataFrame with predictions
            - model_details: Dictionary with model details
            - metrics: Dictionary with performance metrics
        """
        # Ensure both datasets have the same columns
        common_columns = set(breach_data.columns).intersection(set(non_breach_data.columns))
        
        # 处理target_column参数，确保它总是列表形式
        if isinstance(target_column, str):
            target_column_list = [target_column]
        else:
            target_column_list = target_column.copy() if hasattr(target_column, 'copy') else list(target_column)
            
        # Filter feature columns to only include common columns
        if feature_columns is None:
            feature_columns = [col for col in common_columns if col not in target_column_list]
        else:
            if isinstance(feature_columns, str):
                feature_columns = [feature_columns]
            feature_columns = [col for col in feature_columns if col in common_columns]
        
        # Ensure target column is in both datasets
        for col in target_column_list:
            if col not in common_columns:
                raise ValueError(f"Target column '{col}' not found in both datasets")
        
        # Add a dataset identifier column
        breach_data = breach_data.copy()
        non_breach_data = non_breach_data.copy()
        breach_data['dataset_type'] = 1  # 1 for breach data
        non_breach_data['dataset_type'] = 0  # 0 for non-breach data
        
        # Add dataset_type to feature columns if needed for prediction
        use_dataset_type = model_params.pop('use_dataset_type', True)
        if use_dataset_type:
            feature_columns = feature_columns + ['dataset_type']
        
        # Combine the datasets
        combined_data = pd.concat([breach_data, non_breach_data], ignore_index=True)
        
        print(f"Combined dataset size: {combined_data.shape[0]} rows, {combined_data.shape[1]} columns")
        print(f"Breach data: {breach_data.shape[0]} rows, Non-breach data: {non_breach_data.shape[0]} rows")
        
        # Use the standard train_and_predict method with the combined dataset
        return self.train_and_predict(
            data=combined_data,
            target_column=target_column,
            model_type=model_type,
            test_size=test_size,
            feature_columns=feature_columns,
            **model_params
        )
    
    def train_and_predict(self, data, target_column, model_type, test_size=0.2, feature_columns=None, **model_params):
        """
        Train a prediction model and generate forecasts.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Processed input data
        target_column : str or list[str]
            Name(s) of the target column(s)
        model_type : str
            Type of model to train ('Linear Regression', 'Decision Tree', 
            'Random Forest', 'Gradient Boosting', 'Neural Network', 'LSTM Network')
        test_size : float
            Proportion of data to use for testing (default: 0.2)
        feature_columns : str or list[str]
            Name(s) of the feature column(s)
        **model_params : dict
            Additional parameters for the model
            
        Returns:
        --------
        tuple
            (predictions_df, model_details, metrics)
            - predictions_df: DataFrame with predictions
            - model_details: Dictionary with model details
            - metrics: Dictionary with performance metrics
        """
        # Store the model type
        self.model_type = model_type
        
        # Ensure target column exists in the data
        if isinstance(target_column, str):
            target_column = [target_column]
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col not in target_column]
        elif isinstance(feature_columns, str):
            feature_columns = [feature_columns]
        
        # Process custom conditions if provided
        custom_conditions = model_params.pop('custom_conditions', None)
        if custom_conditions:
            # Create a copy of the data to avoid modifying the original
            data_with_conditions = data.copy()
            
            # Add custom condition columns
            for cond_name, cond_value in custom_conditions.items():
                # Add uniform column with the condition value
                data_with_conditions[cond_name] = cond_value
                
                # Add to feature columns if not already there
                if cond_name not in feature_columns:
                    feature_columns.append(cond_name)
            
            # Use the enhanced dataset
            X = data_with_conditions[feature_columns]
        else:
            # Use original data
            X = data[feature_columns]
            
        y = data[target_column]
        
        # Check for non-numeric data and convert or remove as necessary
        non_numeric_columns = []
        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                try:
                    # Try to clean string values by removing any non-numeric characters except decimal points
                    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                        # For string values like "ET 18.90", extract just the numeric part
                        # This regex extracts numbers with decimal points
                        X[col] = X[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
                    
                    # Try to convert strings to float
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                    
                    # Fill any resulting NaN values with the mean of the column
                    if X[col].isna().any():
                        if X[col].notna().any():  # Only compute mean if there are some non-NA values
                            mean_val = X[col].mean()
                            X[col] = X[col].fillna(mean_val)
                            print(f"Column {col} had some values that couldn't be converted to numeric. "
                                  f"These were replaced with the mean value: {mean_val}")
                        else:
                            # If all values became NaN, mark for removal
                            non_numeric_columns.append(col)
                            print(f"Column {col} contains non-numeric data that cannot be converted.")
                except Exception as e:
                    # If conversion fails entirely, remember this column for potential removal
                    non_numeric_columns.append(col)
                    print(f"Column {col} contains non-numeric data that cannot be converted: {e}")
        
        # If we found non-numeric columns that couldn't be converted, remove them
        if non_numeric_columns:
            print(f"Removing non-numeric columns: {non_numeric_columns}")
            X = X.drop(columns=non_numeric_columns)
            # Update feature names list to reflect removed columns
            self.feature_names = [col for col in self.feature_names if col not in non_numeric_columns]
            
            # Warn if we removed all columns
            if len(X.columns) == 0:
                raise ValueError("All feature columns contained non-numeric data and had to be removed. "
                                "Please check your data and ensure at least some columns contain numeric values.")
        
        # Save feature names (moved after non-numeric column removal check)
        self.feature_names = feature_columns if not non_numeric_columns else [col for col in feature_columns if col not in non_numeric_columns]
        self.target_names = target_column
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if model_type in ['Neural Network', 'LSTM Network']:
            # Double-check for any remaining non-numeric values
            # Sometimes pandas allows mixed types in numeric columns
            for col in self.X_train.columns:
                try:
                    # Convert to numpy array and check for non-numeric values
                    col_array = self.X_train[col].values
                    if not np.issubdtype(col_array.dtype, np.number):
                        print(f"Warning: Column {col} contains non-numeric values. Attempting final conversion.")
                        # Try one more conversion with more aggressive error handling
                        try:
                            # Extract numeric parts from strings like "ET 18.90"
                            numeric_vals = self.X_train[col].astype(str).str.extract(r'(\d+\.?\d*)')[0]
                            self.X_train[col] = pd.to_numeric(numeric_vals, errors='coerce')
                            numeric_vals = self.X_test[col].astype(str).str.extract(r'(\d+\.?\d*)')[0]
                            self.X_test[col] = pd.to_numeric(numeric_vals, errors='coerce')
                        except Exception as e:
                            print(f"Error in final conversion of column {col}: {e}")
                            # If conversion still fails, replace with column mean or zero
                            if len(self.X_train[col].dropna()) > 0:
                                col_mean = self.X_train[col].dropna().mean()
                                self.X_train[col] = self.X_train[col].fillna(col_mean)
                                self.X_test[col] = self.X_test[col].fillna(col_mean)
                            else:
                                print(f"Column {col} has no valid numeric values, filling with zeros")
                                self.X_train[col] = 0
                                self.X_test[col] = 0
                except Exception as e:
                    print(f"Error processing column {col}: {e}")
                    # Last resort: fill with zeros
                    self.X_train[col] = 0
                    self.X_test[col] = 0
                    
            # Convert DataFrames to numpy for scaling
            X_train_np = self.X_train.to_numpy()
            X_test_np = self.X_test.to_numpy()
            
            # Neural network models require normalized data
            self.scaler_X = StandardScaler()
            try:
                X_train_scaled = self.scaler_X.fit_transform(X_train_np)
                X_test_scaled = self.scaler_X.transform(X_test_np)
            except Exception as e:
                print(f"Error during feature scaling: {e}")
                print("Attempting to scale each feature individually...")
                # Fallback: Scale each feature individually
                X_train_scaled = np.zeros_like(X_train_np, dtype=float)
                X_test_scaled = np.zeros_like(X_test_np, dtype=float)
                
                for i in range(X_train_np.shape[1]):
                    try:
                        scaler = StandardScaler()
                        X_train_scaled[:, i] = scaler.fit_transform(X_train_np[:, i].reshape(-1, 1)).ravel()
                        X_test_scaled[:, i] = scaler.transform(X_test_np[:, i].reshape(-1, 1)).ravel()
                    except Exception as e:
                        print(f"Error scaling feature {i}: {e}")
                        # If scaling fails, use the original values normalized by their max
                        col_max = max(np.max(np.abs(X_train_np[:, i])), 1e-10)
                        X_train_scaled[:, i] = X_train_np[:, i] / col_max
                        X_test_scaled[:, i] = X_test_np[:, i] / col_max
            
            # Also scale the target variable
            if len(self.y_train.shape) == 1:
                self.y_train = self.y_train.values.reshape(-1, 1)
                self.y_test = self.y_test.values.reshape(-1, 1)
            
            # Check for non-numeric values in target
            if not np.issubdtype(self.y_train.dtype, np.number):
                print("Warning: Target contains non-numeric values. Converting to numeric.")
                try:
                    self.y_train = np.array([float(str(y).replace(',', '')) for y in self.y_train.ravel()]).reshape(-1, 1)
                    self.y_test = np.array([float(str(y).replace(',', '')) for y in self.y_test.ravel()]).reshape(-1, 1)
                except:
                    print("Error converting target to numeric. Using zeros.")
                    self.y_train = np.zeros_like(self.y_train, dtype=float)
                    self.y_test = np.zeros_like(self.y_test, dtype=float)
            
            self.scaler_y = StandardScaler()
            try:
                y_train_scaled = self.scaler_y.fit_transform(self.y_train)
                y_test_scaled = self.scaler_y.transform(self.y_test)
            except Exception as e:
                print(f"Error during target scaling: {e}")
                # Fallback: Simple normalization
                y_max = max(np.max(np.abs(self.y_train)), 1e-10)
                y_train_scaled = self.y_train / y_max
                y_test_scaled = self.y_test / y_max
            
            # Extract neural network parameters
            hidden_dims = model_params.pop('hidden_dims', [64, 32])
            learning_rate = model_params.pop('learning_rate', 0.001)
            batch_size = model_params.pop('batch_size', 32)
            epochs = model_params.pop('epochs', 100)
            dropout_rate = model_params.pop('dropout_rate', 0.2)
            patience = model_params.pop('patience', 10)  # For early stopping
            sequence_length = model_params.pop('sequence_length', 5)  # For LSTM
            
            # Initialize the neural network model
            output_dim = len(target_column)
            self.model = NeuralNetworkRegressor(
                input_dim=X_train_scaled.shape[1],
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                dropout_rate=dropout_rate
            )
            
            # Store model configuration
            self.nn_config = {
                'type': 'Neural Network',
                'input_dim': X_train_scaled.shape[1],
                'hidden_dims': hidden_dims,
                'output_dim': output_dim,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs
            }
            
            # Convert data to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.FloatTensor(y_train_scaled)
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            y_test_tensor = torch.FloatTensor(y_test_scaled)
            
            # Create DataLoader for batch training
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            
            # Train the model with early stopping
            self.training_history = self._train_neural_network(
                train_loader, criterion, optimizer, epochs, patience,
                X_test_tensor, y_test_tensor
            )
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                train_pred_scaled = self.model(X_train_tensor).numpy()
                test_pred_scaled = self.model(X_test_tensor).numpy()
            
            # Rescale predictions
            train_pred = self.scaler_y.inverse_transform(train_pred_scaled)
            test_pred = self.scaler_y.inverse_transform(test_pred_scaled)
            
        else:
            # Traditional ML models
            # Initialize the model based on the specified type
            if model_type == 'Linear Regression':
                self.model = LinearRegression(**model_params)
            elif model_type == 'Decision Tree':
                self.model = DecisionTreeRegressor(random_state=42, **model_params)
            elif model_type == 'Random Forest':
                self.model = RandomForestRegressor(random_state=42, **model_params)
            elif model_type == 'Gradient Boosting':
                self.model = GradientBoostingRegressor(random_state=42, **model_params)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Train the model
            self.model.fit(self.X_train, self.y_train)
            
            # Make predictions
            train_pred = self.model.predict(self.X_train)
            test_pred = self.model.predict(self.X_test)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(train_pred, test_pred)
        
        # Create a dataframe with test data and predictions
        predictions_df = self.X_test.copy()
        predictions_df[target_column] = self.y_test
        predictions_df['predicted'] = test_pred
        predictions_df['error'] = predictions_df['predicted'] - predictions_df[target_column]
        
        # Get model details
        model_details = self._get_model_details(model_type)
        
        return predictions_df, model_details, metrics
    
    def _prepare_sequences(self, data, sequence_length):
        """
        Prepare sequences for LSTM model.
        
        Parameters:
        -----------
        data : numpy.ndarray
            Input data
        sequence_length : int
            Length of sequences to create
            
        Returns:
        --------
        numpy.ndarray
            Sequence data of shape [num_sequences, sequence_length, features]
        """
        sequences = []
        n_samples = len(data) - sequence_length + 1
        
        for i in range(n_samples):
            sequence = data[i:i+sequence_length]
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def _train_neural_network(self, train_loader, criterion, optimizer, epochs, patience, X_val, y_val):
        """
        Train the neural network model.
        
        Parameters:
        -----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training data
        criterion : torch.nn.Module
            Loss function
        optimizer : torch.optim.Optimizer
            Optimizer
        epochs : int
            Number of epochs
        patience : int
            Number of epochs to wait for improvement before early stopping
        X_val : torch.Tensor
            Validation features
        y_val : torch.Tensor
            Validation targets
            
        Returns:
        --------
        dict
            Training history
        """
        print("Starting neural network training...")
        start_time = time.time()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'best_epoch': 0
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        early_stop_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                # Forward pass
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                history['val_loss'].append(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                history['best_epoch'] = epoch
                early_stop_counter = 0
                # Save best model state
                best_model_state = self.model.state_dict().copy()
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Training summary
        training_time = time.time() - start_time
        history['training_time'] = training_time
        history['total_epochs'] = epoch + 1
        
        print(f"Training completed in {training_time:.2f} seconds.")
        print(f"Best validation loss: {best_val_loss:.4f} at epoch {history['best_epoch']+1}")
        
        return history
    
    def _calculate_metrics(self, train_pred, test_pred):
        """
        Calculate performance metrics for the model.
        
        Parameters:
        -----------
        train_pred : numpy.ndarray
            Predictions on the training set
        test_pred : numpy.ndarray
            Predictions on the test set
            
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        # Ensure arrays are 1D
        if isinstance(self.y_train, pd.Series):
            y_train = self.y_train.values
        elif isinstance(self.y_train, pd.DataFrame):
            y_train = self.y_train.values.flatten() if len(self.y_train.shape) > 1 else self.y_train.values
        else:
            y_train = self.y_train.ravel() if hasattr(self.y_train, 'ravel') and len(self.y_train.shape) > 1 else self.y_train
            
        if isinstance(self.y_test, pd.Series):
            y_test = self.y_test.values
        elif isinstance(self.y_test, pd.DataFrame):
            y_test = self.y_test.values.flatten() if len(self.y_test.shape) > 1 else self.y_test.values
        else:
            y_test = self.y_test.ravel() if hasattr(self.y_test, 'ravel') and len(self.y_test.shape) > 1 else self.y_test
            
        # Ensure predictions are 1D
        if hasattr(train_pred, 'ravel') and len(train_pred.shape) > 1:
            train_pred = train_pred.ravel()
        elif isinstance(train_pred, pd.DataFrame) or isinstance(train_pred, pd.Series):
            train_pred = train_pred.values.flatten() if len(train_pred.shape) > 1 else train_pred.values
            
        if hasattr(test_pred, 'ravel') and len(test_pred.shape) > 1:
            test_pred = test_pred.ravel()
        elif isinstance(test_pred, pd.DataFrame) or isinstance(test_pred, pd.Series):
            test_pred = test_pred.values.flatten() if len(test_pred.shape) > 1 else test_pred.values
        
        # Calculate metrics for training set
        train_mse = mean_squared_error(y_train, train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        # Calculate metrics for test set
        test_mse = mean_squared_error(y_test, test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        metrics = {
            'Training MSE': train_mse,
            'Training RMSE': train_rmse,
            'Training MAE': train_mae,
            'Training R²': train_r2,
            'Test MSE': test_mse,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'Test R²': test_r2
        }
        
        # Add NN-specific metrics if available
        if self.training_history is not None:
            metrics.update({
                'Training Time (s)': self.training_history['training_time'],
                'Best Epoch': self.training_history['best_epoch'] + 1,
                'Total Epochs': self.training_history['total_epochs'],
                'Final Training Loss': self.training_history['train_loss'][-1],
                'Final Validation Loss': self.training_history['val_loss'][-1]
            })
        
        return metrics
    
    def _get_model_details(self, model_type):
        """
        Get details about the trained model.
        
        Parameters:
        -----------
        model_type : str
            Type of the trained model
            
        Returns:
        --------
        dict
            Dictionary with model details
        """
        model_details = {
            'model_type': model_type,
            'feature_names': self.feature_names,
            'target_names': self.target_names
        }
        
        # Get feature importance if available
        if model_type in ['Neural Network', 'LSTM Network']:
            # Neural network models don't have native feature importance
            model_details['neural_network_config'] = self.nn_config
            
            if self.training_history:
                model_details['training_history'] = {
                    'train_loss': self.training_history['train_loss'],
                    'val_loss': self.training_history['val_loss'],
                    'best_epoch': self.training_history['best_epoch'],
                    'total_epochs': self.training_history['total_epochs'],
                    'training_time': self.training_history['training_time']
                }
        elif hasattr(self.model, 'feature_importances_'):
            model_details['feature_importance'] = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use coefficients as feature importance
            model_details['feature_importance'] = np.abs(self.model.coef_)
        
        return model_details
    
    def plot_training_history(self):
        """
        Plot the training history for neural network models.
        
        Returns:
        --------
        matplotlib.figure.Figure
            The figure object
        """
        if self.training_history is None:
            raise ValueError("No training history available. Model may not be a neural network or has not been trained yet.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.training_history['train_loss'], label='Training Loss')
        ax.plot(self.training_history['val_loss'], label='Validation Loss')
        ax.axvline(x=self.training_history['best_epoch'], color='r', linestyle='--', label=f'Best Epoch ({self.training_history["best_epoch"]+1})')
        
        ax.set_title(f"Training History - {self.nn_config['type']}")
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (MSE)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def save_model(self, filepath='model.joblib'):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if self.model is None:
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Handle different model types
            if self.model_type in ['Neural Network', 'LSTM Network']:
                # For PyTorch models, save state_dict and configuration
                model_info = {
                    'model_type': self.model_type,
                    'nn_config': self.nn_config,
                    'feature_names': self.feature_names,
                    'target_names': self.target_names,
                    'scaler_X': self.scaler_X,
                    'scaler_y': self.scaler_y
                }
                
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_info': model_info
                }, filepath)
            else:
                # For sklearn models, use joblib
                model_info = {
                    'model': self.model,
                    'model_type': self.model_type,
                    'feature_names': self.feature_names,
                    'target_names': self.target_names
                }
                joblib.dump(model_info, filepath)
            
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath='model.joblib'):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            # Check if it's a PyTorch model (has .pt extension)
            if filepath.endswith('.pt'):
                # Load PyTorch model
                checkpoint = torch.load(filepath)
                model_info = checkpoint['model_info']
                
                self.model_type = model_info['model_type']
                self.nn_config = model_info['nn_config']
                self.feature_names = model_info['feature_names']
                self.target_names = model_info['target_names']
                self.scaler_X = model_info['scaler_X']
                self.scaler_y = model_info['scaler_y']
                
                # Recreate the model architecture
                if self.model_type == 'Neural Network':
                    self.model = NeuralNetworkRegressor(
                        input_dim=self.nn_config['input_dim'],
                        hidden_dims=self.nn_config['hidden_dims'],
                        output_dim=self.nn_config['output_dim'],
                        dropout_rate=self.nn_config['dropout_rate']
                    )
                elif self.model_type == 'LSTM Network':
                    self.model = LSTMRegressor(
                        input_dim=self.nn_config['input_dim'],
                        hidden_dim=self.nn_config['hidden_dim'],
                        num_layers=self.nn_config['num_layers'],
                        output_dim=self.nn_config['output_dim'],
                        dropout_rate=self.nn_config['dropout_rate']
                    )
                
                # Load the state dict
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
            else:
                # Load sklearn model using joblib
                model_info = joblib.load(filepath)
                self.model = model_info['model']
                self.model_type = model_info['model_type']
                self.feature_names = model_info['feature_names']
                self.target_names = model_info['target_names']
            
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
            
    def load_data_from_database(self, db_handler, dataset_id):
        """
        Load a dataset from the database.
        
        Parameters:
        -----------
        db_handler : DatabaseHandler
            Database handler instance
        dataset_id : int
            ID of the dataset to load
            
        Returns:
        --------
        pandas.DataFrame
            The loaded dataset
        """
        try:
            return db_handler.load_dataset(dataset_id=dataset_id)
        except Exception as e:
            print(f"Error loading dataset from database: {e}")
            raise
