import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

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
    
    def train_and_predict(self, data, target_column, model_type, test_size=0.2, **model_params):
        """
        Train a prediction model and generate forecasts.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Processed input data
        target_column : str
            Name of the target column
        model_type : str
            Type of model to train ('Linear Regression', 'Decision Tree', 
            'Random Forest', or 'Gradient Boosting')
        test_size : float
            Proportion of data to use for testing (default: 0.2)
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
        # Ensure target column exists in the data
        if target_column not in data.columns:
            raise ValueError(f"Target column '{target_column}' not found in the data.")
        
        # Split data into features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Save feature names
        self.feature_names = X.columns.tolist()
        
        # Split data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
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
        # Calculate metrics for training set
        train_mse = mean_squared_error(self.y_train, train_pred)
        train_rmse = np.sqrt(train_mse)
        train_mae = mean_absolute_error(self.y_train, train_pred)
        train_r2 = r2_score(self.y_train, train_pred)
        
        # Calculate metrics for test set
        test_mse = mean_squared_error(self.y_test, test_pred)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(self.y_test, test_pred)
        test_r2 = r2_score(self.y_test, test_pred)
        
        return {
            'Training MSE': train_mse,
            'Training RMSE': train_rmse,
            'Training MAE': train_mae,
            'Training R²': train_r2,
            'Test MSE': test_mse,
            'Test RMSE': test_rmse,
            'Test MAE': test_mae,
            'Test R²': test_r2
        }
    
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
            'feature_names': self.feature_names
        }
        
        # Get feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            model_details['feature_importance'] = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use coefficients as feature importance
            model_details['feature_importance'] = np.abs(self.model.coef_)
        
        return model_details
    
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
            
            # Save the model
            joblib.dump(self.model, filepath)
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
            self.model = joblib.load(filepath)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
