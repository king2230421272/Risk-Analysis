import numpy as np
import pandas as pd
from scipy import stats

class RiskAssessor:
    """
    Module for assessing risks in prediction models and results.
    """
    
    def __init__(self):
        """Initialize risk assessor with default settings."""
        pass
    
    def prediction_intervals(self, predictions_df, confidence_level=95):
        """
        Calculate prediction intervals for the predictions.
        
        Parameters:
        -----------
        predictions_df : pandas.DataFrame
            DataFrame containing actual values, predictions, and errors
        confidence_level : int
            Confidence level for prediction intervals (default: 95)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with prediction intervals
        """
        # Make a copy of the predictions dataframe
        risk_df = predictions_df.copy()
        
        # Calculate the standard error of the predictions
        std_error = np.std(risk_df['error'])
        
        # Calculate the confidence interval factor based on the confidence level
        alpha = (100 - confidence_level) / 100
        z_value = stats.norm.ppf(1 - alpha/2)
        
        # Calculate prediction intervals
        risk_df['lower_bound'] = risk_df['predicted'] - z_value * std_error
        risk_df['upper_bound'] = risk_df['predicted'] + z_value * std_error
        
        # Calculate interval width
        risk_df['interval_width'] = risk_df['upper_bound'] - risk_df['lower_bound']
        
        # Check if actual value is within prediction interval
        if 'predicted' in risk_df.columns:
            target_col = [col for col in risk_df.columns if col not in 
                         ['predicted', 'error', 'lower_bound', 'upper_bound', 'interval_width']][0]
            risk_df['within_interval'] = (
                (risk_df[target_col] >= risk_df['lower_bound']) & 
                (risk_df[target_col] <= risk_df['upper_bound'])
            )
        
        return risk_df
    
    def error_distribution(self, predictions_df):
        """
        Analyze the distribution of prediction errors.
        
        Parameters:
        -----------
        predictions_df : pandas.DataFrame
            DataFrame containing predictions and errors
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with error distribution analysis
        """
        # Make a copy of the predictions dataframe
        risk_df = predictions_df.copy()
        
        # Calculate absolute error
        risk_df['abs_error'] = np.abs(risk_df['error'])
        
        # Calculate relative error (percentage)
        target_col = [col for col in risk_df.columns if col not in 
                     ['predicted', 'error', 'abs_error']][0]
        risk_df['rel_error'] = (risk_df['abs_error'] / np.abs(risk_df[target_col])) * 100
        
        # Calculate z-score of errors
        risk_df['error_zscore'] = stats.zscore(risk_df['error'])
        
        # Classify errors based on z-score
        risk_df['error_severity'] = pd.cut(
            np.abs(risk_df['error_zscore']),
            bins=[0, 1, 2, 3, np.inf],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        return risk_df
    
    def outlier_detection(self, predictions_df, threshold=3.0):
        """
        Detect outliers in predictions and errors.
        
        Parameters:
        -----------
        predictions_df : pandas.DataFrame
            DataFrame containing predictions and errors
        threshold : float
            Z-score threshold for outlier detection (default: 3.0)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with outlier detection results
        """
        # Make a copy of the predictions dataframe
        risk_df = predictions_df.copy()
        
        # Calculate z-scores for predictions and actual values
        target_col = [col for col in risk_df.columns if col not in 
                     ['predicted', 'error']][0]
        
        risk_df['actual_zscore'] = stats.zscore(risk_df[target_col])
        risk_df['predicted_zscore'] = stats.zscore(risk_df['predicted'])
        risk_df['error_zscore'] = stats.zscore(risk_df['error'])
        
        # Flag outliers based on z-scores
        risk_df['actual_outlier'] = np.abs(risk_df['actual_zscore']) > threshold
        risk_df['predicted_outlier'] = np.abs(risk_df['predicted_zscore']) > threshold
        risk_df['error_outlier'] = np.abs(risk_df['error_zscore']) > threshold
        
        # Combine outlier flags
        risk_df['is_outlier'] = (
            risk_df['actual_outlier'] | 
            risk_df['predicted_outlier'] | 
            risk_df['error_outlier']
        )
        
        # Calculate outlier severity
        risk_df['outlier_severity'] = pd.cut(
            np.maximum.reduce([
                np.abs(risk_df['actual_zscore']),
                np.abs(risk_df['predicted_zscore']),
                np.abs(risk_df['error_zscore'])
            ]),
            bins=[0, threshold, threshold*1.5, threshold*2, np.inf],
            labels=['Normal', 'Mild', 'Moderate', 'Severe']
        )
        
        return risk_df
    
    def generate_risk_summary(self, risk_assessment, assessment_method):
        """
        Generate a summary of risk assessment results.
        
        Parameters:
        -----------
        risk_assessment : pandas.DataFrame
            DataFrame with risk assessment results
        assessment_method : str
            Method used for risk assessment
            
        Returns:
        --------
        dict
            Dictionary with risk summary statistics
        """
        summary = {}
        
        if assessment_method == "Prediction Intervals":
            # Calculate percentage of actual values within prediction intervals
            within_interval_pct = risk_assessment['within_interval'].mean() * 100
            summary['within_interval_pct'] = f"{within_interval_pct:.2f}%"
            
            # Calculate average interval width
            avg_interval_width = risk_assessment['interval_width'].mean()
            summary['avg_interval_width'] = f"{avg_interval_width:.2f}"
            
            return f"""
            **Prediction Intervals Summary:**
            - {summary['within_interval_pct']} of actual values fall within the prediction intervals
            - Average interval width: {summary['avg_interval_width']}
            """
            
        elif assessment_method == "Error Distribution":
            # Calculate error statistics
            mean_abs_error = risk_assessment['abs_error'].mean()
            median_abs_error = risk_assessment['abs_error'].median()
            mean_rel_error = risk_assessment['rel_error'].mean()
            
            # Count errors by severity
            error_counts = risk_assessment['error_severity'].value_counts()
            
            return f"""
            **Error Distribution Summary:**
            - Mean absolute error: {mean_abs_error:.2f}
            - Median absolute error: {median_abs_error:.2f}
            - Mean relative error: {mean_rel_error:.2f}%
            - Error severity counts:
              - Low: {error_counts.get('Low', 0)}
              - Medium: {error_counts.get('Medium', 0)}
              - High: {error_counts.get('High', 0)}
              - Critical: {error_counts.get('Critical', 0)}
            """
            
        elif assessment_method == "Outlier Detection":
            # Calculate outlier statistics
            outlier_count = risk_assessment['is_outlier'].sum()
            outlier_percentage = (outlier_count / len(risk_assessment)) * 100
            
            # Count outliers by severity
            outlier_severity = risk_assessment['outlier_severity'].value_counts()
            
            return f"""
            **Outlier Detection Summary:**
            - Total outliers detected: {outlier_count} ({outlier_percentage:.2f}%)
            - Outlier severity counts:
              - Mild: {outlier_severity.get('Mild', 0)}
              - Moderate: {outlier_severity.get('Moderate', 0)}
              - Severe: {outlier_severity.get('Severe', 0)}
            """
            
        return "No summary available for the selected assessment method."
