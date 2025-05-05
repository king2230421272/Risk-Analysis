import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

class Visualizer:
    """
    Utility for generating visualizations of data and results.
    """
    
    def __init__(self):
        """Initialize visualizer with default settings."""
        # Set default style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Set default figure size
        self.fig_size = (10, 6)
    
    def plot_histogram(self, data, column):
        """
        Plot a histogram of a column.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        column : str
            Column to plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot histogram with KDE
        sns.histplot(data[column], kde=True, ax=ax)
        
        # Set title and labels
        ax.set_title(f'Distribution of {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_scatter(self, data, x_column, y_column):
        """
        Plot a scatter plot of two columns.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        x_column : str
            Column for x-axis
        y_column : str
            Column for y-axis
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot scatter
        ax.scatter(data[x_column], data[y_column], alpha=0.6)
        
        # Add regression line
        try:
            sns.regplot(x=x_column, y=y_column, data=data, scatter=False, ax=ax)
        except:
            pass
        
        # Set title and labels
        ax.set_title(f'{y_column} vs {x_column}')
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_line(self, data, columns):
        """
        Plot a line chart of selected columns.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        columns : list
            Columns to plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot each column
        for column in columns:
            ax.plot(data.index, data[column], label=column)
        
        # Set title and labels
        ax.set_title('Line Chart')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_bar(self, data, x_column, y_column):
        """
        Plot a bar chart.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        x_column : str
            Column for x-axis (categories)
        y_column : str
            Column for y-axis (values)
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Aggregate data if needed
        if data[x_column].nunique() > 15:
            # Too many categories, use top categories
            top_categories = data[x_column].value_counts().head(15).index
            filtered_data = data[data[x_column].isin(top_categories)]
            plot_data = filtered_data.groupby(x_column)[y_column].mean().reset_index()
            title_suffix = " (Top 15 Categories)"
        else:
            # Aggregate data by category
            plot_data = data.groupby(x_column)[y_column].mean().reset_index()
            title_suffix = ""
        
        # Plot bar chart
        sns.barplot(x=x_column, y=y_column, data=plot_data, ax=ax)
        
        # Set title and labels
        ax.set_title(f'Average {y_column} by {x_column}{title_suffix}')
        ax.set_xlabel(x_column)
        ax.set_ylabel(f'Average {y_column}')
        
        # Rotate x-axis labels if needed
        if plot_data[x_column].nunique() > 5:
            plt.xticks(rotation=45, ha='right')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_correlation(self, data):
        """
        Plot a correlation matrix of numerical columns.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Get numerical columns only
        numeric_data = data.select_dtypes(include=np.number)
        
        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax, fmt='.2f')
        
        # Set title
        ax.set_title('Correlation Matrix')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_box(self, data, column):
        """
        Plot a box plot of a column.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Input data
        column : str
            Column to plot
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot box plot
        sns.boxplot(y=data[column], ax=ax)
        
        # Set title and labels
        ax.set_title(f'Box Plot of {column}')
        ax.set_ylabel(column)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_feature_importance(self, feature_names, feature_importance):
        """
        Plot feature importance.
        
        Parameters:
        -----------
        feature_names : list
            Names of features
        feature_importance : numpy.ndarray
            Importance values
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create a DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Plot horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        
        # Set title and labels
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_prediction_intervals(self, risk_assessment):
        """
        Plot predictions with confidence intervals.
        
        Parameters:
        -----------
        risk_assessment : pandas.DataFrame
            DataFrame with prediction intervals
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Identify target column (not predicted, error, or interval columns)
        target_col = [col for col in risk_assessment.columns if col not in 
                     ['predicted', 'error', 'lower_bound', 'upper_bound', 
                      'interval_width', 'within_interval']][0]
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        # Sort data by actual values for better visualization
        sorted_data = risk_assessment.sort_values(target_col).reset_index(drop=True)
        
        # Plot actual values and predictions
        ax.plot(sorted_data.index, sorted_data[target_col], 'o-', label='Actual')
        ax.plot(sorted_data.index, sorted_data['predicted'], 'o-', label='Predicted')
        
        # Plot confidence intervals
        ax.fill_between(
            sorted_data.index,
            sorted_data['lower_bound'],
            sorted_data['upper_bound'],
            alpha=0.2,
            label='Prediction Interval'
        )
        
        # Set title and labels
        ax.set_title('Predictions with Confidence Intervals')
        ax.set_xlabel('Index')
        ax.set_ylabel('Value')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_error_distribution(self, risk_assessment):
        """
        Plot the distribution of prediction errors.
        
        Parameters:
        -----------
        risk_assessment : pandas.DataFrame
            DataFrame with error distribution
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot error distribution
        sns.histplot(risk_assessment['error'], kde=True, ax=ax1)
        ax1.set_title('Distribution of Prediction Errors')
        ax1.set_xlabel('Error')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Plot severity of errors
        error_counts = risk_assessment['error_severity'].value_counts().sort_index()
        ax2.bar(error_counts.index, error_counts.values, color=sns.color_palette("coolwarm", len(error_counts)))
        ax2.set_title('Error Severity')
        ax2.set_xlabel('Severity')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_outlier_detection(self, risk_assessment):
        """
        Plot outlier detection results.
        
        Parameters:
        -----------
        risk_assessment : pandas.DataFrame
            DataFrame with outlier detection results
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object
        """
        # Identify target column
        target_col = [col for col in risk_assessment.columns if col not in 
                     ['predicted', 'error', 'actual_zscore', 'predicted_zscore', 
                      'error_zscore', 'actual_outlier', 'predicted_outlier', 
                      'error_outlier', 'is_outlier', 'outlier_severity']][0]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot actual vs predicted with outliers highlighted
        normal_points = risk_assessment[~risk_assessment['is_outlier']]
        outlier_points = risk_assessment[risk_assessment['is_outlier']]
        
        ax1.scatter(normal_points[target_col], normal_points['predicted'], 
                   alpha=0.6, label='Normal')
        ax1.scatter(outlier_points[target_col], outlier_points['predicted'], 
                   color='red', alpha=0.6, label='Outlier')
        
        # Add diagonal line (perfect predictions)
        lims = [
            min(ax1.get_xlim()[0], ax1.get_ylim()[0]),
            max(ax1.get_xlim()[1], ax1.get_ylim()[1])
        ]
        ax1.plot(lims, lims, 'k--', alpha=0.5)
        
        ax1.set_title('Actual vs Predicted with Outliers')
        ax1.set_xlabel('Actual')
        ax1.set_ylabel('Predicted')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot outlier severity
        severity_counts = risk_assessment['outlier_severity'].value_counts().sort_index()
        ax2.bar(severity_counts.index, severity_counts.values, 
               color=sns.color_palette("coolwarm", len(severity_counts)))
        ax2.set_title('Outlier Severity')
        ax2.set_xlabel('Severity')
        ax2.set_ylabel('Count')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
