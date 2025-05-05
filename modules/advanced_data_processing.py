"""
Advanced Data Processing Module

This module provides advanced data processing capabilities including:
1. MCMC-based interpolation
2. Conditional Generative Adversarial Network (CGAN) training and prediction
3. Isolated Forest Outlier Detection
4. K-S Distribution Detection
5. Spearman Rank Correlation analysis
6. Permutation testing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

class AdvancedDataProcessor:
    """Advanced data processing with sophisticated interpolation and analysis methods."""
    
    def __init__(self):
        """Initialize the advanced data processor."""
        self.mcmc_samples = None
        self.cgan_model = None
        self.outliers_mask = None
        self.ks_test_results = None
        self.spearman_results = None
        self.permutation_results = None
        self.original_data = None
        self.interpolated_data = None
        self.processed_data = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def preprocess_data(self, data):
        """
        Preprocess the data for analysis.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            The input data
            
        Returns:
        --------
        pandas.DataFrame
            Preprocessed data
        """
        # Basic preprocessing
        numeric_cols = data.select_dtypes(include=np.number).columns
        categorical_cols = data.select_dtypes(exclude=np.number).columns
        
        # Create a copy to avoid modifying the original data
        processed_data = data.copy()
        
        # Handle categorical variables
        if len(categorical_cols) > 0:
            processed_data = pd.get_dummies(processed_data, columns=categorical_cols)
        
        return processed_data
    
    def mcmc_interpolation(self, data, num_samples=500, chains=2):
        """
        Use Markov Chain Monte Carlo to interpolate missing values.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data with missing values
        num_samples : int
            Number of samples to draw
        chains : int
            Number of chains to run
            
        Returns:
        --------
        pandas.DataFrame
            Data with interpolated values
        """
        print("Starting MCMC interpolation...")
        processed_data = self.preprocess_data(data)
        missing_mask = processed_data.isna()
        
        # Only process columns with missing values
        cols_with_missing = [col for col in processed_data.columns if processed_data[col].isna().any()]
        
        if not cols_with_missing:
            print("No missing values found in the dataset.")
            return processed_data
        
        # Only consider numeric columns for now as they're easier to model
        numeric_cols_with_missing = [col for col in cols_with_missing 
                                    if np.issubdtype(processed_data[col].dtype, np.number)]
        
        if not numeric_cols_with_missing:
            print("No missing values in numeric columns.")
            return processed_data
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = pd.DataFrame(
            scaler.fit_transform(processed_data[numeric_cols_with_missing].fillna(0)),
            columns=numeric_cols_with_missing
        )
        
        # Create a PyMC model
        with pm.Model() as model:
            # Create variables for each column with missing values
            variables = {}
            for col in numeric_cols_with_missing:
                # Define priors based on the observed data distribution
                mu = pm.Normal(f'mu_{col}', mu=0, sigma=1)
                sigma = pm.HalfNormal(f'sigma_{col}', sigma=1)
                
                # Define the likelihood for observed and missing values
                variables[col] = pm.Normal(
                    col, 
                    mu=mu, 
                    sigma=sigma, 
                    observed=scaled_data[col].where(~missing_mask[col])
                )
            
            # Sample from the posterior
            self.mcmc_samples = pm.sample(num_samples, chains=chains, progressbar=True)
        
        # Extract interpolated values
        interpolated_data = processed_data.copy()
        
        for col in numeric_cols_with_missing:
            # Extract the posterior samples for this column
            col_samples = self.mcmc_samples.posterior[col].values
            
            # Take the mean of the posterior as our point estimate
            col_samples_mean = col_samples.mean(axis=(0, 1))
            
            # Scale back to original range
            col_samples_mean_rescaled = scaler.inverse_transform(
                np.column_stack([col_samples_mean if c == col else np.zeros(len(col_samples_mean)) 
                               for c in numeric_cols_with_missing])
            )[:, numeric_cols_with_missing.index(col)]
            
            # Replace missing values
            missing_indices = interpolated_data[col].isna()
            interpolated_data.loc[missing_indices, col] = col_samples_mean_rescaled[missing_indices]
        
        print("MCMC interpolation completed.")
        return interpolated_data
    
    class Generator(nn.Module):
        """Generator network for CGAN."""
        def __init__(self, input_dim, condition_dim, output_dim, hidden_dim=128):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim + condition_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 2, hidden_dim * 4),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim * 4, output_dim),
                nn.Tanh()
            )
            
        def forward(self, noise, condition):
            # Concatenate noise and condition
            x = torch.cat([noise, condition], 1)
            return self.model(x)
    
    class Discriminator(nn.Module):
        """Discriminator network for CGAN."""
        def __init__(self, input_dim, condition_dim, hidden_dim=128):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim + condition_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x, condition):
            # Concatenate input and condition
            x = torch.cat([x, condition], 1)
            return self.model(x)
    
    def train_cgan(self, original_data, condition_cols, target_cols, epochs=200, batch_size=32, noise_dim=100):
        """
        Train a Conditional Generative Adversarial Network.
        
        Parameters:
        -----------
        original_data : pandas.DataFrame
            The original non-interpolated data
        condition_cols : list
            Column names to use as conditions
        target_cols : list
            Column names to generate/predict
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        noise_dim : int
            Dimension of the input noise vector
            
        Returns:
        --------
        tuple
            (Generator, Discriminator)
        """
        print("Starting CGAN training...")
        
        # Preprocess the data
        data = self.preprocess_data(original_data)
        
        # Extract condition and target features
        condition_data = data[condition_cols].values
        target_data = data[target_cols].values
        
        # Scale the data to [-1, 1] for tanh
        condition_scaler = MinMaxScaler(feature_range=(-1, 1))
        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        scaled_condition = condition_scaler.fit_transform(condition_data)
        scaled_target = target_scaler.fit_transform(target_data)
        
        # Convert to PyTorch tensors
        condition_tensor = torch.FloatTensor(scaled_condition).to(self.device)
        target_tensor = torch.FloatTensor(scaled_target).to(self.device)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(condition_tensor, target_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Instantiate models
        generator = self.Generator(noise_dim, len(condition_cols), len(target_cols)).to(self.device)
        discriminator = self.Discriminator(len(target_cols), len(condition_cols)).to(self.device)
        
        # Optimizers
        g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Training loop
        for epoch in range(epochs):
            for i, (condition, real_target) in enumerate(dataloader):
                batch_size = condition.size(0)
                
                # Ground truths
                real_label = torch.ones(batch_size, 1).to(self.device)
                fake_label = torch.zeros(batch_size, 1).to(self.device)
                
                # ---------------------
                # Train Discriminator
                # ---------------------
                d_optimizer.zero_grad()
                
                # Real samples
                d_real_output = discriminator(real_target, condition)
                d_real_loss = criterion(d_real_output, real_label)
                
                # Fake samples
                noise = torch.randn(batch_size, noise_dim).to(self.device)
                fake_target = generator(noise, condition)
                d_fake_output = discriminator(fake_target.detach(), condition)
                d_fake_loss = criterion(d_fake_output, fake_label)
                
                # Combined loss
                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                d_optimizer.step()
                
                # ---------------------
                # Train Generator
                # ---------------------
                g_optimizer.zero_grad()
                
                # Generate fake samples
                noise = torch.randn(batch_size, noise_dim).to(self.device)
                fake_target = generator(noise, condition)
                d_output = discriminator(fake_target, condition)
                
                # Calculate loss
                g_loss = criterion(d_output, real_label)
                g_loss.backward()
                g_optimizer.step()
            
            # Print progress
            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
        
        print("CGAN training completed.")
        
        # Save the models
        self.cgan_model = {
            'generator': generator,
            'discriminator': discriminator,
            'condition_scaler': condition_scaler,
            'target_scaler': target_scaler,
            'condition_cols': condition_cols,
            'target_cols': target_cols,
            'noise_dim': noise_dim
        }
        
        return generator, discriminator
    
    def cgan_analysis(self, interpolated_data, noise_samples=100):
        """
        Use trained CGAN model to analyze interpolated data.
        
        Parameters:
        -----------
        interpolated_data : pandas.DataFrame
            The interpolated data to analyze
        noise_samples : int
            Number of noise samples to generate for each condition
            
        Returns:
        --------
        pandas.DataFrame
            Data with CGAN analysis results
        """
        if self.cgan_model is None:
            raise ValueError("CGAN model not trained. Please call train_cgan first.")
        
        print("Starting CGAN analysis...")
        
        # Extract components from the saved model
        generator = self.cgan_model['generator']
        condition_scaler = self.cgan_model['condition_scaler']
        target_scaler = self.cgan_model['target_scaler']
        condition_cols = self.cgan_model['condition_cols']
        target_cols = self.cgan_model['target_cols']
        noise_dim = self.cgan_model['noise_dim']
        
        # Preprocess the interpolated data
        data = self.preprocess_data(interpolated_data)
        
        # Extract condition data
        condition_data = data[condition_cols].values
        
        # Scale the condition data
        scaled_condition = condition_scaler.transform(condition_data)
        
        # Convert to PyTorch tensor
        condition_tensor = torch.FloatTensor(scaled_condition).to(self.device)
        
        # Set generator to evaluation mode
        generator.eval()
        
        # Generate multiple samples for each condition
        all_generations = []
        
        with torch.no_grad():
            for _ in range(noise_samples):
                # Generate random noise
                noise = torch.randn(condition_tensor.size(0), noise_dim).to(self.device)
                
                # Generate fake samples
                fake_target = generator(noise, condition_tensor)
                
                # Convert back to numpy
                fake_target_np = fake_target.cpu().numpy()
                
                # Inverse transform to original scale
                fake_target_rescaled = target_scaler.inverse_transform(fake_target_np)
                
                all_generations.append(fake_target_rescaled)
        
        # Stack all generations
        all_generations_stacked = np.stack(all_generations, axis=0)
        
        # Calculate statistics
        mean_generations = np.mean(all_generations_stacked, axis=0)
        std_generations = np.std(all_generations_stacked, axis=0)
        
        # Create a DataFrame for the CGAN analysis results
        cgan_results = pd.DataFrame()
        
        for i, col in enumerate(target_cols):
            cgan_results[f'{col}_mean'] = mean_generations[:, i]
            cgan_results[f'{col}_std'] = std_generations[:, i]
            
            # Add original values for comparison
            if col in interpolated_data.columns:
                cgan_results[f'{col}_original'] = interpolated_data[col].values
                
                # Calculate deviation from CGAN prediction
                cgan_results[f'{col}_deviation'] = np.abs(
                    interpolated_data[col].values - mean_generations[:, i]
                )
        
        print("CGAN analysis completed.")
        return cgan_results
    
    def isolated_forest_detection(self, data, contamination=0.05):
        """
        Perform Isolated Forest outlier detection.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data to analyze for outliers
        contamination : float
            Expected proportion of outliers in the dataset
            
        Returns:
        --------
        pandas.DataFrame
            Data with outlier detection results
        """
        print("Starting Isolated Forest outlier detection...")
        
        # Preprocess the data
        processed_data = self.preprocess_data(data)
        
        # Select only numeric columns
        numeric_data = processed_data.select_dtypes(include=np.number)
        
        # Create and fit the Isolation Forest model
        isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Fit the model and predict
        self.outliers_mask = isolation_forest.fit_predict(numeric_data)
        
        # Convert to binary mask (1 for inliers, 0 for outliers)
        outliers_binary = np.where(self.outliers_mask == -1, 0, 1)
        
        # Calculate anomaly scores (higher score = more anomalous)
        anomaly_scores = isolation_forest.decision_function(numeric_data)
        anomaly_scores = -anomaly_scores  # Invert so higher = more anomalous
        
        # Create results DataFrame
        results = data.copy()
        results['is_outlier'] = outliers_binary == 0
        results['anomaly_score'] = anomaly_scores
        
        # Count outliers
        outlier_count = sum(results['is_outlier'])
        print(f"Isolated Forest detection completed. Found {outlier_count} outliers.")
        
        return results
    
    def ks_distribution_test(self, data1, data2, alpha=0.05):
        """
        Perform Kolmogorov-Smirnov distribution test between two datasets.
        
        Parameters:
        -----------
        data1 : pandas.DataFrame
            First dataset
        data2 : pandas.DataFrame
            Second dataset
        alpha : float
            Significance level
            
        Returns:
        --------
        pandas.DataFrame
            Results of the KS test
        """
        print("Starting Kolmogorov-Smirnov distribution testing...")
        
        # Get common numeric columns
        numeric_cols1 = data1.select_dtypes(include=np.number).columns
        numeric_cols2 = data2.select_dtypes(include=np.number).columns
        common_cols = list(set(numeric_cols1).intersection(set(numeric_cols2)))
        
        if not common_cols:
            raise ValueError("No common numeric columns found between the datasets")
        
        # Perform KS test for each column
        results = []
        
        for col in common_cols:
            # Get data without NaNs
            x = data1[col].dropna().values
            y = data2[col].dropna().values
            
            if len(x) == 0 or len(y) == 0:
                results.append({
                    'column': col,
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'significant': False
                })
                continue
            
            # Perform KS test
            statistic, p_value = stats.ks_2samp(x, y)
            
            results.append({
                'column': col,
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < alpha
            })
        
        # Create a DataFrame from the results
        results_df = pd.DataFrame(results)
        
        # Apply multiple testing correction (Benjamini-Hochberg)
        if len(results_df) > 0 and not results_df['p_value'].isna().all():
            _, corrected_p, _, _ = multipletests(
                results_df['p_value'].fillna(1),
                alpha=alpha, 
                method='fdr_bh'
            )
            results_df['corrected_p_value'] = corrected_p
            results_df['significant_corrected'] = results_df['corrected_p_value'] < alpha
        
        self.ks_test_results = results_df
        print("Kolmogorov-Smirnov testing completed.")
        return results_df
    
    def spearman_correlation(self, data1, data2, alpha=0.05):
        """
        Perform Spearman rank correlation analysis between two datasets.
        
        Parameters:
        -----------
        data1 : pandas.DataFrame
            First dataset
        data2 : pandas.DataFrame
            Second dataset
        alpha : float
            Significance level
            
        Returns:
        --------
        pandas.DataFrame
            Results of the Spearman correlation
        """
        print("Starting Spearman rank correlation analysis...")
        
        # Get common numeric columns
        numeric_cols1 = data1.select_dtypes(include=np.number).columns
        numeric_cols2 = data2.select_dtypes(include=np.number).columns
        common_cols = list(set(numeric_cols1).intersection(set(numeric_cols2)))
        
        if not common_cols:
            raise ValueError("No common numeric columns found between the datasets")
        
        # Perform Spearman correlation for each column
        results = []
        
        for col in common_cols:
            # Get matched pairs without NaNs
            df = pd.DataFrame({
                'data1': data1[col],
                'data2': data2[col]
            }).dropna()
            
            if len(df) == 0:
                results.append({
                    'column': col,
                    'correlation': np.nan,
                    'p_value': np.nan,
                    'significant': False
                })
                continue
            
            # Perform Spearman correlation
            correlation, p_value = stats.spearmanr(df['data1'], df['data2'])
            
            results.append({
                'column': col,
                'correlation': correlation,
                'p_value': p_value,
                'significant': p_value < alpha
            })
        
        # Create a DataFrame from the results
        results_df = pd.DataFrame(results)
        
        # Apply multiple testing correction (Benjamini-Hochberg)
        if len(results_df) > 0 and not results_df['p_value'].isna().all():
            _, corrected_p, _, _ = multipletests(
                results_df['p_value'].fillna(1),
                alpha=alpha, 
                method='fdr_bh'
            )
            results_df['corrected_p_value'] = corrected_p
            results_df['significant_corrected'] = results_df['corrected_p_value'] < alpha
        
        self.spearman_results = results_df
        print("Spearman correlation analysis completed.")
        return results_df
    
    def permutation_test(self, data1, data2, num_permutations=1000, alpha=0.05):
        """
        Perform permutation test between two datasets.
        
        Parameters:
        -----------
        data1 : pandas.DataFrame
            First dataset
        data2 : pandas.DataFrame
            Second dataset
        num_permutations : int
            Number of permutations to perform
        alpha : float
            Significance level
            
        Returns:
        --------
        pandas.DataFrame
            Results of the permutation test
        """
        print("Starting permutation test analysis...")
        
        # Get common numeric columns
        numeric_cols1 = data1.select_dtypes(include=np.number).columns
        numeric_cols2 = data2.select_dtypes(include=np.number).columns
        common_cols = list(set(numeric_cols1).intersection(set(numeric_cols2)))
        
        if not common_cols:
            raise ValueError("No common numeric columns found between the datasets")
        
        # Perform permutation test for each column
        results = []
        
        for col in common_cols:
            # Get data without NaNs
            x = data1[col].dropna().values
            y = data2[col].dropna().values
            
            if len(x) < 5 or len(y) < 5:  # Need reasonable sample sizes
                results.append({
                    'column': col,
                    'observed_diff': np.nan,
                    'p_value': np.nan,
                    'significant': False
                })
                continue
            
            # Calculate observed difference in means
            observed_diff = np.abs(np.mean(x) - np.mean(y))
            
            # Combine data for permutation
            combined = np.concatenate([x, y])
            
            # Perform permutation test
            count = 0
            for _ in range(num_permutations):
                # Shuffle the combined data
                np.random.shuffle(combined)
                
                # Split into two groups of the original sizes
                perm_x = combined[:len(x)]
                perm_y = combined[len(x):]
                
                # Calculate permuted difference
                perm_diff = np.abs(np.mean(perm_x) - np.mean(perm_y))
                
                # Count how many permuted differences are >= observed
                if perm_diff >= observed_diff:
                    count += 1
            
            # Calculate p-value
            p_value = count / num_permutations
            
            results.append({
                'column': col,
                'observed_diff': observed_diff,
                'p_value': p_value,
                'significant': p_value < alpha
            })
        
        # Create a DataFrame from the results
        results_df = pd.DataFrame(results)
        
        # Apply multiple testing correction (Benjamini-Hochberg)
        if len(results_df) > 0 and not results_df['p_value'].isna().all():
            _, corrected_p, _, _ = multipletests(
                results_df['p_value'].fillna(1),
                alpha=alpha, 
                method='fdr_bh'
            )
            results_df['corrected_p_value'] = corrected_p
            results_df['significant_corrected'] = results_df['corrected_p_value'] < alpha
        
        self.permutation_results = results_df
        print("Permutation test analysis completed.")
        return results_df
    
    def process_advanced_pipeline(self, original_data, interpolated_data=None):
        """
        Process the full advanced analysis pipeline.
        
        Parameters:
        -----------
        original_data : pandas.DataFrame
            The original non-interpolated data
        interpolated_data : pandas.DataFrame, optional
            Pre-interpolated data (if None, will be created through MCMC)
            
        Returns:
        --------
        dict
            Dictionary containing all analysis results
        """
        results = {}
        
        # Store original data
        self.original_data = original_data
        
        # 1. MCMC interpolation
        if interpolated_data is None:
            self.interpolated_data = self.mcmc_interpolation(original_data)
        else:
            self.interpolated_data = interpolated_data
        
        results['interpolated_data'] = self.interpolated_data
        
        # 2. Independent analysis of interpolated data (basic stats)
        interpolated_stats = self.interpolated_data.describe()
        results['interpolated_stats'] = interpolated_stats
        
        # 3. Train CGAN on original data
        # Identify potential condition and target columns
        numeric_cols = original_data.select_dtypes(include=np.number).columns.tolist()
        
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns to train CGAN")
        
        # Simple heuristic: use half columns as conditions, half as targets
        # In a real app, this would be user-selected
        mid_point = len(numeric_cols) // 2
        condition_cols = numeric_cols[:mid_point]
        target_cols = numeric_cols[mid_point:]
        
        # Train the CGAN
        generator, discriminator = self.train_cgan(
            original_data, 
            condition_cols=condition_cols, 
            target_cols=target_cols
        )
        
        # 4. Use CGAN to analyze interpolated data
        cgan_results = self.cgan_analysis(self.interpolated_data)
        results['cgan_analysis'] = cgan_results
        
        # 5. Perform Isolated Forest outlier detection
        outlier_results = self.isolated_forest_detection(cgan_results)
        results['outlier_detection'] = outlier_results
        
        # 6. Perform K-S distribution test
        ks_results = self.ks_distribution_test(original_data, outlier_results)
        results['ks_test'] = ks_results
        
        # 7. Perform Spearman correlation and permutation test
        spearman_results = self.spearman_correlation(original_data, outlier_results)
        results['spearman_correlation'] = spearman_results
        
        permutation_results = self.permutation_test(original_data, outlier_results)
        results['permutation_test'] = permutation_results
        
        # 8. Final processed data (store for download)
        self.processed_data = outlier_results.copy()
        self.processed_data['is_valid'] = ~self.processed_data['is_outlier']
        
        # Add analysis metrics
        for col in original_data.columns:
            if col in numeric_cols:
                # Find index in spearman results
                if col in spearman_results['column'].values:
                    idx = spearman_results[spearman_results['column'] == col].index[0]
                    self.processed_data[f'{col}_correlation'] = spearman_results.loc[idx, 'correlation']
        
        results['processed_data'] = self.processed_data
        
        return results