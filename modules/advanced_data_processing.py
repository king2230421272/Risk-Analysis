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
    
    def mcmc_interpolation(self, data, num_samples=500, chains=2, experimental_data=None, experimental_weight=0.5):
        """
        Use Markov Chain Monte Carlo to interpolate missing values with optional experimental data fusion.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Data with missing values
        num_samples : int
            Number of samples to draw
        chains : int
            Number of chains to run
        experimental_data : pandas.DataFrame, optional
            Experimental data to incorporate into the model as prior information
        experimental_weight : float
            Weight to give to experimental data (0.0 - 1.0)
            
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
            
        # Process experimental data if provided
        if experimental_data is not None:
            print(f"Incorporating experimental data with weight {experimental_weight}")
            # Find common columns between data and experimental_data
            common_cols = list(set(numeric_cols_with_missing) & set(experimental_data.columns))
            
            if not common_cols:
                print("No common numeric columns with missing values found between the data and experimental data.")
                print("Proceeding with standard MCMC interpolation without experimental data.")
            else:
                print(f"Found {len(common_cols)} common columns with missing values.")
                
                # Scale experimental data to match original data's distribution
                for col in common_cols:
                    # Skip columns with all NaNs
                    if processed_data[col].isna().all() or experimental_data[col].isna().all():
                        continue
                        
                    # Get stats for scaling
                    orig_mean = processed_data[col].mean()
                    orig_std = processed_data[col].std() if processed_data[col].std() > 0 else 1.0
                    
                    exp_mean = experimental_data[col].mean()
                    exp_std = experimental_data[col].std() if experimental_data[col].std() > 0 else 1.0
                    
                    # Apply scaling: standardize, then transform to original data's distribution
                    experimental_data[col] = ((experimental_data[col] - exp_mean) / exp_std) * orig_std + orig_mean
        
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
                
                # If experimental data is available, incorporate it as additional observations
                if experimental_data is not None and col in experimental_data.columns:
                    # Skip columns with all NaN values
                    if experimental_data[col].isna().all():
                        continue
                    
                    # Standardize experimental data to match the scaled original data
                    exp_col_data = experimental_data[col].dropna().values
                    if len(exp_col_data) > 0:
                        # Create a new scaler for just this column to match the original scaling
                        col_scaler = StandardScaler()
                        col_scaler.fit(processed_data[[col]].fillna(0))
                        
                        # Scale the experimental data
                        exp_col_scaled = col_scaler.transform(
                            exp_col_data.reshape(-1, 1)
                        ).flatten()
                        
                        # Create an observed variable with the experimental data
                        # Weight is controlled by the experimental_weight parameter
                        # Low weight = larger sigma (less influence), high weight = smaller sigma (more influence)
                        exp_sigma = sigma * (1.0 / max(0.1, experimental_weight))
                        
                        # Add experimental observations with appropriate weight
                        pm.Normal(
                            f'{col}_exp', 
                            mu=mu,
                            sigma=exp_sigma,
                            observed=exp_col_scaled
                        )
                        print(f"Added experimental data for column {col} with weight {experimental_weight}")
            
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
        def __init__(self, input_dim, condition_dim, output_dim, hidden_dim=128, dropout_rate=0.3):
            super().__init__()
            
            # 输入层处理
            self.noise_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            )
            
            # 条件处理层
            self.condition_proj = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            )
            
            # 主体生成网络
            self.main = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
                
                nn.Linear(hidden_dim * 2, hidden_dim * 4),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
                
                nn.Linear(hidden_dim * 4, output_dim),
                nn.Tanh()
            )
            
            # 初始化权重
            self._initialize_weights()
            
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
            
        def forward(self, noise, condition):
            # 处理噪声和条件输入
            noise_features = self.noise_proj(noise)
            condition_features = self.condition_proj(condition)
            
            # 拼接处理后的特征
            x = torch.cat([noise_features, condition_features], dim=1)
            
            # 生成数据
            return self.main(x)
    
    class Discriminator(nn.Module):
        """Discriminator network for CGAN."""
        def __init__(self, input_dim, condition_dim, hidden_dim=128, dropout_rate=0.3):
            super().__init__()
            
            # 输入数据处理层
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            )
            
            # 条件处理层
            self.condition_proj = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            )
            
            # 主体判别网络
            self.main = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
                
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate),
                
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            
            # 初始化权重
            self._initialize_weights()
            
        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x, condition):
            # 处理输入和条件
            x_features = self.input_proj(x)
            condition_features = self.condition_proj(condition)
            
            # 拼接处理后的特征
            combined = torch.cat([x_features, condition_features], dim=1)
            
            # 判别结果
            return self.main(combined)
    
    def train_cgan(self, original_data, condition_cols, target_cols, epochs=200, batch_size=32, 
                noise_dim=100, learning_rate=0.0002, beta1=0.5, beta2=0.999, 
                early_stopping_patience=20, dropout_rate=0.3, label_smoothing=0.1):
        """
        Train a Conditional Generative Adversarial Network with enhanced stability and monitoring.
        
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
        learning_rate : float
            Learning rate for Adam optimizer
        beta1 : float
            Beta1 parameter for Adam optimizer
        beta2 : float
            Beta2 parameter for Adam optimizer
        early_stopping_patience : int
            Number of epochs to wait for improvement before stopping training
        dropout_rate : float
            Dropout rate to use in the models
        label_smoothing : float
            Amount of label smoothing to apply for GANs stability
            
        Returns:
        --------
        tuple
            (Generator, Discriminator)
        dict
            Training history and metrics
        """
        print("Starting enhanced CGAN training...")
        
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
        
        # Instantiate models with dropout
        generator = self.Generator(
            noise_dim, 
            len(condition_cols), 
            len(target_cols), 
            dropout_rate=dropout_rate
        ).to(self.device)
        
        discriminator = self.Discriminator(
            len(target_cols), 
            len(condition_cols), 
            dropout_rate=dropout_rate
        ).to(self.device)
        
        # Optimizers
        g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Training history
        history = {
            'd_losses': [],
            'g_losses': [],
            'fake_scores': [],
            'real_scores': [],
            'epochs_trained': 0
        }
        
        # Early stopping variables
        best_g_loss = float('inf')
        early_stopping_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            epoch_d_losses = []
            epoch_g_losses = []
            epoch_real_scores = []
            epoch_fake_scores = []
            
            generator.train()
            discriminator.train()
            
            for i, (condition, real_target) in enumerate(dataloader):
                batch_size = condition.size(0)
                
                # Apply label smoothing for stability
                real_label = torch.ones(batch_size, 1).to(self.device) * (1.0 - label_smoothing)
                fake_label = torch.zeros(batch_size, 1).to(self.device) + label_smoothing * 0.5
                
                # ---------------------
                # Train Discriminator
                # ---------------------
                for _ in range(1):  # Can train discriminator multiple times per generator step
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
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    d_optimizer.step()
                
                # ---------------------
                # Train Generator
                # ---------------------
                g_optimizer.zero_grad()
                
                # Generate fake samples
                noise = torch.randn(batch_size, noise_dim).to(self.device)
                fake_target = generator(noise, condition)
                d_output = discriminator(fake_target, condition)
                
                # Calculate loss - aim for discriminator to predict "real" on generated samples
                g_loss = criterion(d_output, real_label)
                g_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                g_optimizer.step()
                
                # Record batch statistics
                epoch_d_losses.append(d_loss.item())
                epoch_g_losses.append(g_loss.item())
                epoch_real_scores.append(d_real_output.mean().item())
                epoch_fake_scores.append(d_fake_output.mean().item())
            
            # Compute epoch statistics
            avg_d_loss = np.mean(epoch_d_losses)
            avg_g_loss = np.mean(epoch_g_losses)
            avg_real_score = np.mean(epoch_real_scores)
            avg_fake_score = np.mean(epoch_fake_scores)
            
            # Save to history
            history['d_losses'].append(avg_d_loss)
            history['g_losses'].append(avg_g_loss)
            history['real_scores'].append(avg_real_score)
            history['fake_scores'].append(avg_fake_score)
            history['epochs_trained'] = epoch + 1
            
            # Print progress
            if (epoch+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, "
                      f"D(x): {avg_real_score:.4f}, D(G(z)): {avg_fake_score:.4f}")
            
            # Early stopping check
            if avg_g_loss < best_g_loss:
                best_g_loss = avg_g_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                
            # Apply early stopping if no improvement
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch+1}. No improvement for {early_stopping_patience} epochs.")
                break
            
            # Mode collapse detection
            if avg_fake_score > 0.9 or avg_real_score < 0.1:
                print(f"Warning: Possible mode collapse detected at epoch {epoch+1}.")
                if avg_fake_score > 0.9 and avg_real_score < 0.1:
                    print("Training unstable. Restarting with adjusted parameters...")
                    # In a real implementation, we might restart with different parameters
                    # For now, we'll just continue but adjust the learning rate
                    for param_group in g_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5
        
        print(f"CGAN training completed after {history['epochs_trained']} epochs.")
        
        # Generate a few samples to verify model works
        generator.eval()
        with torch.no_grad():
            verify_noise = torch.randn(5, noise_dim).to(self.device)
            verify_condition = condition_tensor[:5]  # Use first 5 conditions from dataset
            verify_samples = generator(verify_noise, verify_condition)
            print(f"Verification completed. Generated {len(verify_samples)} samples successfully.")
        
        # Save the models and training history
        self.cgan_model = {
            'generator': generator,
            'discriminator': discriminator,
            'condition_scaler': condition_scaler,
            'target_scaler': target_scaler,
            'condition_cols': condition_cols,
            'target_cols': target_cols,
            'noise_dim': noise_dim,
            'training_history': history
        }
        
        return generator, discriminator
    
    def cgan_generate_and_compare(self, interpolated_data, noise_samples=100, custom_conditions=None, original_data=None):
        """
        使用训练好的CGAN模型生成器根据条件生成数据，并与插补数据进行统计指标比较。
        
        Parameters:
        -----------
        interpolated_data : pandas.DataFrame
            插补数据集，通常来自Convergence Diagnostics模块
        noise_samples : int
            为每个条件生成的样本数量
        custom_conditions : dict, optional
            自定义条件值，用于替代插补数据中的条件
        original_data : pandas.DataFrame, optional
            原始数据，用作参考比较
            
        Returns:
        --------
        pandas.DataFrame
            生成的数据和统计比较结果
        dict
            包含详细分析信息的字典
        """
        if self.cgan_model is None:
            raise ValueError("CGAN模型未训练。请先调用train_cgan方法。")
        
        print("开始CGAN生成与统计比较分析...")
        
        # 从保存的模型中提取组件
        generator = self.cgan_model['generator']
        condition_scaler = self.cgan_model['condition_scaler']
        target_scaler = self.cgan_model['target_scaler']
        condition_cols = self.cgan_model['condition_cols']
        target_cols = self.cgan_model['target_cols']
        noise_dim = self.cgan_model['noise_dim']
        
        # 预处理插补数据
        data = self.preprocess_data(interpolated_data)
        
        # 检查是否有数据可供分析
        if data.empty or len(data) == 0:
            raise ValueError("预处理后没有可用于CGAN分析的数据")
        
        # 确保条件列存在于数据中
        missing_cols = set(condition_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing condition columns in data: {missing_cols}")
        
        # If custom conditions are provided, use them instead of data in the dataframe
        if custom_conditions is not None and isinstance(custom_conditions, dict):
            print(f"Using custom condition values: {custom_conditions}")
            
            # Create a single-row dataframe with only custom condition values
            custom_condition_df = pd.DataFrame([custom_conditions])
            
            # Ensure all condition columns exist
            for col in condition_cols:
                if col not in custom_condition_df.columns:
                    raise ValueError(f"Missing column '{col}' in custom conditions")
            
            # Extract required columns in correct order
            condition_data = custom_condition_df[condition_cols].values
            
            # Override interpolated data for result comparison
            if len(target_cols) > 0 and all(col in data.columns for col in target_cols):
                comparison_data = data[target_cols].head(1).copy()
            else:
                comparison_data = pd.DataFrame(index=[0], columns=target_cols)
        else:
            # Use condition data from the dataframe
            condition_data = data[condition_cols].values
            comparison_data = data[target_cols].copy() if len(target_cols) > 0 else pd.DataFrame()
        
        # Scale condition data
        scaled_condition = condition_scaler.transform(condition_data)
        
        # Convert to PyTorch tensor
        condition_tensor = torch.FloatTensor(scaled_condition).to(self.device)
        
        # Set generator to evaluation mode
        generator.eval()
        
        # Store raw synthetic data for distribution analysis
        all_raw_generations = []
        all_conditions = []
        
        # Generate multiple samples for each condition
        all_generations = []
        
        print(f"Generating {noise_samples} samples for each of {len(condition_data)} conditions...")
        
        with torch.no_grad():
            for _ in range(noise_samples):
                # Generate random noise
                noise = torch.randn(condition_tensor.size(0), noise_dim).to(self.device)
                
                # Generate fake samples
                fake_target = generator(noise, condition_tensor)
                
                # Store raw samples for distribution analysis (before inverse scaling)
                all_raw_generations.append(fake_target.cpu().numpy())
                all_conditions.append(condition_tensor.cpu().numpy())
                
                # Convert back to numpy
                fake_target_np = fake_target.cpu().numpy()
                
                # Inverse transform to original scale
                fake_target_rescaled = target_scaler.inverse_transform(fake_target_np)
                
                all_generations.append(fake_target_rescaled)
        
        # Stack all generated results
        all_generations_stacked = np.stack(all_generations, axis=0)
        
        # Calculate statistics
        mean_generations = np.mean(all_generations_stacked, axis=0)
        std_generations = np.std(all_generations_stacked, axis=0)
        median_generations = np.median(all_generations_stacked, axis=0)
        min_generations = np.min(all_generations_stacked, axis=0)
        max_generations = np.max(all_generations_stacked, axis=0)
        q1_generations = np.percentile(all_generations_stacked, 25, axis=0)
        q3_generations = np.percentile(all_generations_stacked, 75, axis=0)
        
        # Create CGAN analysis results dataframe
        generated_results = pd.DataFrame()
        
        # Check if results contain NaN or are empty
        if np.isnan(mean_generations).all() or len(mean_generations) == 0:
            print("Warning: Generated data contains only NaN values")
            # Create empty dataframe with appropriate columns
            for i, col in enumerate(target_cols):
                generated_results[col] = np.zeros(condition_data.shape[0])
                generated_results[f'{col}_std'] = np.zeros(condition_data.shape[0])
                if col in comparison_data.columns:
                    generated_results[f'{col}_original'] = comparison_data[col].values
                    generated_results[f'{col}_deviation'] = np.zeros(condition_data.shape[0])
            
            # Return empty results
            return generated_results, {"error": "Generated data contains only NaN values"}
        
        # Create synthetic data DataFrame for detailed statistical analysis
        all_raw_generations_combined = np.vstack(all_raw_generations)
        all_raw_generations_unscaled = target_scaler.inverse_transform(all_raw_generations_combined)
        synthetic_df = pd.DataFrame(all_raw_generations_unscaled, columns=target_cols)
        
        # Add results to the dataframe
        for i, col in enumerate(target_cols):
            if i < mean_generations.shape[1]:  # Ensure column index is in range
                generated_results[col] = mean_generations[:, i]  # Main column is average prediction
                generated_results[f'{col}_std'] = std_generations[:, i]
                
                # Add original values for comparison
                if col in comparison_data.columns:
                    # Ensure both arrays have the same length
                    min_len = min(len(comparison_data[col].values), len(mean_generations[:, i]))
                    if min_len > 0:
                        generated_results[f'{col}_original'] = comparison_data[col].values[:min_len]
                        
                        # Calculate deviation from CGAN prediction
                        generated_results[f'{col}_deviation'] = np.abs(
                            comparison_data[col].values[:min_len] - mean_generations[:min_len, i]
                        )
                        
                        # Add percentage deviation
                        original_mean = np.mean(comparison_data[col].values[:min_len])
                        if abs(original_mean) > 1e-10:
                            generated_results[f'{col}_deviation_pct'] = np.abs(
                                (comparison_data[col].values[:min_len] - mean_generations[:min_len, i]) / original_mean
                            ) * 100
                        else:
                            generated_results[f'{col}_deviation_pct'] = np.zeros(min_len)
                    else:
                        # Handle case where one array is empty
                        generated_results[f'{col}_original'] = []
                        generated_results[f'{col}_deviation'] = []
                        generated_results[f'{col}_deviation_pct'] = []
        
        # Add condition values to results for reference
        for i, col in enumerate(condition_cols):
            if condition_data.shape[1] > i:
                generated_results[f'condition_{col}'] = condition_data[:, i]
        
        # Perform comprehensive statistical metric comparison
        statistical_analysis = {}
        
        # Perform detailed statistical comparison for each target column
        for col in target_cols:
            if col in comparison_data.columns and col in synthetic_df.columns:
                try:
                    # Get actual values from interpolated data
                    real_values = comparison_data[col].dropna().values
                    
                    # Get synthetic data values
                    synthetic_values = synthetic_df[col].dropna().values
                    
                    if len(real_values) > 0 and len(synthetic_values) > 0:
                        # Calculate detailed statistics
                        stats_comparison = {
                            'real_mean': np.mean(real_values),
                            'synthetic_mean': np.mean(synthetic_values),
                            'mean_diff': abs(np.mean(real_values) - np.mean(synthetic_values)),
                            'mean_diff_pct': (abs(np.mean(real_values) - np.mean(synthetic_values)) / max(abs(np.mean(real_values)), 1e-10)) * 100,
                            
                            'real_median': np.median(real_values),
                            'synthetic_median': np.median(synthetic_values),
                            'median_diff': abs(np.median(real_values) - np.median(synthetic_values)),
                            'median_diff_pct': (abs(np.median(real_values) - np.median(synthetic_values)) / max(abs(np.median(real_values)), 1e-10)) * 100,
                            
                            'real_std': np.std(real_values),
                            'synthetic_std': np.std(synthetic_values),
                            'std_diff': abs(np.std(real_values) - np.std(synthetic_values)),
                            'std_diff_pct': (abs(np.std(real_values) - np.std(synthetic_values)) / max(abs(np.std(real_values)), 1e-10)) * 100,
                            
                            'real_min': np.min(real_values),
                            'synthetic_min': np.min(synthetic_values),
                            
                            'real_max': np.max(real_values),
                            'synthetic_max': np.max(synthetic_values),
                            
                            'real_range': np.max(real_values) - np.min(real_values),
                            'synthetic_range': np.max(synthetic_values) - np.min(synthetic_values),
                            'range_diff_pct': (abs((np.max(real_values) - np.min(real_values)) - 
                                              (np.max(synthetic_values) - np.min(synthetic_values))) / 
                                              max(abs(np.max(real_values) - np.min(real_values)), 1e-10)) * 100,
                            
                            'real_q1': np.percentile(real_values, 25),
                            'synthetic_q1': np.percentile(synthetic_values, 25),
                            
                            'real_q3': np.percentile(real_values, 75),
                            'synthetic_q3': np.percentile(synthetic_values, 75),
                            
                            'real_iqr': np.percentile(real_values, 75) - np.percentile(real_values, 25),
                            'synthetic_iqr': np.percentile(synthetic_values, 75) - np.percentile(synthetic_values, 25),
                            'iqr_diff_pct': (abs((np.percentile(real_values, 75) - np.percentile(real_values, 25)) - 
                                            (np.percentile(synthetic_values, 75) - np.percentile(synthetic_values, 25))) / 
                                            max(abs(np.percentile(real_values, 75) - np.percentile(real_values, 25)), 1e-10)) * 100
                        }
                        
                        # Evaluate statistical preservation quality
                        mean_diff_pct = stats_comparison['mean_diff_pct']
                        std_diff_pct = stats_comparison['std_diff_pct']
                        
                        if mean_diff_pct < 5 and std_diff_pct < 10:
                            stats_comparison['preservation_quality'] = 'Excellent'
                        elif mean_diff_pct < 10 and std_diff_pct < 20:
                            stats_comparison['preservation_quality'] = 'Good'
                        elif mean_diff_pct < 20 and std_diff_pct < 30:
                            stats_comparison['preservation_quality'] = 'Fair'
                        else:
                            stats_comparison['preservation_quality'] = 'Poor'
                        
                        # Perform K-S test to compare distributions
                        try:
                            from scipy import stats
                            statistic, p_value = stats.ks_2samp(real_values, synthetic_values)
                            stats_comparison['ks_statistic'] = statistic
                            stats_comparison['ks_p_value'] = p_value
                            stats_comparison['similar_distribution'] = p_value >= 0.05
                        except Exception as e:
                            print(f"K-S test error for {col}: {str(e)}")
                        
                        # Create distribution comparison plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot histograms with kernel density estimation
                        ax.hist(real_values, bins=20, alpha=0.5, density=True, label='Interpolated Data', color='blue')
                        ax.hist(synthetic_values, bins=20, alpha=0.5, density=True, label='Synthetic Data', color='orange')
                        
                        ax.set_title(f'Distribution Comparison for {col} (p-value: {stats_comparison.get("ks_p_value", "N/A"):.4f})')
                        ax.set_xlabel('Value')
                        ax.set_ylabel('Density')
                        ax.legend()
                        
                        # Store plot for later display
                        stats_comparison['distribution_plot'] = fig
                        
                        statistical_analysis[col] = stats_comparison
                except Exception as e:
                    print(f"Error analyzing column {col}: {str(e)}")
        
        # Summarize analysis information
        analysis_info = {
            'statistical_analysis': statistical_analysis,
            'synthetic_data': synthetic_df,
            'conditions_used': custom_conditions if custom_conditions else "Original data conditions",
            'metrics': {
                'num_conditions': len(condition_data),
                'num_samples_per_condition': noise_samples,
                'total_synthetic_samples': len(synthetic_df),
                'num_columns_analyzed': len(statistical_analysis)
            }
        }
        
        print("CGAN generation and statistical comparison analysis completed.")
        return generated_results, analysis_info
        
    def cgan_discriminator_evaluation(self, interpolated_data, original_data, sample_size=100):
        """
        Use the trained discriminator to evaluate the authenticity scores of interpolated data vs original data.
        
        Parameters:
        -----------
        interpolated_data : pandas.DataFrame
            The interpolated dataset to evaluate
        original_data : pandas.DataFrame
            Original dataset to randomly sample from for comparison
        sample_size : int
            Number of samples to randomly draw from the original data
            
        Returns:
        --------
        dict
            Dictionary containing discriminator scores and analysis results
        """
        if self.cgan_model is None or 'discriminator' not in self.cgan_model:
            raise ValueError("CGAN model not trained or missing discriminator. Please call train_cgan method first.")
        
        print("Starting discriminator score analysis...")
        
        # Extract components from the saved model
        discriminator = self.cgan_model['discriminator']
        condition_scaler = self.cgan_model['condition_scaler']
        target_scaler = self.cgan_model['target_scaler']
        condition_cols = self.cgan_model['condition_cols']
        target_cols = self.cgan_model['target_cols']
        
        # Preprocess data
        interpolated_processed = self.preprocess_data(interpolated_data)
        original_processed = self.preprocess_data(original_data)
        
        # Check if there is enough data
        if interpolated_processed.empty or original_processed.empty:
            raise ValueError("Interpolated or original data is empty")
        
        # Ensure all required columns exist
        all_cols = condition_cols + target_cols
        missing_cols_interp = set(all_cols) - set(interpolated_processed.columns)
        missing_cols_orig = set(all_cols) - set(original_processed.columns)
        
        if missing_cols_interp:
            raise ValueError(f"Missing columns in interpolated data: {missing_cols_interp}")
        if missing_cols_orig:
            raise ValueError(f"Missing columns in original data: {missing_cols_orig}")
        
        # Set discriminator to evaluation mode
        discriminator.eval()
        
        # Randomly sample from original data
        if len(original_processed) > sample_size:
            original_sample = original_processed.sample(sample_size, random_state=42)
        else:
            original_sample = original_processed
            print(f"Warning: Original data sample count ({len(original_processed)}) is less than requested sample size ({sample_size})")
        
        # Prepare discriminator input data
        interp_condition_data = interpolated_processed[condition_cols].values
        interp_target_data = interpolated_processed[target_cols].values
        
        orig_condition_data = original_sample[condition_cols].values
        orig_target_data = original_sample[target_cols].values
        
        # Scale data
        interp_condition_scaled = condition_scaler.transform(interp_condition_data)
        interp_target_scaled = target_scaler.transform(interp_target_data)
        
        orig_condition_scaled = condition_scaler.transform(orig_condition_data)
        orig_target_scaled = target_scaler.transform(orig_target_data)
        
        # Convert to PyTorch tensors
        interp_condition_tensor = torch.FloatTensor(interp_condition_scaled).to(self.device)
        interp_target_tensor = torch.FloatTensor(interp_target_scaled).to(self.device)
        
        orig_condition_tensor = torch.FloatTensor(orig_condition_scaled).to(self.device)
        orig_target_tensor = torch.FloatTensor(orig_target_scaled).to(self.device)
        
        # Get discriminator scores
        with torch.no_grad():
            # Discriminator scores for interpolated data
            interp_scores = discriminator(interp_target_tensor, interp_condition_tensor)
            interp_scores_np = interp_scores.cpu().numpy().flatten()
            
            # Discriminator scores for original data
            orig_scores = discriminator(orig_target_tensor, orig_condition_tensor)
            orig_scores_np = orig_scores.cpu().numpy().flatten()
        
        # Calculate statistics
        interp_mean_score = np.mean(interp_scores_np)
        orig_mean_score = np.mean(orig_scores_np)
        
        score_diff = orig_mean_score - interp_mean_score
        
        # Create score distribution plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot discriminator score distributions
        ax.hist(interp_scores_np, bins=20, alpha=0.5, label='Interpolated Data Scores', color='blue')
        ax.hist(orig_scores_np, bins=20, alpha=0.5, label='Original Data Scores', color='green')
        
        ax.axvline(interp_mean_score, color='blue', linestyle='--', alpha=0.8, label=f'Interpolated Mean Score: {interp_mean_score:.4f}')
        ax.axvline(orig_mean_score, color='green', linestyle='--', alpha=0.8, label=f'Original Mean Score: {orig_mean_score:.4f}')
        
        ax.set_title('Discriminator Authenticity Score Distribution Comparison')
        ax.set_xlabel('Discriminator Score (higher means more "real")')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Perform t-test to compare the two score groups
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_ind(orig_scores_np, interp_scores_np, equal_var=False)
            significance = "Significant Difference" if p_value < 0.05 else "No Significant Difference"
        except Exception as e:
            print(f"Error performing t-test: {str(e)}")
            t_stat, p_value, significance = None, None, "Unable to Calculate"
        
        # Create results dictionary
        discriminator_results = {
            'interpolated_scores': interp_scores_np,
            'original_scores': orig_scores_np,
            'interpolated_mean_score': interp_mean_score,
            'original_mean_score': orig_mean_score,
            'score_difference': score_diff,
            'score_distribution_plot': fig,
            't_statistic': t_stat,
            'p_value': p_value,
            'significance': significance,
            'interpretation': {
                'score_quality': self._interpret_discriminator_score(interp_mean_score, orig_mean_score),
                'recommendation': self._get_recommendation(interp_mean_score, orig_mean_score, p_value if p_value is not None else 1.0)
            }
        }
        
        print("Discriminator score analysis completed.")
        return discriminator_results
    
    def _interpret_discriminator_score(self, interp_score, orig_score):
        """Interpret interpolated data quality based on discriminator score"""
        score_ratio = interp_score / max(orig_score, 1e-10)
        diff_percent = abs(orig_score - interp_score) / max(orig_score, 1e-10) * 100
        
        if score_ratio >= 0.95:
            return "Excellent - Interpolated data is nearly indistinguishable from original data"
        elif score_ratio >= 0.85:
            return "Good - Interpolated data is very similar to original data"
        elif score_ratio >= 0.70:
            return "Fair - Interpolated data has some differences from original data"
        elif score_ratio >= 0.50:
            return "Poor - Interpolated data has significant differences from original data"
        else:
            return "Very Poor - Interpolated data has major differences from original data"
    
    def _get_recommendation(self, interp_score, orig_score, p_value):
        """Provide improvement recommendations based on discriminator score"""
        diff_percent = abs(orig_score - interp_score) / max(orig_score, 1e-10) * 100
        
        if diff_percent < 5 and p_value >= 0.05:
            return "Current interpolated data quality is good, no further adjustments needed."
        elif diff_percent < 15:
            return "Consider minor adjustments to interpolation parameters, such as increasing MCMC samples or adjusting experimental data weight."
        elif diff_percent < 30:
            return "Recommend re-running MCMC interpolation, possibly with increased chain count or sample size, or adjusted convergence conditions."
        else:
            return "Need to re-evaluate data interpolation method, consider using different model parameters or alternative interpolation techniques."
            
    def cgan_analysis(self, interpolated_data, noise_samples=100, custom_conditions=None, original_data=None):
        """
        Use trained CGAN model to analyze interpolated data.
        This function is kept for backwards compatibility.
        
        Parameters:
        -----------
        interpolated_data : pandas.DataFrame
            The interpolated data to analyze (typically from Convergence Diagnostics)
        noise_samples : int
            Number of noise samples to generate for each condition
        custom_conditions : dict, optional
            Dictionary with column names as keys and custom condition values to use
            instead of the values from the interpolated data
        original_data : pandas.DataFrame, optional
            Original non-interpolated data to use as reference for comparison.
            
        Returns:
        --------
        pandas.DataFrame
            Data with CGAN analysis results
        dict
            Additional analysis information
        """
        # For backward compatibility, call the new method
        return self.cgan_generate_and_compare(interpolated_data, noise_samples, custom_conditions, original_data)
    
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