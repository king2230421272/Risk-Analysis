import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from modules.data_processing import DataProcessor
from modules.advanced_data_processing import AdvancedDataProcessor
from modules.prediction import Predictor
from modules.risk_assessment import RiskAssessor
from utils.data_handler import DataHandler
from utils.visualization import Visualizer
from utils.database import DatabaseHandler

# Set page configuration
st.set_page_config(
    page_title="Data Analysis Platform",
    page_icon="📊",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'interpolated_data' not in st.session_state:
    st.session_state.interpolated_data = None
if 'data' not in st.session_state:  # Active data for processing
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'risk_assessment' not in st.session_state:
    st.session_state.risk_assessment = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'visualization_type' not in st.session_state:
    st.session_state.visualization_type = None
if 'active_dataset' not in st.session_state:
    st.session_state.active_dataset = "None"
    
# Advanced data processing state variables
if 'interpolated_result' not in st.session_state:
    st.session_state.interpolated_result = None
if 'cgan_results' not in st.session_state:
    st.session_state.cgan_results = None
if 'distribution_test_results' not in st.session_state:
    st.session_state.distribution_test_results = None
if 'outlier_results' not in st.session_state:
    st.session_state.outlier_results = None

# Initialize modules
data_handler = DataHandler()
data_processor = DataProcessor()
advanced_processor = AdvancedDataProcessor()
predictor = Predictor()
risk_assessor = RiskAssessor()
visualizer = Visualizer()
db_handler = DatabaseHandler()

# Application title and description
st.title("Integrated Data Analysis Platform")
st.markdown("""
    This platform provides an integrated workflow for data analysis, from data import to visualization.
    All steps are available in one interface for easier access and navigation.
""")

# Main container
main_container = st.container()

with main_container:
    # Create tabs for each section of the workflow
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1️⃣ Data Import", 
        "2️⃣ Data Processing", 
        "3️⃣ Prediction", 
        "4️⃣ Risk Assessment", 
        "5️⃣ Visualization",
        "6️⃣ Database"
    ])
    
    # 1. DATA IMPORT TAB
    with tab1:
        st.header("Data Import")
        
        # Data import section with two columns for original and interpolated data
        st.subheader("Import Multiple Datasets")
        
        # Import methods tabs
        import_tabs = st.tabs(["Upload Files", "Load from Database"])
        
        # TAB: UPLOAD FILES
        with import_tabs[0]:
            st.markdown("""
            Import both your original data and data to be interpolated to compare distributions and verify interpolation accuracy.
            """)
            
            col1, col2 = st.columns(2)
            
            # ORIGINAL DATA IMPORT
            with col1:
                st.subheader("Original Data")
                original_file = st.file_uploader("Upload Original Data (CSV/Excel)", type=["csv", "xlsx", "xls"], key="original_data_uploader")
                
                if original_file is not None:
                    try:
                        st.session_state.original_data = data_handler.import_data(original_file)
                        st.success(f"Original data imported: {st.session_state.original_data.shape[0]} rows, {st.session_state.original_data.shape[1]} columns")
                        
                        # Show data preview
                        st.write("Preview:")
                        st.dataframe(st.session_state.original_data.head())
                        
                    except Exception as e:
                        st.error(f"Error importing original data: {e}")
            
            # INTERPOLATED DATA IMPORT
            with col2:
                st.subheader("Data to Interpolate")
                interpolated_file = st.file_uploader("Upload Data for Interpolation (CSV/Excel)", type=["csv", "xlsx", "xls"], key="interpolated_data_uploader")
                
                if interpolated_file is not None:
                    try:
                        st.session_state.interpolated_data = data_handler.import_data(interpolated_file)
                        st.success(f"Interpolation data imported: {st.session_state.interpolated_data.shape[0]} rows, {st.session_state.interpolated_data.shape[1]} columns")
                        
                        # Show data preview
                        st.write("Preview:")
                        st.dataframe(st.session_state.interpolated_data.head())
                        
                    except Exception as e:
                        st.error(f"Error importing interpolation data: {e}")
        
        # TAB: LOAD FROM DATABASE
        with import_tabs[1]:
            st.markdown("""
            Load saved datasets from the database to continue your analysis.
            """)
            
            # Check if database is available
            if not hasattr(db_handler, 'db_available') or not db_handler.db_available:
                st.error("⚠️ Database connection is not available. Cannot load data from database.")
                st.info("Please check your database connection settings or continue using file upload.")
            else:
                col1, col2 = st.columns(2)
                
                # LOAD AS ORIGINAL DATA
                with col1:
                    st.subheader("Load as Original Data")
                    
                    try:
                        # Get list of datasets from database
                        all_datasets = db_handler.list_datasets()
                        
                        if not all_datasets:
                            st.info("No datasets found in the database. Please save some datasets first.")
                        else:
                            # Create a formatted selectbox for datasets
                            dataset_options = [(ds['id'], f"{ds['name']} ({ds['data_type']}, {ds['row_count']}x{ds['column_count']})") 
                                              for ds in all_datasets]
                            
                            selected_dataset = st.selectbox(
                                "Select dataset to load as Original Data:",
                                options=dataset_options,
                                format_func=lambda x: x[1],
                                key="original_data_db_select"
                            )
                            
                            if st.button("Load as Original Data", key="load_original_btn"):
                                try:
                                    # Load the selected dataset
                                    loaded_df = db_handler.load_dataset(dataset_id=selected_dataset[0])
                                    st.session_state.original_data = loaded_df
                                    
                                    st.success(f"Dataset loaded as Original Data: {loaded_df.shape[0]} rows, {loaded_df.shape[1]} columns")
                                    
                                    # Show data preview
                                    st.write("Preview:")
                                    st.dataframe(loaded_df.head())
                                    
                                except Exception as e:
                                    st.error(f"Error loading dataset: {e}")
                    
                    except Exception as e:
                        st.error(f"Error accessing database: {e}")
                
                # LOAD AS INTERPOLATED DATA
                with col2:
                    st.subheader("Load as Data to Interpolate")
                    
                    try:
                        # Get list of datasets from database
                        all_datasets = db_handler.list_datasets()
                        
                        if not all_datasets:
                            st.info("No datasets found in the database. Please save some datasets first.")
                        else:
                            # Create a formatted selectbox for datasets
                            dataset_options = [(ds['id'], f"{ds['name']} ({ds['data_type']}, {ds['row_count']}x{ds['column_count']})") 
                                              for ds in all_datasets]
                            
                            selected_dataset = st.selectbox(
                                "Select dataset to load as Data to Interpolate:",
                                options=dataset_options,
                                format_func=lambda x: x[1],
                                key="interpolated_data_db_select"
                            )
                            
                            if st.button("Load as Data to Interpolate", key="load_interpolated_btn"):
                                try:
                                    # Load the selected dataset
                                    loaded_df = db_handler.load_dataset(dataset_id=selected_dataset[0])
                                    st.session_state.interpolated_data = loaded_df
                                    
                                    st.success(f"Dataset loaded as Data to Interpolate: {loaded_df.shape[0]} rows, {loaded_df.shape[1]} columns")
                                    
                                    # Show data preview
                                    st.write("Preview:")
                                    st.dataframe(loaded_df.head())
                                    
                                except Exception as e:
                                    st.error(f"Error loading dataset: {e}")
                    
                    except Exception as e:
                        st.error(f"Error accessing database: {e}")
        
        # Select active dataset for analysis
        st.subheader("Select Active Dataset")
        
        dataset_options = ["None"]
        if st.session_state.original_data is not None:
            dataset_options.append("Original Data")
        if st.session_state.interpolated_data is not None:
            dataset_options.append("Interpolated Data")
            
        st.session_state.active_dataset = st.radio(
            "Select which dataset to use for analysis:",
            dataset_options
        )
        
        # Set the active dataset
        if st.session_state.active_dataset == "Original Data":
            st.session_state.data = st.session_state.original_data
            st.success("Original data set as active dataset for analysis.")
        elif st.session_state.active_dataset == "Interpolated Data":
            st.session_state.data = st.session_state.interpolated_data
            st.success("Interpolated data set as active dataset for analysis.")
        else:
            st.session_state.data = None
            
        # Data Comparison (if both datasets are available)
        if st.session_state.original_data is not None and st.session_state.interpolated_data is not None:
            st.subheader("Dataset Comparison")
            
            with st.expander("Compare Dataset Statistics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("Original Data Statistics")
                    st.write(st.session_state.original_data.describe())
                    
                with col2:
                    st.write("Interpolated Data Statistics")
                    st.write(st.session_state.interpolated_data.describe())
                    
            # Basic comparison metrics can be added here
            st.write("Shape Comparison:")
            st.write(f"Original: {st.session_state.original_data.shape} | Interpolated: {st.session_state.interpolated_data.shape}")
                
            # If columns match, show correlation
            if set(st.session_state.original_data.columns) == set(st.session_state.interpolated_data.columns):
                common_cols = list(set(st.session_state.original_data.select_dtypes(include=np.number).columns) & 
                                set(st.session_state.interpolated_data.select_dtypes(include=np.number).columns))
                
                if common_cols:
                    st.write("Compare Distribution of a Column:")
                    selected_col = st.selectbox("Select column to compare:", common_cols, key="compare_column_select")
                    
                    if selected_col:
                        # Create a comparison histogram
                        fig = plt.figure(figsize=(10, 6))
                        plt.hist(st.session_state.original_data[selected_col], alpha=0.5, label='Original')
                        plt.hist(st.session_state.interpolated_data[selected_col], alpha=0.5, label='Interpolated')
                        plt.legend()
                        plt.title(f'Distribution Comparison: {selected_col}')
                        plt.xlabel(selected_col)
                        plt.ylabel('Frequency')
                        st.pyplot(fig)
    
    # 2. DATA PROCESSING TAB
    with tab2:
        st.header("Data Processing")
        
        if (st.session_state.original_data is None and st.session_state.interpolated_data is None):
            st.warning("No data available. Please import data in the Data Import tab.")
        else:
            # Create processing tabs for Basic and Advanced processing
            processing_tabs = st.tabs(["Basic Processing", "Advanced Processing"])
            
            # BASIC PROCESSING TAB
            with processing_tabs[0]:
                st.subheader("Basic Data Processing")
                
                if st.session_state.data is None:
                    st.warning("No active dataset selected. Please select a dataset in the Data Import tab.")
                else:
                    # Data preview
                    st.write("Active Dataset Preview:")
                    st.dataframe(st.session_state.data.head())
                    
                    # Data processing options
                    st.write("Data Processing Options")
                    
                    # Feature selection
                    st.write("Select features for processing:")
                    all_columns = st.session_state.data.columns.tolist()
                    st.session_state.selected_features = st.multiselect(
                        "Features",
                        all_columns,
                        default=st.session_state.selected_features if st.session_state.selected_features else all_columns
                    )
                    
                    # Target variable selection
                    st.write("Select target variable (for prediction):")
                    
                    # Handle target column selection
                    current_index = 0  # Default to 'None'
                    if st.session_state.target_column is not None and st.session_state.target_column != "None":
                        try:
                            current_index = all_columns.index(st.session_state.target_column) + 1
                        except ValueError:
                            # Target column not in the list, reset to None
                            st.session_state.target_column = None
                            
                    st.session_state.target_column = st.selectbox(
                        "Target Variable",
                        ["None"] + all_columns,
                        index=current_index,
                        key="target_column_select"
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Data cleaning options
                        st.subheader("Data Cleaning")
                        handle_missing = st.checkbox("Handle missing values")
                        missing_method = "Mean imputation"  # Default value
                        if handle_missing:
                            missing_method = st.radio(
                                "Method for handling missing values:",
                                ["Remove rows", "Mean imputation", "Median imputation", "Mode imputation"]
                            )
                        
                        remove_duplicates = st.checkbox("Remove duplicate rows")
                    
                    with col2:
                        # Data transformation options
                        st.subheader("Data Transformation")
                        normalize_data = st.checkbox("Normalize numerical features")
                        norm_method = "Min-Max Scaling"  # Default value
                        if normalize_data:
                            norm_method = st.radio(
                                "Normalization method:",
                                ["Min-Max Scaling", "Standard Scaling"]
                            )
                    
                    # Process data button
                    if st.button("Process Data", key="basic_process_btn"):
                        try:
                            if st.session_state.target_column == "None":
                                target = None
                            else:
                                target = st.session_state.target_column
                            
                            # Get a subset of data with selected features
                            if st.session_state.selected_features:
                                data_subset = st.session_state.data[st.session_state.selected_features].copy()
                            else:
                                data_subset = st.session_state.data.copy()
                            
                            # Process the data
                            st.session_state.processed_data = data_processor.process_data(
                                data_subset,
                                target_column=target,
                                handle_missing=handle_missing,
                                missing_method=missing_method,
                                remove_duplicates=remove_duplicates,
                                normalize_data=normalize_data,
                                norm_method=norm_method
                            )
                            
                            st.success("Data processed successfully!")
                            
                            # Display processed data
                            st.subheader("Processed Data Preview")
                            st.dataframe(st.session_state.processed_data.head())
                            
                            # Display summary of processing
                            st.subheader("Processing Summary")
                            original_shape = st.session_state.data.shape
                            processed_shape = st.session_state.processed_data.shape
                            
                            st.write(f"Original data shape: {original_shape[0]} rows, {original_shape[1]} columns")
                            st.write(f"Processed data shape: {processed_shape[0]} rows, {processed_shape[1]} columns")
                            
                            if handle_missing:
                                st.write(f"Missing values handled using: {missing_method}")
                            
                            if remove_duplicates:
                                st.write(f"Duplicate rows removed")
                            
                            if normalize_data:
                                st.write(f"Numerical features normalized using: {norm_method}")
                            
                        except Exception as e:
                            st.error(f"Error processing data: {e}")
            
            # ADVANCED PROCESSING TAB
            with processing_tabs[1]:
                st.subheader("Advanced Data Processing")
                
                # Check if we have both original and interpolation data
                if st.session_state.original_data is None:
                    st.warning("No original data available. Please import original data in the Data Import tab.")
                elif st.session_state.interpolated_data is None and 'interpolated_result' not in st.session_state:
                    st.info("No data to interpolate detected. You can either:")
                    st.info("1. Import data to interpolate in the Data Import tab, or")
                    st.info("2. Use MCMC interpolation on your original data (with artificially created missing values)")
                    
                    # Option to create artificial missing values
                    create_missing = st.checkbox("Create artificial missing values in original data for interpolation demo", value=False)
                    
                    if create_missing:
                        missing_ratio = st.slider("Percentage of values to make missing", min_value=5, max_value=50, value=20) / 100
                        
                        if st.button("Create Missing Values Dataset"):
                            # Create a copy of original data with artificial missing values
                            import numpy as np
                            
                            data_with_missing = st.session_state.original_data.copy()
                            
                            # Only make numeric columns have missing values
                            numeric_cols = data_with_missing.select_dtypes(include=np.number).columns
                            
                            # Loop through numeric columns and set some values to NaN
                            for col in numeric_cols:
                                mask = np.random.random(len(data_with_missing)) < missing_ratio
                                data_with_missing.loc[mask, col] = np.nan
                            
                            # Store the result
                            st.session_state.interpolated_data = data_with_missing
                            
                            st.success("Created dataset with artificial missing values for interpolation!")
                            st.write("Preview of data with missing values:")
                            st.dataframe(data_with_missing.head())
                            
                            # Update active dataset if needed
                            if st.session_state.active_dataset == "None":
                                st.session_state.active_dataset = "Interpolated Data"
                                st.session_state.data = st.session_state.interpolated_data
                else:
                    # We have both datasets or previously interpolated result, show advanced processing options
                    
                    # Store both datasets for reference
                    original_data = st.session_state.original_data
                    
                    # Choose interpolated data source
                    if 'interpolated_result' in st.session_state:
                        interpolation_source = st.radio(
                            "Interpolated Data Source",
                            ["Use previously interpolated result", "Use imported data for interpolation"],
                            index=0,
                            key="interpolation_source"
                        )
                        
                        if interpolation_source == "Use previously interpolated result":
                            interpolated_data = st.session_state.interpolated_result
                        else:
                            interpolated_data = st.session_state.interpolated_data
                    else:
                        interpolated_data = st.session_state.interpolated_data
                    
                    # Create advanced processing options with tabs
                    advanced_options = st.tabs([
                        "Step 1: MCMC Interpolation", 
                        "Step 2: Multiple Imputation Analysis",
                        "Step 3: CGAN Analysis", 
                        "Step 4: Distribution Testing", 
                        "Step 5: Outlier Detection"
                    ])
                    
                    # 1. MCMC INTERPOLATION TAB
                    with advanced_options[0]:
                        st.write("### MCMC-based Interpolation")
                        st.write("""
                        Markov Chain Monte Carlo (MCMC) interpolation uses Bayesian methods to fill missing values 
                        while accounting for uncertainty in the interpolated values.
                        """)
                        
                        # Interpolation parameters
                        with st.expander("Interpolation Parameters", expanded=True):
                            num_samples = st.slider("Number of MCMC samples", min_value=100, max_value=1000, value=500, step=100)
                            chains = st.slider("Number of MCMC chains", min_value=1, max_value=4, value=2)
                        
                        # Run MCMC interpolation button
                        if st.button("Run MCMC Interpolation", key="mcmc_btn"):
                            try:
                                # Check if we have missing values to interpolate
                                if not interpolated_data.isna().any().any():
                                    st.warning("No missing values detected in the data. MCMC interpolation requires missing values.")
                                else:
                                    with st.spinner("Running MCMC interpolation... (this may take a while)"):
                                        # Run MCMC interpolation
                                        interpolated_result = advanced_processor.mcmc_interpolation(
                                            interpolated_data,
                                            num_samples=num_samples,
                                            chains=chains
                                        )
                                        
                                        # Store the result
                                        st.session_state.interpolated_result = interpolated_result
                                        
                                        # Show results
                                        st.success("MCMC interpolation completed successfully!")
                                        
                                        # Display side-by-side comparison of before and after
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write("Before Interpolation:")
                                            st.dataframe(interpolated_data.head())
                                            
                                        with col2:
                                            st.write("After Interpolation:")
                                            st.dataframe(interpolated_result.head())
                                        
                                        # Show missing value counts before and after
                                        missing_before = interpolated_data.isna().sum().sum()
                                        missing_after = interpolated_result.isna().sum().sum()
                                        
                                        st.write(f"Missing values before: {missing_before}")
                                        st.write(f"Missing values after: {missing_after}")
                                        
                                        # Option to set as active dataset
                                        if st.button("Use Interpolated Result for Analysis", key="use_mcmc_result"):
                                            st.session_state.data = interpolated_result
                                            st.success("Interpolated result set as active dataset for analysis.")
                                        
                                        # Add download button for interpolated data
                                        csv = data_handler.export_data(interpolated_result, format='csv')
                                        st.download_button(
                                            label="Download Interpolated Data as CSV",
                                            data=csv,
                                            file_name="mcmc_interpolated_data.csv",
                                            mime="text/csv"
                                        )
                            except Exception as e:
                                st.error(f"Error during MCMC interpolation: {e}")
                    
                    # 2. MULTIPLE IMPUTATION ANALYSIS TAB
                    with advanced_options[1]:
                        st.write("### Multiple Imputation Analysis")
                        st.write("""
                        After MCMC interpolation, it's important to analyze the imputed data to ensure
                        the statistical properties are preserved and the imputation is reliable.
                        """)
                        
                        # Check if we have MCMC interpolated result
                        if 'interpolated_result' not in st.session_state:
                            st.info("Please run MCMC interpolation first before performing multiple imputation analysis.")
                        else:
                            st.write("#### Statistical Summary of Interpolated Data")
                            
                            # Display side-by-side comparison of original and interpolated data statistics
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("Original Data Statistics")
                                st.dataframe(original_data.describe())
                                
                            with col2:
                                st.write("Interpolated Data Statistics")
                                st.dataframe(st.session_state.interpolated_result.describe())
                            
                            # Compare distributions of original vs interpolated
                            st.write("#### Distribution Comparison")
                            st.write("Compare the distribution of a specific column before and after interpolation:")
                            
                            # Get common numeric columns
                            numeric_cols = list(set(original_data.select_dtypes(include=np.number).columns) & 
                                              set(st.session_state.interpolated_result.select_dtypes(include=np.number).columns))
                            
                            if numeric_cols:
                                selected_col = st.selectbox("Select column:", numeric_cols, key="imputation_compare_col")
                                
                                if selected_col:
                                    # Create a histogram comparison
                                    fig = plt.figure(figsize=(10, 6))
                                    plt.hist(original_data[selected_col].dropna(), alpha=0.5, label='Original Data')
                                    plt.hist(st.session_state.interpolated_result[selected_col].dropna(), alpha=0.5, label='Interpolated Data')
                                    plt.xlabel(selected_col)
                                    plt.ylabel('Frequency')
                                    plt.title(f'Distribution Comparison for {selected_col}')
                                    plt.legend()
                                    st.pyplot(fig)
                                    
                                    # Show statistical tests
                                    st.write("#### Statistical Comparison")
                                    
                                    # Calculate basic statistics
                                    orig_mean = original_data[selected_col].mean()
                                    interp_mean = st.session_state.interpolated_result[selected_col].mean()
                                    mean_diff = abs(orig_mean - interp_mean)
                                    mean_pct_diff = (mean_diff / abs(orig_mean)) * 100 if orig_mean != 0 else 0
                                    
                                    orig_std = original_data[selected_col].std()
                                    interp_std = st.session_state.interpolated_result[selected_col].std()
                                    std_diff = abs(orig_std - interp_std)
                                    std_pct_diff = (std_diff / abs(orig_std)) * 100 if orig_std != 0 else 0
                                    
                                    # Create a comparison dataframe
                                    stats_df = pd.DataFrame({
                                        'Statistic': ['Mean', 'Standard Deviation', 'Min', 'Max', 'Median'],
                                        'Original': [
                                            orig_mean,
                                            orig_std,
                                            original_data[selected_col].min(),
                                            original_data[selected_col].max(),
                                            original_data[selected_col].median()
                                        ],
                                        'Interpolated': [
                                            interp_mean,
                                            interp_std,
                                            st.session_state.interpolated_result[selected_col].min(),
                                            st.session_state.interpolated_result[selected_col].max(),
                                            st.session_state.interpolated_result[selected_col].median()
                                        ],
                                        'Absolute Difference': [
                                            mean_diff,
                                            std_diff,
                                            abs(original_data[selected_col].min() - st.session_state.interpolated_result[selected_col].min()),
                                            abs(original_data[selected_col].max() - st.session_state.interpolated_result[selected_col].max()),
                                            abs(original_data[selected_col].median() - st.session_state.interpolated_result[selected_col].median())
                                        ],
                                        'Percentage Difference': [
                                            f"{mean_pct_diff:.2f}%",
                                            f"{std_pct_diff:.2f}%",
                                            f"{(abs(original_data[selected_col].min() - st.session_state.interpolated_result[selected_col].min()) / abs(original_data[selected_col].min())) * 100:.2f}%" if original_data[selected_col].min() != 0 else "N/A",
                                            f"{(abs(original_data[selected_col].max() - st.session_state.interpolated_result[selected_col].max()) / abs(original_data[selected_col].max())) * 100:.2f}%" if original_data[selected_col].max() != 0 else "N/A",
                                            f"{(abs(original_data[selected_col].median() - st.session_state.interpolated_result[selected_col].median()) / abs(original_data[selected_col].median())) * 100:.2f}%" if original_data[selected_col].median() != 0 else "N/A"
                                        ]
                                    })
                                    
                                    st.dataframe(stats_df)
                                    
                                    # Add quality assessment based on percentage differences
                                    if mean_pct_diff < 5 and std_pct_diff < 10:
                                        st.success("✅ The interpolation has preserved the statistical properties very well!")
                                    elif mean_pct_diff < 10 and std_pct_diff < 20:
                                        st.info("ℹ️ The interpolation has preserved the statistical properties reasonably well.")
                                    else:
                                        st.warning("⚠️ The interpolation has significant differences from the original data. Consider adjusting parameters.")
                            
                            # Scatter plot of original vs interpolated values (for non-missing values)
                            st.write("#### Correlation of Non-Missing Values")
                            st.write("This plot shows how well the interpolation preserved the original non-missing values:")
                            
                            if numeric_cols:
                                selected_col_scatter = st.selectbox("Select column for correlation analysis:", numeric_cols, key="imputation_scatter_col")
                                
                                if selected_col_scatter:
                                    # Get indices where both original and interpolated have non-null values
                                    common_indices = original_data[selected_col_scatter].notna() & interpolated_data[selected_col_scatter].notna()
                                    
                                    if sum(common_indices) > 0:
                                        # Create a scatter plot
                                        fig = plt.figure(figsize=(10, 6))
                                        plt.scatter(
                                            original_data.loc[common_indices, selected_col_scatter],
                                            st.session_state.interpolated_result.loc[common_indices, selected_col_scatter],
                                            alpha=0.5
                                        )
                                        plt.xlabel(f'Original {selected_col_scatter}')
                                        plt.ylabel(f'Interpolated {selected_col_scatter}')
                                        plt.title(f'Original vs Interpolated Values: {selected_col_scatter}')
                                        
                                        # Add perfect correlation line
                                        min_val = min(original_data.loc[common_indices, selected_col_scatter].min(),
                                                   st.session_state.interpolated_result.loc[common_indices, selected_col_scatter].min())
                                        max_val = max(original_data.loc[common_indices, selected_col_scatter].max(),
                                                   st.session_state.interpolated_result.loc[common_indices, selected_col_scatter].max())
                                        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                                        
                                        st.pyplot(fig)
                                        
                                        # Calculate correlation
                                        corr = original_data.loc[common_indices, selected_col_scatter].corr(
                                            st.session_state.interpolated_result.loc[common_indices, selected_col_scatter]
                                        )
                                        
                                        st.metric("Correlation Coefficient", f"{corr:.4f}")
                                        
                                        if corr > 0.95:
                                            st.success("✅ The interpolation has preserved the original values extremely well!")
                                        elif corr > 0.9:
                                            st.success("✅ The interpolation has preserved the original values very well.")
                                        elif corr > 0.7:
                                            st.info("ℹ️ The interpolation has preserved the original values reasonably well.")
                                        else:
                                            st.warning("⚠️ The interpolation shows notable differences from the original values.")
                                    else:
                                        st.warning("No common non-null values found for this column in both datasets.")
                    
                    # 3. CGAN ANALYSIS TAB
                    with advanced_options[2]:
                        st.write("### Conditional Generative Adversarial Network (CGAN) Analysis")
                        st.write("""
                        CGAN analysis uses a generative model to learn patterns in the data and generate synthetic samples
                        that can help validate interpolated data quality.
                        """)
                        
                        # Check if we have MCMC interpolated result
                        if 'interpolated_result' not in st.session_state:
                            st.info("Please run MCMC interpolation first before performing CGAN analysis.")
                        else:
                            # CGAN parameters
                            with st.expander("CGAN Parameters", expanded=True):
                                # Let user select condition and target columns
                                numeric_cols = original_data.select_dtypes(include=np.number).columns.tolist()
                                
                                if len(numeric_cols) < 2:
                                    st.warning("Need at least 2 numeric columns to train CGAN.")
                                else:
                                    # Default to splitting columns in half for conditions and targets
                                    mid_point = len(numeric_cols) // 2
                                    default_conditions = numeric_cols[:mid_point]
                                    default_targets = numeric_cols[mid_point:]
                                    
                                    condition_cols = st.multiselect(
                                        "Condition Columns (features that determine the generated output)",
                                        numeric_cols,
                                        default=default_conditions
                                    )
                                    
                                    remaining_cols = [col for col in numeric_cols if col not in condition_cols]
                                    target_cols = st.multiselect(
                                        "Target Columns (features to be generated/predicted)",
                                        remaining_cols,
                                        default=remaining_cols[:min(len(remaining_cols), 3)]
                                    )
                                    
                                    epochs = st.slider("Training Epochs", min_value=50, max_value=500, value=200, step=50)
                                    batch_size = st.slider("Batch Size", min_value=8, max_value=64, value=32, step=8)
                                    noise_dim = st.slider("Noise Dimension", min_value=50, max_value=200, value=100, step=50)
                            
                                    # Run CGAN analysis button
                                    if len(condition_cols) > 0 and len(target_cols) > 0:
                                        if st.button("Run CGAN Analysis", key="cgan_btn"):
                                            try:
                                                with st.spinner("Training CGAN... (this may take a while)"):
                                                    # Train CGAN on original data
                                                    generator, discriminator = advanced_processor.train_cgan(
                                                        original_data,
                                                        condition_cols=condition_cols,
                                                        target_cols=target_cols,
                                                        epochs=epochs,
                                                        batch_size=batch_size,
                                                        noise_dim=noise_dim
                                                    )
                                                    
                                                    # Analyze interpolated data using CGAN
                                                    cgan_results = advanced_processor.cgan_analysis(
                                                        st.session_state.interpolated_result,
                                                        noise_samples=100
                                                    )
                                                    
                                                    # Store the result
                                                    st.session_state.cgan_results = cgan_results
                                                    
                                                    # Show results
                                                    st.success("CGAN analysis completed successfully!")
                                                    st.write("CGAN Analysis Results:")
                                                    st.dataframe(cgan_results.head())
                                                    
                                                    # Create visualizations of the CGAN analysis
                                                    st.write("### Visualization of CGAN Analysis")
                                                    
                                                    # Select a column to visualize
                                                    col_to_visualize = st.selectbox(
                                                        "Select column to visualize:",
                                                        target_cols
                                                    )
                                                    
                                                    if col_to_visualize:
                                                        # Plot original vs CGAN predicted values
                                                        fig = plt.figure(figsize=(10, 6))
                                                        plt.scatter(
                                                            st.session_state.interpolated_result[col_to_visualize],
                                                            cgan_results[f'{col_to_visualize}_mean'],
                                                            alpha=0.5
                                                        )
                                                        plt.xlabel(f'Interpolated {col_to_visualize}')
                                                        plt.ylabel(f'CGAN Predicted {col_to_visualize}')
                                                        plt.title(f'Interpolated vs CGAN Predicted: {col_to_visualize}')
                                                        
                                                        # Add perfect prediction line
                                                        min_val = min(st.session_state.interpolated_result[col_to_visualize].min(),
                                                                    cgan_results[f'{col_to_visualize}_mean'].min())
                                                        max_val = max(st.session_state.interpolated_result[col_to_visualize].max(),
                                                                    cgan_results[f'{col_to_visualize}_mean'].max())
                                                        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
                                                        
                                                        st.pyplot(fig)
                                                        
                                                        # Plot deviation histogram
                                                        fig = plt.figure(figsize=(10, 6))
                                                        plt.hist(cgan_results[f'{col_to_visualize}_deviation'], bins=20)
                                                        plt.xlabel(f'Deviation in {col_to_visualize}')
                                                        plt.ylabel('Frequency')
                                                        plt.title(f'Deviation between Interpolated and CGAN Predicted: {col_to_visualize}')
                                                        st.pyplot(fig)
                                                    
                                                    # Add download button for CGAN results
                                                    csv = data_handler.export_data(cgan_results, format='csv')
                                                    st.download_button(
                                                        label="Download CGAN Analysis Results as CSV",
                                                        data=csv,
                                                        file_name="cgan_analysis_results.csv",
                                                        mime="text/csv"
                                                    )
                                            except Exception as e:
                                                st.error(f"Error during CGAN analysis: {e}")
                                    else:
                                        st.warning("Please select at least one condition column and one target column.")
                    
                    # 4. DISTRIBUTION TESTING TAB
                    with advanced_options[3]:
                        st.write("### Statistical Distribution Testing")
                        st.write("""
                        These tests compare the distributions of original and interpolated data to verify
                        that the interpolation preserves the underlying statistical properties.
                        """)
                        
                        # Check if we have interpolated result
                        interpolation_data_available = ('interpolated_result' in st.session_state) or (st.session_state.interpolated_data is not None)
                        
                        if not interpolation_data_available:
                            st.info("Please run MCMC interpolation or import interpolated data before performing distribution testing.")
                        else:
                            # Determine which interpolated data to use
                            if 'interpolated_result' in st.session_state:
                                test_data = st.session_state.interpolated_result
                            else:
                                test_data = st.session_state.interpolated_data
                            
                            # Test parameters
                            with st.expander("Test Parameters", expanded=True):
                                alpha = st.slider("Significance Level (alpha)", min_value=0.01, max_value=0.10, value=0.05, step=0.01)
                                num_permutations = st.slider("Number of Permutations", min_value=100, max_value=2000, value=1000, step=100)
                            
                            # Test selection
                            test_options = st.multiselect(
                                "Select tests to perform",
                                ["Kolmogorov-Smirnov Test", "Spearman Rank Correlation", "Permutation Test"],
                                default=["Kolmogorov-Smirnov Test", "Spearman Rank Correlation"]
                            )
                            
                            # Run tests button
                            if len(test_options) > 0:
                                if st.button("Run Distribution Tests", key="dist_test_btn"):
                                    try:
                                        # Store results
                                        test_results = {}
                                        
                                        with st.spinner("Running statistical tests..."):
                                            # Run selected tests
                                            if "Kolmogorov-Smirnov Test" in test_options:
                                                ks_results = advanced_processor.ks_distribution_test(
                                                    original_data, 
                                                    test_data,
                                                    alpha=alpha
                                                )
                                                test_results["ks_test"] = ks_results
                                                
                                                st.subheader("Kolmogorov-Smirnov Test Results")
                                                st.write("""
                                                The KS test compares two distributions to determine if they are statistically different.
                                                A non-significant result (p > alpha) suggests the distributions are similar.
                                                """)
                                                st.dataframe(ks_results)
                                                
                                                # Visualize KS test results
                                                fig = plt.figure(figsize=(10, 6))
                                                plt.bar(ks_results['column'], ks_results['p_value'])
                                                plt.axhline(y=alpha, color='r', linestyle='--', label=f'Alpha = {alpha}')
                                                plt.xlabel('Column')
                                                plt.ylabel('p-value')
                                                plt.title('KS Test p-values (higher is better)')
                                                plt.xticks(rotation=45)
                                                plt.legend()
                                                st.pyplot(fig)
                                            
                                            if "Spearman Rank Correlation" in test_options:
                                                spearman_results = advanced_processor.spearman_correlation(
                                                    original_data, 
                                                    test_data,
                                                    alpha=alpha
                                                )
                                                test_results["spearman_correlation"] = spearman_results
                                                
                                                st.subheader("Spearman Rank Correlation Results")
                                                st.write("""
                                                Spearman correlation measures the monotonic relationship between two datasets.
                                                Values close to 1 indicate a strong positive relationship.
                                                """)
                                                st.dataframe(spearman_results)
                                                
                                                # Visualize Spearman correlation
                                                fig = plt.figure(figsize=(10, 6))
                                                plt.bar(spearman_results['column'], spearman_results['correlation'])
                                                plt.axhline(y=0.7, color='r', linestyle='--', label='Correlation = 0.7')
                                                plt.xlabel('Column')
                                                plt.ylabel('Correlation')
                                                plt.title('Spearman Rank Correlation (higher is better)')
                                                plt.xticks(rotation=45)
                                                plt.legend()
                                                st.pyplot(fig)
                                            
                                            if "Permutation Test" in test_options:
                                                permutation_results = advanced_processor.permutation_test(
                                                    original_data, 
                                                    test_data,
                                                    num_permutations=num_permutations,
                                                    alpha=alpha
                                                )
                                                test_results["permutation_test"] = permutation_results
                                                
                                                st.subheader("Permutation Test Results")
                                                st.write("""
                                                The permutation test assesses if the difference between two datasets is statistically significant.
                                                A non-significant result (p > alpha) suggests the datasets are similar.
                                                """)
                                                st.dataframe(permutation_results)
                                                
                                                # Visualize Permutation test results
                                                fig = plt.figure(figsize=(10, 6))
                                                plt.bar(permutation_results['column'], permutation_results['p_value'])
                                                plt.axhline(y=alpha, color='r', linestyle='--', label=f'Alpha = {alpha}')
                                                plt.xlabel('Column')
                                                plt.ylabel('p-value')
                                                plt.title('Permutation Test p-values (higher is better)')
                                                plt.xticks(rotation=45)
                                                plt.legend()
                                                st.pyplot(fig)
                                            
                                            # Store all test results in session state
                                            st.session_state.distribution_test_results = test_results
                                            
                                            # Overall assessment
                                            st.subheader("Distribution Testing Summary")
                                            
                                            # Calculate overall assessment
                                            num_columns_tested = 0
                                            num_columns_similar = 0
                                            
                                            if "ks_test" in test_results:
                                                ks_df = test_results["ks_test"]
                                                non_sig_cols = ks_df[~ks_df['significant']]['column'].tolist()
                                                num_columns_tested += len(ks_df)
                                                num_columns_similar += len(non_sig_cols)
                                                
                                                st.write(f"KS Test: {len(non_sig_cols)} out of {len(ks_df)} columns have similar distributions.")
                                            
                                            if "spearman_correlation" in test_results:
                                                sp_df = test_results["spearman_correlation"]
                                                high_corr_cols = sp_df[sp_df['correlation'] > 0.7]['column'].tolist()
                                                
                                                st.write(f"Spearman Correlation: {len(high_corr_cols)} out of {len(sp_df)} columns have strong correlation (>0.7).")
                                            
                                            if "permutation_test" in test_results:
                                                perm_df = test_results["permutation_test"]
                                                non_sig_cols = perm_df[~perm_df['significant']]['column'].tolist()
                                                
                                                st.write(f"Permutation Test: {len(non_sig_cols)} out of {len(perm_df)} columns have similar distributions.")
                                            
                                            # Overall similarity assessment
                                            overall_similarity = 0
                                            if "ks_test" in test_results:
                                                ks_similarity = sum(~test_results["ks_test"]['significant']) / len(test_results["ks_test"])
                                                overall_similarity += ks_similarity
                                            
                                            if "spearman_correlation" in test_results:
                                                sp_similarity = sum(test_results["spearman_correlation"]['correlation'] > 0.7) / len(test_results["spearman_correlation"])
                                                overall_similarity += sp_similarity
                                            
                                            if "permutation_test" in test_results:
                                                perm_similarity = sum(~test_results["permutation_test"]['significant']) / len(test_results["permutation_test"])
                                                overall_similarity += perm_similarity
                                            
                                            overall_similarity /= len(test_options)
                                            
                                            # Display similarity as a percentage
                                            st.metric("Overall Distribution Similarity", f"{overall_similarity*100:.1f}%")
                                            
                                            if overall_similarity >= 0.8:
                                                st.success("✅ The interpolated data has very similar statistical properties to the original data.")
                                            elif overall_similarity >= 0.6:
                                                st.info("ℹ️ The interpolated data has moderately similar statistical properties to the original data.")
                                            else:
                                                st.warning("⚠️ The interpolated data shows significant differences from the original data.")
                                    
                                    except Exception as e:
                                        st.error(f"Error during distribution testing: {e}")
                            else:
                                st.warning("Please select at least one test to perform.")
                    
                    # 5. OUTLIER DETECTION TAB
                    with advanced_options[4]:
                        st.write("### Isolated Forest Outlier Detection")
                        st.write("""
                        Isolated Forest is an algorithm that can detect anomalous data points that don't fit the expected pattern.
                        It's useful for identifying potential issues with interpolated values.
                        """)
                        
                        # Check if we have interpolated result
                        interpolation_data_available = ('interpolated_result' in st.session_state) or (st.session_state.interpolated_data is not None)
                        
                        if not interpolation_data_available:
                            st.info("Please run MCMC interpolation or import interpolated data before performing outlier detection.")
                        else:
                            # Determine which interpolated data to use
                            if 'interpolated_result' in st.session_state:
                                test_data = st.session_state.interpolated_result
                            elif 'cgan_results' in st.session_state:
                                test_data = st.session_state.cgan_results
                            else:
                                test_data = st.session_state.interpolated_data
                            
                            # Outlier detection parameters
                            with st.expander("Outlier Detection Parameters", expanded=True):
                                contamination = st.slider(
                                    "Contamination (expected proportion of outliers)",
                                    min_value=0.01,
                                    max_value=0.20,
                                    value=0.05,
                                    step=0.01
                                )
                            
                            # Run outlier detection button
                            if st.button("Run Outlier Detection", key="outlier_btn"):
                                try:
                                    with st.spinner("Running Isolated Forest outlier detection..."):
                                        # Run outlier detection
                                        outlier_results = advanced_processor.isolated_forest_detection(
                                            test_data,
                                            contamination=contamination
                                        )
                                        
                                        # Store the result
                                        st.session_state.outlier_results = outlier_results
                                        
                                        # Show results
                                        st.success("Outlier detection completed successfully!")
                                        st.write("Outlier Detection Results:")
                                        
                                        # Count outliers
                                        outlier_count = outlier_results['is_outlier'].sum()
                                        outlier_percentage = outlier_count / len(outlier_results) * 100
                                        
                                        st.metric("Outliers Detected", f"{outlier_count} ({outlier_percentage:.1f}%)")
                                        
                                        # Display data with outliers highlighted
                                        st.write("Data with Outliers (highlighted rows are outliers):")
                                        
                                        # Use DataFrame styler to highlight outliers
                                        def highlight_outliers(row):
                                            return ['background-color: #ffcccc' if row['is_outlier'] else '' for _ in row]
                                        
                                        styled_df = outlier_results.head(20).style.apply(highlight_outliers, axis=1)
                                        st.dataframe(styled_df)
                                        
                                        # Visualize anomaly scores
                                        st.write("### Anomaly Score Distribution")
                                        fig = plt.figure(figsize=(10, 6))
                                        plt.hist(outlier_results['anomaly_score'], bins=30)
                                        plt.axvline(x=np.percentile(outlier_results['anomaly_score'], 100*(1-contamination)), 
                                                color='r', linestyle='--', 
                                                label=f'Threshold ({100*contamination}% contamination)')
                                        plt.xlabel('Anomaly Score')
                                        plt.ylabel('Frequency')
                                        plt.title('Distribution of Anomaly Scores')
                                        plt.legend()
                                        st.pyplot(fig)
                                        
                                        # Create scatter plot of the first two numeric features with outliers highlighted
                                        numeric_cols = outlier_results.select_dtypes(include=np.number).columns
                                        numeric_cols = [col for col in numeric_cols if col not in ['is_outlier', 'anomaly_score']]
                                        
                                        if len(numeric_cols) >= 2:
                                            st.write("### Outlier Visualization")
                                            
                                            # Let user select features for visualization
                                            x_feature = st.selectbox("X-axis feature:", numeric_cols, index=0)
                                            y_feature = st.selectbox("Y-axis feature:", numeric_cols, index=min(1, len(numeric_cols)-1))
                                            
                                            if x_feature and y_feature:
                                                fig = plt.figure(figsize=(10, 6))
                                                
                                                # Plot non-outliers
                                                plt.scatter(
                                                    outlier_results[~outlier_results['is_outlier']][x_feature],
                                                    outlier_results[~outlier_results['is_outlier']][y_feature],
                                                    alpha=0.5,
                                                    label="Normal points"
                                                )
                                                
                                                # Plot outliers
                                                plt.scatter(
                                                    outlier_results[outlier_results['is_outlier']][x_feature],
                                                    outlier_results[outlier_results['is_outlier']][y_feature],
                                                    color='red',
                                                    marker='x',
                                                    alpha=0.8,
                                                    label="Outliers"
                                                )
                                                
                                                plt.xlabel(x_feature)
                                                plt.ylabel(y_feature)
                                                plt.title(f'Outlier Detection: {x_feature} vs {y_feature}')
                                                plt.legend()
                                                st.pyplot(fig)
                                        
                                        # Add download button for outlier results
                                        csv = data_handler.export_data(outlier_results, format='csv')
                                        st.download_button(
                                            label="Download Outlier Detection Results as CSV",
                                            data=csv,
                                            file_name="outlier_detection_results.csv",
                                            mime="text/csv"
                                        )
                                        
                                        # Option to filter out outliers
                                        if st.button("Filter Out Outliers"):
                                            filtered_data = outlier_results[~outlier_results['is_outlier']].drop(
                                                columns=['is_outlier', 'anomaly_score']
                                            )
                                            
                                            # Store the filtered data
                                            st.session_state.processed_data = filtered_data
                                            
                                            st.success("Outliers filtered out successfully!")
                                            st.write("Filtered Data:")
                                            st.dataframe(filtered_data.head())
                                except Exception as e:
                                    st.error(f"Error during outlier detection: {e}")
    
    # 3. PREDICTION TAB
    with tab3:
        st.header("Prediction")
        
        if st.session_state.processed_data is None:
            st.warning("No processed data available. Please process data in the Data Processing tab.")
        else:
            # Data preview
            st.subheader("Processed Data Preview")
            st.dataframe(st.session_state.processed_data.head())
            
            # Check if target column is selected
            if st.session_state.target_column is None or st.session_state.target_column == "None":
                st.warning("No target column selected. Please go back to the Data Processing tab and select a target column.")
            else:
                st.subheader("Prediction Configuration")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Model selection
                    model_type = st.selectbox(
                        "Select prediction model:",
                        ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"],
                        key="model_type_select"
                    )
                    
                    # Train-test split
                    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
                
                with col2:
                    # Advanced model parameters (optional)
                    show_advanced = st.checkbox("Show advanced model parameters")
                    
                    model_params = {}
                    if show_advanced:
                        if model_type == "Linear Regression":
                            model_params['fit_intercept'] = st.checkbox("Fit intercept", value=True)
                            
                        elif model_type == "Decision Tree":
                            model_params['max_depth'] = st.slider("Maximum depth", 1, 30, 5)
                            model_params['min_samples_split'] = st.slider("Minimum samples to split", 2, 20, 2)
                            
                        elif model_type == "Random Forest":
                            model_params['n_estimators'] = st.slider("Number of trees", 10, 200, 100)
                            model_params['max_depth'] = st.slider("Maximum depth", 1, 30, 5)
                            
                        elif model_type == "Gradient Boosting":
                            model_params['n_estimators'] = st.slider("Number of estimators", 10, 200, 100)
                            model_params['learning_rate'] = st.slider("Learning rate", 0.01, 0.3, 0.1)
                
                # Train model and make predictions
                if st.button("Generate Predictions"):
                    try:
                        # Train model and make predictions
                        predictions, model_details, metrics = predictor.train_and_predict(
                            st.session_state.processed_data,
                            st.session_state.target_column,
                            model_type,
                            test_size,
                            **model_params
                        )
                        
                        st.session_state.predictions = predictions
                        
                        # Display results
                        st.success("Predictions generated successfully!")
                        
                        # Model performance metrics
                        st.subheader("Model Performance")
                        metrics_df = pd.DataFrame(metrics, index=[0])
                        st.table(metrics_df)
                        
                        # Predictions preview
                        st.subheader("Predictions Preview")
                        st.dataframe(predictions.head())
                        
                        # Feature importance (if available)
                        if 'feature_importance' in model_details:
                            st.subheader("Feature Importance")
                            fig = visualizer.plot_feature_importance(
                                model_details['feature_names'],
                                model_details['feature_importance']
                            )
                            st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error generating predictions: {e}")
    
    # 4. RISK ASSESSMENT TAB
    with tab4:
        st.header("Risk Assessment")
        
        if st.session_state.predictions is None:
            st.warning("No predictions available. Please generate predictions in the Prediction tab.")
        else:
            # Predictions preview
            st.subheader("Predictions Preview")
            st.dataframe(st.session_state.predictions.head())
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Assessment Configuration")
                
                # Risk assessment method
                assessment_method = st.selectbox(
                    "Select risk assessment method:",
                    ["Prediction Intervals", "Error Distribution", "Outlier Detection"],
                    key="assessment_method_select"
                )
            
            with col2:
                # Method-specific parameters
                if assessment_method == "Prediction Intervals":
                    confidence_level = st.slider("Confidence level (%)", 70, 99, 95)
                elif assessment_method == "Outlier Detection":
                    threshold = st.slider("Outlier threshold (standard deviations)", 1.0, 5.0, 3.0)
                else:
                    st.write("No additional parameters needed for this method.")
            
            # Perform risk assessment
            if st.button("Assess Risk"):
                try:
                    if assessment_method == "Prediction Intervals":
                        st.session_state.risk_assessment = risk_assessor.prediction_intervals(
                            st.session_state.predictions,
                            confidence_level=confidence_level
                        )
                    elif assessment_method == "Error Distribution":
                        st.session_state.risk_assessment = risk_assessor.error_distribution(
                            st.session_state.predictions
                        )
                    elif assessment_method == "Outlier Detection":
                        st.session_state.risk_assessment = risk_assessor.outlier_detection(
                            st.session_state.predictions,
                            threshold=threshold
                        )
                    
                    st.success("Risk assessment completed!")
                    
                    # Display risk assessment results
                    st.subheader("Risk Assessment Results")
                    st.dataframe(st.session_state.risk_assessment.head())
                    
                    # Visualization based on assessment method
                    st.subheader("Risk Visualization")
                    
                    if assessment_method == "Prediction Intervals":
                        fig = visualizer.plot_prediction_intervals(st.session_state.risk_assessment)
                        st.pyplot(fig)
                    
                    elif assessment_method == "Error Distribution":
                        fig = visualizer.plot_error_distribution(st.session_state.risk_assessment)
                        st.pyplot(fig)
                    
                    elif assessment_method == "Outlier Detection":
                        fig = visualizer.plot_outlier_detection(st.session_state.risk_assessment)
                        st.pyplot(fig)
                    
                    # Risk summary
                    st.subheader("Risk Summary")
                    risk_summary = risk_assessor.generate_risk_summary(st.session_state.risk_assessment, assessment_method)
                    st.markdown(risk_summary)
                    
                except Exception as e:
                    st.error(f"Error assessing risk: {e}")
    
    # 5. VISUALIZATION TAB
    with tab5:
        st.header("Visualization")
        
        if (st.session_state.original_data is None and st.session_state.interpolated_data is None and 
            st.session_state.data is None):
            st.warning("No data available. Please import data in the Data Import tab.")
        else:
            # Data selection
            st.subheader("Select Data to Visualize")
            
            data_options = []
            
            if st.session_state.original_data is not None:
                data_options.append("Original Data")
            if st.session_state.interpolated_data is not None:
                data_options.append("Interpolated Data")
            if st.session_state.processed_data is not None:
                data_options.append("Processed Data")
            if st.session_state.predictions is not None:
                data_options.append("Predictions")
            if st.session_state.risk_assessment is not None:
                data_options.append("Risk Assessment")
                
            if not data_options:
                st.warning("No datasets available for visualization.")
                
            # Additional option for comparison mode
            st.subheader("Visualization Mode")
            viz_mode = st.radio(
                "Select visualization mode:",
                ["Single Dataset", "Compare Datasets"],
                index=1 if (st.session_state.original_data is not None and 
                            st.session_state.interpolated_data is not None) else 0
            )
            
            if viz_mode == "Single Dataset":
                # Single dataset visualization
                data_to_visualize = st.selectbox("Select dataset:", data_options)
                
                # Get the selected data
                if data_to_visualize == "Original Data":
                    viz_data = st.session_state.original_data
                elif data_to_visualize == "Interpolated Data":
                    viz_data = st.session_state.interpolated_data
                elif data_to_visualize == "Processed Data":
                    viz_data = st.session_state.processed_data
                elif data_to_visualize == "Predictions":
                    viz_data = st.session_state.predictions
                elif data_to_visualize == "Risk Assessment":
                    viz_data = st.session_state.risk_assessment
            else:
                # Comparison visualization
                st.subheader("Comparison Visualization")
                
                comparison_options = []
                if st.session_state.original_data is not None and st.session_state.interpolated_data is not None:
                    comparison_options.append("Original vs Interpolated")
                if st.session_state.original_data is not None and st.session_state.processed_data is not None:
                    comparison_options.append("Original vs Processed")
                if st.session_state.interpolated_data is not None and st.session_state.processed_data is not None:
                    comparison_options.append("Interpolated vs Processed")
                
                if not comparison_options:
                    st.warning("Not enough datasets available for comparison. Please import or process more data.")
                    viz_mode = "Single Dataset"
                    data_to_visualize = st.selectbox("Select single dataset instead:", data_options)
                    
                    # Get the selected data for single mode
                    if data_to_visualize == "Original Data":
                        viz_data = st.session_state.original_data
                    elif data_to_visualize == "Interpolated Data":
                        viz_data = st.session_state.interpolated_data
                    elif data_to_visualize == "Processed Data":
                        viz_data = st.session_state.processed_data
                    elif data_to_visualize == "Predictions":
                        viz_data = st.session_state.predictions
                    elif data_to_visualize == "Risk Assessment":
                        viz_data = st.session_state.risk_assessment
                else:
                    # Continue with comparison mode
                    comparison_selection = st.selectbox("Select datasets to compare:", comparison_options)
                    
                    # Set visualization data based on comparison selection
                    if comparison_selection == "Original vs Interpolated":
                        dataset1 = st.session_state.original_data
                        dataset2 = st.session_state.interpolated_data
                        dataset1_name = "Original"
                        dataset2_name = "Interpolated"
                    elif comparison_selection == "Original vs Processed":
                        dataset1 = st.session_state.original_data
                        dataset2 = st.session_state.processed_data
                        dataset1_name = "Original"
                        dataset2_name = "Processed"
                    elif comparison_selection == "Interpolated vs Processed":
                        dataset1 = st.session_state.interpolated_data
                        dataset2 = st.session_state.processed_data
                        dataset1_name = "Interpolated"
                        dataset2_name = "Processed"
                        
                    # Find common numeric columns
                    common_cols = list(set(dataset1.select_dtypes(include=np.number).columns) & 
                                    set(dataset2.select_dtypes(include=np.number).columns))
                    
                    if not common_cols:
                        st.warning("No common numeric columns found for comparison.")
                    
                    # Select column for comparison
                    selected_col = st.selectbox("Select column to compare:", common_cols)
                    
                    # Create comparison visualizations
                    st.subheader(f"Comparing: {selected_col}")
                    
                    # Statistical comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"{dataset1_name} Statistics:")
                        st.write(dataset1[selected_col].describe())
                    with col2:
                        st.write(f"{dataset2_name} Statistics:")
                        st.write(dataset2[selected_col].describe())
                    
                    # Visualization types for comparison
                    comparison_viz_type = st.selectbox(
                        "Select comparison visualization type:",
                        ["Histogram Overlay", "Box Plot Comparison", "Scatter Plot", "Q-Q Plot"]
                    )
                    
                    if comparison_viz_type == "Histogram Overlay":
                        fig = plt.figure(figsize=(10, 6))
                        plt.hist(dataset1[selected_col], alpha=0.5, label=dataset1_name)
                        plt.hist(dataset2[selected_col], alpha=0.5, label=dataset2_name)
                        plt.legend()
                        plt.title(f'Distribution Comparison: {selected_col}')
                        plt.xlabel(selected_col)
                        plt.ylabel('Frequency')
                        st.pyplot(fig)
                        
                    elif comparison_viz_type == "Box Plot Comparison":
                        fig = plt.figure(figsize=(10, 6))
                        # Create combined dataframe for boxplot
                        import pandas as pd
                        combined_data = pd.DataFrame({
                            dataset1_name: dataset1[selected_col],
                            dataset2_name: dataset2[selected_col]
                        })
                        plt.boxplot([dataset1[selected_col], dataset2[selected_col]])
                        plt.xticks([1, 2], [dataset1_name, dataset2_name])
                        plt.title(f'Box Plot Comparison: {selected_col}')
                        plt.ylabel(selected_col)
                        st.pyplot(fig)
                        
                    elif comparison_viz_type == "Scatter Plot":
                        # Only if datasets have same length and are sortable
                        if len(dataset1) == len(dataset2):
                            fig = plt.figure(figsize=(10, 6))
                            plt.scatter(dataset1[selected_col], dataset2[selected_col], alpha=0.5)
                            plt.plot([dataset1[selected_col].min(), dataset1[selected_col].max()], 
                                     [dataset1[selected_col].min(), dataset1[selected_col].max()], 
                                     'r--')
                            plt.title(f'Scatter Plot: {dataset1_name} vs {dataset2_name} - {selected_col}')
                            plt.xlabel(f'{dataset1_name} {selected_col}')
                            plt.ylabel(f'{dataset2_name} {selected_col}')
                            st.pyplot(fig)
                        else:
                            st.warning("Scatter plot comparison requires datasets of equal length.")
                            
                    elif comparison_viz_type == "Q-Q Plot":
                        import scipy.stats as stats
                        fig = plt.figure(figsize=(10, 6))
                        stats.probplot(dataset1[selected_col], dist="norm", plot=plt)
                        plt.title(f'Q-Q Plot - {dataset1_name}: {selected_col}')
                        st.pyplot(fig)
                        
                        fig = plt.figure(figsize=(10, 6))
                        stats.probplot(dataset2[selected_col], dist="norm", plot=plt)
                        plt.title(f'Q-Q Plot - {dataset2_name}: {selected_col}')
                        st.pyplot(fig)
                    
                    # Skip regular visualization as we're in comparison mode
                    viz_data = None
            
            if viz_data is not None:
                # Visualization type
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Select Visualization Type")
                    viz_type = st.selectbox(
                        "Visualization type:",
                        ["Data Overview", "Histogram", "Scatter Plot", "Line Chart", 
                         "Bar Chart", "Correlation Matrix", "Box Plot"]
                    )
                
                # Generate visualization
                if viz_type == "Data Overview":
                    # Show data preview and statistics
                    st.subheader("Data Preview")
                    st.dataframe(viz_data.head())
                    
                    st.subheader("Data Shape")
                    st.write(f"Rows: {viz_data.shape[0]}, Columns: {viz_data.shape[1]}")
                    
                    st.subheader("Data Types")
                    st.write(viz_data.dtypes)
                    
                    # Show statistics for numeric columns
                    numeric_cols = viz_data.select_dtypes(include=np.number).columns
                    if len(numeric_cols) > 0:
                        st.subheader("Numeric Data Statistics")
                        st.write(viz_data[numeric_cols].describe())
                
                elif viz_type == "Histogram":
                    with col2:
                        # Select column for histogram
                        column = st.selectbox("Select column for histogram:", viz_data.columns)
                    
                    # Check if column is numeric
                    if pd.api.types.is_numeric_dtype(viz_data[column]):
                        fig = visualizer.plot_histogram(viz_data, column)
                        st.pyplot(fig)
                    else:
                        st.warning("Histograms can only be plotted for numeric columns.")
                
                elif viz_type == "Scatter Plot":
                    with col2:
                        # Select columns for scatter plot
                        x_column = st.selectbox("Select X-axis column:", viz_data.columns)
                        y_column = st.selectbox("Select Y-axis column:", viz_data.columns, index=min(1, len(viz_data.columns)-1))
                    
                    # Check if columns are numeric
                    if pd.api.types.is_numeric_dtype(viz_data[x_column]) and pd.api.types.is_numeric_dtype(viz_data[y_column]):
                        fig = visualizer.plot_scatter(viz_data, x_column, y_column)
                        st.pyplot(fig)
                    else:
                        st.warning("Scatter plots can only be plotted for numeric columns.")
                
                elif viz_type == "Line Chart":
                    with col2:
                        # Select columns for line chart
                        columns = st.multiselect("Select columns for line chart:", viz_data.select_dtypes(include=np.number).columns)
                    
                    if columns:
                        fig = visualizer.plot_line(viz_data, columns)
                        st.pyplot(fig)
                    else:
                        st.warning("Please select at least one column for the line chart.")
                
                elif viz_type == "Bar Chart":
                    with col2:
                        # Select columns for bar chart
                        x_column = st.selectbox("Select X-axis column (categories):", viz_data.columns)
                        y_column = st.selectbox("Select Y-axis column (values):", viz_data.select_dtypes(include=np.number).columns)
                    
                    fig = visualizer.plot_bar(viz_data, x_column, y_column)
                    st.pyplot(fig)
                
                elif viz_type == "Correlation Matrix":
                    # Get numeric columns
                    numeric_cols = viz_data.select_dtypes(include=np.number).columns
                    
                    if len(numeric_cols) > 1:
                        fig = visualizer.plot_correlation(viz_data[numeric_cols])
                        st.pyplot(fig)
                    else:
                        st.warning("Correlation matrix requires at least two numeric columns.")
                
                elif viz_type == "Box Plot":
                    with col2:
                        # Select column for box plot
                        column = st.selectbox("Select column for box plot:", viz_data.select_dtypes(include=np.number).columns)
                    
                    fig = visualizer.plot_box(viz_data, column)
                    st.pyplot(fig)
                
                # Export visualization
                st.subheader("Export Options")
                
                if st.button("Export Data"):
                    try:
                        export_format = "csv"
                        export_success, download_link = data_handler.export_data(viz_data, export_format)
                        if export_success:
                            st.markdown(download_link, unsafe_allow_html=True)
                        else:
                            st.error("Failed to export data.")
                    except Exception as e:
                        st.error(f"Error exporting data: {e}")

# 6. DATABASE TAB
    with tab6:
        st.header("Database Management")
        
        # Check if database is available
        if not hasattr(db_handler, 'db_available') or not db_handler.db_available:
            st.error("⚠️ Database connection is not available. Database features are disabled.")
            st.info("The application will continue to work without database functionality. You can still import, process, and analyze data, but you won't be able to save or load from the database.")
            st.info("Please check your database connection settings or contact your administrator for assistance.")
            
            # Show technical details in an expander for troubleshooting
            with st.expander("Technical Details"):
                st.code("""
# Common issues:
1. PostgreSQL service may not be running
2. Database credentials may be incorrect
3. Network connectivity issues
4. SSL connection failures

# Troubleshooting:
- Check that the PostgreSQL database service is running
- Verify that the DATABASE_URL environment variable is correctly set
- Ensure network connectivity to the database server
- Check firewall settings that might block database connections
                """)
        else:
            # Database operations
            db_operation = st.radio(
                "Select Database Operation:",
                ["Save Dataset", "Load Dataset", "List Saved Datasets", "Save Analysis Result", "View Analysis Results", "Delete Dataset"]
            )
        
        if not hasattr(db_handler, 'db_available') or not db_handler.db_available:
            pass  # Already handled with error message above
        elif db_operation == "Save Dataset":
            st.subheader("Save Dataset to Database")
            
            # Select dataset to save
            save_data_options = []
            if st.session_state.original_data is not None:
                save_data_options.append("Original Data")
            if st.session_state.interpolated_data is not None:
                save_data_options.append("Interpolated Data")
            if st.session_state.processed_data is not None:
                save_data_options.append("Processed Data")
            if st.session_state.predictions is not None:
                save_data_options.append("Predictions")
                
            if not save_data_options:
                st.warning("No datasets available to save. Please import or generate data first.")
            else:
                dataset_to_save = st.selectbox("Select dataset to save:", save_data_options)
                
                # Get selected dataset
                if dataset_to_save == "Original Data":
                    save_df = st.session_state.original_data
                    data_type = "original"
                elif dataset_to_save == "Interpolated Data":
                    save_df = st.session_state.interpolated_data
                    data_type = "interpolated"
                elif dataset_to_save == "Processed Data":
                    save_df = st.session_state.processed_data
                    data_type = "processed"
                elif dataset_to_save == "Predictions":
                    save_df = st.session_state.predictions
                    data_type = "predictions"
                
                # Dataset name and description
                save_name = st.text_input("Dataset name:", f"{dataset_to_save} - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
                save_description = st.text_area("Description (optional):", f"Saved from {dataset_to_save} tab")
                
                # Save button
                if st.button("Save to Database"):
                    try:
                        dataset_id = db_handler.save_dataset(
                            save_df, 
                            name=save_name,
                            description=save_description,
                            data_type=data_type
                        )
                        st.success(f"Dataset saved successfully with ID: {dataset_id}")
                    except Exception as e:
                        st.error(f"Error saving dataset: {e}")
        
        elif db_operation == "Load Dataset":
            st.subheader("Load Dataset from Database")
            
            # List available datasets
            try:
                saved_datasets = db_handler.list_datasets()
                
                if not saved_datasets:
                    st.info("No datasets found in the database.")
                else:
                    # Create a formatted dataframe for display
                    datasets_df = pd.DataFrame(saved_datasets)
                    datasets_df['created_at'] = datasets_df['created_at'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
                    if 'modified_at' in datasets_df.columns:
                        datasets_df['modified_at'] = datasets_df['modified_at'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M') if x else "")
                    
                    st.write("Available Datasets:")
                    st.dataframe(datasets_df)
                    
                    # Select dataset to load
                    dataset_id = st.selectbox(
                        "Select dataset to load:",
                        [(ds['id'], ds['name']) for ds in saved_datasets],
                        format_func=lambda x: f"{x[1]} (ID: {x[0]})"
                    )
                    
                    # Select where to load
                    load_target = st.radio(
                        "Load as:",
                        ["Original Data", "Interpolated Data"]
                    )
                    
                    # Load button
                    if st.button("Load Dataset"):
                        try:
                            loaded_df = db_handler.load_dataset(dataset_id=dataset_id[0])
                            
                            if load_target == "Original Data":
                                st.session_state.original_data = loaded_df
                                st.success(f"Dataset loaded as Original Data: {loaded_df.shape[0]} rows, {loaded_df.shape[1]} columns")
                            else:
                                st.session_state.interpolated_data = loaded_df
                                st.success(f"Dataset loaded as Interpolated Data: {loaded_df.shape[0]} rows, {loaded_df.shape[1]} columns")
                                
                            # Preview
                            st.subheader("Data Preview")
                            st.dataframe(loaded_df.head())
                            
                        except Exception as e:
                            st.error(f"Error loading dataset: {e}")
            
            except Exception as e:
                st.error(f"Error accessing database: {e}")
        
        elif db_operation == "List Saved Datasets":
            st.subheader("Saved Datasets")
            
            # Filter options
            filter_type = st.radio(
                "Filter by data type:",
                ["All", "Original", "Interpolated", "Processed", "Predictions"]
            )
            
            data_type_filter = None
            if filter_type != "All":
                data_type_filter = filter_type.lower()
            
            # Get and display datasets
            try:
                datasets = db_handler.list_datasets(data_type=data_type_filter)
                
                if not datasets:
                    st.info(f"No {'datasets' if filter_type == 'All' else filter_type + ' datasets'} found in the database.")
                else:
                    # Convert to dataframe for display
                    datasets_df = pd.DataFrame(datasets)
                    datasets_df['created_at'] = datasets_df['created_at'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
                    if 'modified_at' in datasets_df.columns:
                        datasets_df['modified_at'] = datasets_df['modified_at'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M') if x else "")
                    
                    st.write(f"Found {len(datasets)} datasets:")
                    st.dataframe(datasets_df)
            
            except Exception as e:
                st.error(f"Error listing datasets: {e}")
        
        elif db_operation == "Save Analysis Result":
            st.subheader("Save Analysis Result to Database")
            
            # Check if we have results to save
            if st.session_state.predictions is None and st.session_state.risk_assessment is None:
                st.warning("No analysis results available to save. Please generate predictions or perform risk assessment first.")
            else:
                # Select result type
                result_options = []
                if st.session_state.predictions is not None:
                    result_options.append("Prediction Results")
                if st.session_state.risk_assessment is not None:
                    result_options.append("Risk Assessment Results")
                
                result_type = st.selectbox("Select result type to save:", result_options)
                
                # Get datasets to choose from
                try:
                    datasets = db_handler.list_datasets()
                    
                    if not datasets:
                        st.warning("No datasets found in the database. Please save a dataset first.")
                    else:
                        # Select related dataset
                        dataset_id = st.selectbox(
                            "Select related dataset:",
                            [(ds['id'], ds['name']) for ds in datasets],
                            format_func=lambda x: f"{x[1]} (ID: {x[0]})"
                        )
                        
                        # Get result data
                        if result_type == "Prediction Results":
                            result_data = st.session_state.predictions
                            analysis_type = "prediction"
                        else:
                            result_data = st.session_state.risk_assessment
                            analysis_type = "risk_assessment"
                        
                        # Analysis parameters
                        st.write("Analysis parameters (optional):")
                        param_description = st.text_area("Description:", "")
                        
                        # Convert to parameters dict
                        analysis_params = {"description": param_description}
                        
                        # Save button
                        if st.button("Save Analysis Result"):
                            try:
                                result_id = db_handler.save_analysis_result(
                                    dataset_id=dataset_id[0],
                                    analysis_type=analysis_type,
                                    analysis_params=analysis_params,
                                    result_data=result_data
                                )
                                st.success(f"Analysis result saved successfully with ID: {result_id}")
                            except Exception as e:
                                st.error(f"Error saving analysis result: {e}")
                        
                except Exception as e:
                    st.error(f"Error accessing database: {e}")
        
        elif db_operation == "View Analysis Results":
            st.subheader("View Analysis Results")
            
            # Filter options
            filter_type = st.radio(
                "Filter by analysis type:",
                ["All", "Prediction", "Risk Assessment"]
            )
            
            analysis_type_filter = None
            if filter_type != "All":
                analysis_type_filter = filter_type.lower().replace(" ", "_")
            
            # Get datasets to filter by
            try:
                datasets = db_handler.list_datasets()
                dataset_options = [("all", "All Datasets")] + [(ds['id'], ds['name']) for ds in datasets]
                
                dataset_filter = st.selectbox(
                    "Filter by dataset:",
                    dataset_options,
                    format_func=lambda x: x[1]
                )
                
                dataset_id_filter = None if dataset_filter[0] == "all" else dataset_filter[0]
                
                # Get and display analysis results
                results = db_handler.list_analysis_results(
                    dataset_id=dataset_id_filter,
                    analysis_type=analysis_type_filter
                )
                
                if not results:
                    st.info("No analysis results found matching the selected filters.")
                else:
                    # Convert to dataframe for display
                    results_df = pd.DataFrame(results)
                    results_df['created_at'] = results_df['created_at'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
                    
                    # Add dataset name
                    if datasets:
                        dataset_map = {ds['id']: ds['name'] for ds in datasets}
                        results_df['dataset_name'] = results_df['dataset_id'].apply(lambda x: dataset_map.get(x, "Unknown"))
                    
                    # Format analysis type
                    results_df['analysis_type'] = results_df['analysis_type'].apply(
                        lambda x: "Prediction" if x == "prediction" else "Risk Assessment"
                    )
                    
                    st.write(f"Found {len(results)} analysis results:")
                    st.dataframe(results_df)
                    
                    # View specific result
                    if not results_df.empty:
                        result_id = st.selectbox(
                            "Select result to view:",
                            results_df['id'].tolist(),
                            format_func=lambda x: f"Result ID: {x}"
                        )
                        
                        if st.button("View Result"):
                            try:
                                dataset_id, analysis_type, analysis_params, result_data = db_handler.load_analysis_result(result_id)
                                
                                st.subheader("Analysis Result Details")
                                st.write(f"Dataset ID: {dataset_id}")
                                st.write(f"Analysis Type: {analysis_type}")
                                
                                if analysis_params:
                                    st.write("Analysis Parameters:")
                                    st.json(analysis_params)
                                
                                st.write("Result Data:")
                                if isinstance(result_data, pd.DataFrame):
                                    st.dataframe(result_data)
                                else:
                                    st.json(result_data)
                                    
                            except Exception as e:
                                st.error(f"Error loading analysis result: {e}")
            
            except Exception as e:
                st.error(f"Error accessing database: {e}")
                
        elif db_operation == "Delete Dataset":
            st.subheader("Delete Dataset")
            st.warning("⚠️ Warning: This will permanently delete the dataset and all associated analysis results!")
            
            # Get datasets to delete
            try:
                datasets = db_handler.list_datasets()
                
                if not datasets:
                    st.info("No datasets found in the database.")
                else:
                    # Convert to dataframe for display
                    datasets_df = pd.DataFrame(datasets)
                    datasets_df['created_at'] = datasets_df['created_at'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M'))
                    
                    st.write("Available Datasets:")
                    st.dataframe(datasets_df)
                    
                    # Select dataset to delete
                    dataset_id = st.selectbox(
                        "Select dataset to delete:",
                        [(ds['id'], ds['name']) for ds in datasets],
                        format_func=lambda x: f"{x[1]} (ID: {x[0]})"
                    )
                    
                    # Confirmation
                    confirm = st.checkbox("I understand this action cannot be undone")
                    
                    # Delete button
                    if st.button("Delete Dataset", disabled=not confirm):
                        try:
                            success = db_handler.delete_dataset(dataset_id[0])
                            if success:
                                st.success(f"Dataset {dataset_id[1]} (ID: {dataset_id[0]}) deleted successfully")
                            else:
                                st.error("Dataset could not be deleted")
                        except Exception as e:
                            st.error(f"Error deleting dataset: {e}")
            
            except Exception as e:
                st.error(f"Error accessing database: {e}")

# Add a footer with reset option
st.markdown("---")
col1, col2 = st.columns([4, 1])
with col2:
    if st.button("Reset All"):
        # Reset all session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
        
with col1:
    st.markdown("💡 **Tip**: Use the tabs above to navigate between different steps of the analysis pipeline. You can move freely between tabs at any time.")