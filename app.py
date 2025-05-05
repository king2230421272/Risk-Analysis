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
                    # Initialize the processed data output if needed
                    if 'basic_processed_outputs' not in st.session_state:
                        st.session_state.basic_processed_outputs = {}
                
                    # Data preview
                    st.write("Active Dataset Preview:")
                    st.dataframe(st.session_state.data.head())
                    
                    # Create tabs for different processing categories
                    basic_tabs = st.tabs(["Feature Selection", "Data Cleaning", "Transformation", "Interpolation", "Results"])
                    
                    # 1. FEATURE SELECTION TAB
                    with basic_tabs[0]:
                        st.write("### Feature Selection")
                        st.write("Select features and target variable for processing:")
                        
                        # Get all columns
                        all_columns = st.session_state.data.columns.tolist()
                        
                        # Feature selection
                        st.session_state.selected_features = st.multiselect(
                            "Features to include in processing",
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
                        
                        # Apply feature selection
                        if st.button("Apply Feature Selection", key="apply_feature_btn"):
                            # Get a subset of data with selected features
                            if st.session_state.selected_features:
                                subset_data = st.session_state.data[st.session_state.selected_features].copy()
                                
                                # Store the subset data
                                st.session_state.basic_processed_outputs['selected_features'] = subset_data
                                
                                st.success(f"Selected {len(st.session_state.selected_features)} features")
                                st.write("Preview of selected features:")
                                st.dataframe(subset_data.head())
                            else:
                                st.warning("No features selected.")
                    
                    # 2. DATA CLEANING TAB
                    with basic_tabs[1]:
                        st.write("### Data Cleaning")
                        st.write("Options for cleaning and preparing the data:")
                        
                        # Choose data source
                        data_source_options = ["Original Data"]
                        if st.session_state.basic_processed_outputs:
                            data_source_options.extend(list(st.session_state.basic_processed_outputs.keys()))
                        
                        data_source = st.selectbox(
                            "Select data source for cleaning:",
                            data_source_options,
                            key="cleaning_data_source"
                        )
                        
                        # Get the selected data
                        if data_source == "Original Data":
                            cleaning_data = st.session_state.data
                        else:
                            cleaning_data = st.session_state.basic_processed_outputs[data_source]
                        
                        # Missing value handling
                        st.write("#### Missing Value Handling")
                        handle_missing = st.checkbox("Handle missing values", key="handle_missing")
                        
                        if handle_missing:
                            missing_method = st.radio(
                                "Method for handling missing values:",
                                ["Remove rows", "Remove columns", "Mean imputation", "Median imputation", "Mode imputation"],
                                key="missing_method"
                            )
                            
                            # Show missing value statistics
                            missing_df = pd.DataFrame({
                                'Column': cleaning_data.columns,
                                'Missing Values': cleaning_data.isna().sum().values,
                                'Percentage': (cleaning_data.isna().sum().values / len(cleaning_data) * 100).round(2)
                            }).sort_values(by='Missing Values', ascending=False)
                            
                            st.write("#### Missing Value Statistics")
                            st.dataframe(missing_df)
                        
                        # Duplicate handling
                        st.write("#### Duplicate Handling")
                        remove_duplicates = st.checkbox("Remove duplicate rows", key="remove_duplicates")
                        
                        if remove_duplicates:
                            # Show duplicate information
                            duplicate_count = cleaning_data.duplicated().sum()
                            if duplicate_count > 0:
                                st.write(f"Found {duplicate_count} duplicate rows ({duplicate_count/len(cleaning_data)*100:.2f}% of data)")
                            else:
                                st.write("No duplicate rows found in the data.")
                        
                        # Apply cleaning button
                        if st.button("Apply Data Cleaning", key="apply_cleaning_btn"):
                            # Create a copy of the data
                            cleaned_data = cleaning_data.copy()
                            cleaning_steps = []
                            
                            try:
                                # Handle missing values
                                if handle_missing:
                                    if missing_method == "Remove rows":
                                        rows_before = len(cleaned_data)
                                        cleaned_data = cleaned_data.dropna()
                                        rows_removed = rows_before - len(cleaned_data)
                                        cleaning_steps.append(f"Removed {rows_removed} rows with missing values")
                                    
                                    elif missing_method == "Remove columns":
                                        cols_before = len(cleaned_data.columns)
                                        # Remove columns with more than 50% missing values
                                        threshold = 0.5
                                        cleaned_data = cleaned_data.loc[:, cleaned_data.isna().mean() < threshold]
                                        cols_removed = cols_before - len(cleaned_data.columns)
                                        cleaning_steps.append(f"Removed {cols_removed} columns with >50% missing values")
                                    
                                    elif missing_method == "Mean imputation":
                                        # Only apply to numeric columns
                                        numeric_cols = cleaned_data.select_dtypes(include=['number']).columns
                                        for col in numeric_cols:
                                            cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].mean())
                                        cleaning_steps.append(f"Filled missing values in numeric columns with mean")
                                    
                                    elif missing_method == "Median imputation":
                                        # Only apply to numeric columns
                                        numeric_cols = cleaned_data.select_dtypes(include=['number']).columns
                                        for col in numeric_cols:
                                            cleaned_data[col] = cleaned_data[col].fillna(cleaned_data[col].median())
                                        cleaning_steps.append(f"Filled missing values in numeric columns with median")
                                    
                                    elif missing_method == "Mode imputation":
                                        # Apply to all columns
                                        for col in cleaned_data.columns:
                                            # Check if mode exists
                                            mode_values = cleaned_data[col].mode()
                                            if not mode_values.empty:
                                                cleaned_data[col] = cleaned_data[col].fillna(mode_values[0])
                                        cleaning_steps.append(f"Filled missing values in all columns with mode")
                                
                                # Handle duplicates
                                if remove_duplicates:
                                    rows_before = len(cleaned_data)
                                    cleaned_data = cleaned_data.drop_duplicates()
                                    rows_removed = rows_before - len(cleaned_data)
                                    cleaning_steps.append(f"Removed {rows_removed} duplicate rows")
                                
                                # Store the cleaned data
                                st.session_state.basic_processed_outputs['cleaned_data'] = cleaned_data
                                
                                # Display summary
                                st.success("Data cleaning applied successfully!")
                                st.write("#### Cleaning Summary")
                                for step in cleaning_steps:
                                    st.write(f"- {step}")
                                
                                # Display shape information
                                st.write(f"Original shape: {cleaning_data.shape[0]} rows, {cleaning_data.shape[1]} columns")
                                st.write(f"Cleaned shape: {cleaned_data.shape[0]} rows, {cleaned_data.shape[1]} columns")
                                
                                # Display cleaned data
                                st.write("#### Cleaned Data Preview")
                                st.dataframe(cleaned_data.head())
                                
                            except Exception as e:
                                st.error(f"Error during data cleaning: {e}")
                    
                    # 3. TRANSFORMATION TAB
                    with basic_tabs[2]:
                        st.write("### Data Transformation")
                        st.write("Apply transformations to prepare data for modeling:")
                        
                        # Choose data source
                        data_source_options = ["Original Data"]
                        if st.session_state.basic_processed_outputs:
                            data_source_options.extend(list(st.session_state.basic_processed_outputs.keys()))
                        
                        data_source = st.selectbox(
                            "Select data source for transformation:",
                            data_source_options,
                            key="transform_data_source"
                        )
                        
                        # Get the selected data
                        if data_source == "Original Data":
                            transform_data = st.session_state.data
                        else:
                            transform_data = st.session_state.basic_processed_outputs[data_source]
                        
                        # Normalization/Scaling
                        st.write("#### Normalization & Scaling")
                        normalize_data = st.checkbox("Normalize/Scale numerical features", key="normalize_data")
                        
                        if normalize_data:
                            norm_method = st.radio(
                                "Normalization method:",
                                ["Min-Max Scaling", "Standard Scaling", "Robust Scaling"],
                                key="norm_method"
                            )
                        
                        # Apply transformations button
                        if st.button("Apply Transformations", key="apply_transform_btn"):
                            # Create a copy of the data
                            transformed_data = transform_data.copy()
                            transform_steps = []
                            
                            try:
                                # Apply normalizations
                                if normalize_data:
                                    # Get numeric columns
                                    numeric_cols = transformed_data.select_dtypes(include=['number']).columns.tolist()
                                    
                                    if numeric_cols:
                                        # Process data using existing data_processor
                                        transformed_data = data_processor.process_data(
                                            transformed_data,
                                            target_column=st.session_state.target_column if st.session_state.target_column != "None" else None,
                                            handle_missing=False,
                                            remove_duplicates=False,
                                            normalize_data=True,
                                            norm_method=norm_method
                                        )
                                        
                                        transform_steps.append(f"Applied {norm_method} to {len(numeric_cols)} numeric columns")
                                    else:
                                        st.warning("No numeric columns found for normalization.")
                                
                                # Store the transformed data
                                st.session_state.basic_processed_outputs['transformed_data'] = transformed_data
                                
                                # Display summary
                                st.success("Data transformations applied successfully!")
                                st.write("#### Transformation Summary")
                                for step in transform_steps:
                                    st.write(f"- {step}")
                                
                                # Display shape information
                                st.write(f"Original shape: {transform_data.shape[0]} rows, {transform_data.shape[1]} columns")
                                st.write(f"Transformed shape: {transformed_data.shape[0]} rows, {transformed_data.shape[1]} columns")
                                
                                # Display the transformed data
                                st.write("#### Transformed Data Preview")
                                st.dataframe(transformed_data.head())
                                
                            except Exception as e:
                                st.error(f"Error during data transformation: {e}")
                    
                    # 4. INTERPOLATION TAB
                    with basic_tabs[3]:
                        st.write("### Data Interpolation")
                        st.write("Fill missing values using various interpolation methods:")
                        
                        # Choose data source
                        data_source_options = ["Original Data"]
                        if st.session_state.basic_processed_outputs:
                            data_source_options.extend(list(st.session_state.basic_processed_outputs.keys()))
                        
                        data_source = st.selectbox(
                            "Select data source for interpolation:",
                            data_source_options,
                            key="interp_data_source"
                        )
                        
                        # Get the selected data
                        if data_source == "Original Data":
                            interp_data = st.session_state.data
                        else:
                            interp_data = st.session_state.basic_processed_outputs[data_source]
                        
                        # Check if there are any missing values
                        missing_counts = interp_data.isna().sum()
                        total_missing = missing_counts.sum()
                        
                        if total_missing == 0:
                            st.info("The selected data does not contain any missing values. Consider creating artificial missing values for testing interpolation.")
                            
                            # Option to create artificial missing values
                            create_missing = st.checkbox("Create artificial missing values", key="create_artificial_missing")
                            
                            if create_missing:
                                # Choose percentage of missing values
                                missing_pct = st.slider(
                                    "Percentage of values to make missing:",
                                    min_value=5,
                                    max_value=50,
                                    value=20,
                                    key="missing_pct"
                                )
                                
                                # Select columns where missing values will be created
                                missing_cols = st.multiselect(
                                    "Select columns where missing values will be created:",
                                    interp_data.columns.tolist(),
                                    key="missing_cols"
                                )
                                
                                # Create missing values
                                if st.button("Create Missing Values", key="create_missing_btn"):
                                    try:
                                        import numpy as np
                                        
                                        # Create a copy of the data
                                        data_with_missing = interp_data.copy()
                                        
                                        # Create missing values only in selected columns
                                        for col in missing_cols:
                                            # Calculate how many values to make missing
                                            n_missing = int(len(data_with_missing) * missing_pct / 100)
                                            
                                            # Create random indices for missing values
                                            missing_indices = np.random.choice(
                                                data_with_missing.index, 
                                                size=n_missing, 
                                                replace=False
                                            )
                                            
                                            # Set values as missing
                                            data_with_missing.loc[missing_indices, col] = np.nan
                                        
                                        # Store the data with artificial missing values
                                        st.session_state.basic_processed_outputs['data_with_missing'] = data_with_missing
                                        
                                        # Display the result
                                        st.success(f"Created artificial missing values in {len(missing_cols)} columns")
                                        st.write("#### Missing Value Counts After")
                                        missing_df = pd.DataFrame({
                                            'Column': data_with_missing.columns,
                                            'Missing Values': data_with_missing.isna().sum().values,
                                            'Percentage': (data_with_missing.isna().sum().values / len(data_with_missing) * 100).round(2)
                                        }).sort_values(by='Missing Values', ascending=False)
                                        
                                        st.dataframe(missing_df)
                                        
                                        # Update the interpolation data to use the one with missing values
                                        interp_data = data_with_missing
                                        
                                    except Exception as e:
                                        st.error(f"Error creating artificial missing values: {e}")
                        else:
                            # Display missing value statistics
                            st.write("#### Missing Value Statistics")
                            missing_df = pd.DataFrame({
                                'Column': interp_data.columns,
                                'Missing Values': interp_data.isna().sum().values,
                                'Percentage': (interp_data.isna().sum().values / len(interp_data) * 100).round(2)
                            }).sort_values(by='Missing Values', ascending=False)
                            
                            st.dataframe(missing_df)
                        
                        # Interpolation options
                        st.write("#### Interpolation Method")
                        interp_method = st.radio(
                            "Select interpolation method:",
                            ["Linear Interpolation", "Polynomial Interpolation", "Spline Interpolation", "Nearest Neighbor", "MCMC (Monte Carlo)"],
                            key="interp_method"
                        )
                        
                        # Method-specific options
                        if interp_method == "Polynomial Interpolation":
                            poly_order = st.slider(
                                "Polynomial order:",
                                min_value=1,
                                max_value=5,
                                value=2,
                                key="poly_order"
                            )
                        
                        elif interp_method == "Spline Interpolation":
                            spline_order = st.slider(
                                "Spline order:",
                                min_value=1,
                                max_value=5,
                                value=3,
                                key="spline_order"
                            )
                        
                        elif interp_method == "MCMC (Monte Carlo)":
                            st.info("This will use the Advanced Processing MCMC method.")
                            
                            # MCMC specific parameters
                            num_samples = st.slider(
                                "Number of MCMC samples", 
                                min_value=100, 
                                max_value=1000, 
                                value=500, 
                                step=100
                            )
                            
                            chains = st.slider(
                                "Number of MCMC chains", 
                                min_value=1, 
                                max_value=4, 
                                value=2
                            )
                        
                        # Select columns for interpolation
                        interp_columns = st.multiselect(
                            "Select columns for interpolation (leave empty for all columns with missing values):",
                            interp_data.columns.tolist(),
                            key="interp_columns"
                        )
                        
                        # If no columns are selected, use all columns with missing values
                        if not interp_columns:
                            interp_columns = missing_counts[missing_counts > 0].index.tolist()
                        
                        # Apply interpolation button
                        if st.button("Apply Interpolation", key="apply_interp_btn"):
                            try:
                                # Create a copy of the data
                                interpolated_data = interp_data.copy()
                                
                                # Apply the selected interpolation method
                                if interp_method == "Linear Interpolation":
                                    # Apply linear interpolation to selected columns
                                    for col in interp_columns:
                                        interpolated_data[col] = interpolated_data[col].interpolate(method='linear')
                                    
                                    st.success(f"Applied linear interpolation to {len(interp_columns)} columns")
                                
                                elif interp_method == "Polynomial Interpolation":
                                    # Apply polynomial interpolation to selected columns
                                    for col in interp_columns:
                                        if pd.api.types.is_numeric_dtype(interpolated_data[col]):
                                            interpolated_data[col] = interpolated_data[col].interpolate(
                                                method='polynomial', order=poly_order
                                            )
                                    
                                    st.success(f"Applied polynomial interpolation (order {poly_order}) to {len(interp_columns)} columns")
                                
                                elif interp_method == "Spline Interpolation":
                                    # Apply spline interpolation to selected columns
                                    for col in interp_columns:
                                        if pd.api.types.is_numeric_dtype(interpolated_data[col]):
                                            interpolated_data[col] = interpolated_data[col].interpolate(
                                                method='spline', order=spline_order
                                            )
                                    
                                    st.success(f"Applied spline interpolation (order {spline_order}) to {len(interp_columns)} columns")
                                
                                elif interp_method == "Nearest Neighbor":
                                    # Apply nearest neighbor interpolation to selected columns
                                    for col in interp_columns:
                                        interpolated_data[col] = interpolated_data[col].interpolate(
                                            method='nearest'
                                        )
                                    
                                    st.success(f"Applied nearest neighbor interpolation to {len(interp_columns)} columns")
                                
                                elif interp_method == "MCMC (Monte Carlo)":
                                    # Use the advanced processor for MCMC interpolation
                                    with st.spinner("Running MCMC interpolation... (this may take a while)"):
                                        interpolated_data = advanced_processor.mcmc_interpolation(
                                            interpolated_data,
                                            num_samples=num_samples,
                                            chains=chains
                                        )
                                    
                                    st.success(f"Applied MCMC interpolation to {len(interp_columns)} columns")
                                
                                # Store the interpolated data
                                st.session_state.basic_processed_outputs['interpolated_data'] = interpolated_data
                                
                                # Store in the interpolated_data session state for compatibility with advanced processing
                                st.session_state.interpolated_data = interpolated_data
                                
                                # Display missing value statistics after interpolation
                                st.write("#### Missing Value Counts After Interpolation")
                                missing_after_df = pd.DataFrame({
                                    'Column': interpolated_data.columns,
                                    'Missing Values': interpolated_data.isna().sum().values,
                                    'Percentage': (interpolated_data.isna().sum().values / len(interpolated_data) * 100).round(2)
                                }).sort_values(by='Missing Values', ascending=False)
                                
                                st.dataframe(missing_after_df)
                                
                                # Display the interpolated data
                                st.write("#### Interpolated Data Preview")
                                st.dataframe(interpolated_data.head())
                                
                            except Exception as e:
                                st.error(f"Error during interpolation: {e}")
                    
                    # 5. RESULTS TAB
                    with basic_tabs[4]:
                        st.write("### Processing Results")
                        st.write("Review and set processed data for further analysis:")
                        
                        # Display available processed outputs
                        if st.session_state.basic_processed_outputs:
                            st.write("#### Available Processed Outputs")
                            
                            # Create a table to show available outputs
                            output_info = []
                            for output_name, output_data in st.session_state.basic_processed_outputs.items():
                                output_info.append({
                                    'Output Name': output_name,
                                    'Shape': f"{output_data.shape[0]} rows × {output_data.shape[1]} columns",
                                    'Missing Values': output_data.isna().sum().sum(),
                                    'Data Types': len(output_data.dtypes.unique())
                                })
                            
                            output_df = pd.DataFrame(output_info)
                            st.dataframe(output_df)
                            
                            # Select output to view
                            selected_output = st.selectbox(
                                "Select output to view:",
                                list(st.session_state.basic_processed_outputs.keys()),
                                key="view_output"
                            )
                            
                            # Display selected output
                            if selected_output:
                                output_data = st.session_state.basic_processed_outputs[selected_output]
                                
                                st.write(f"#### Preview of '{selected_output}'")
                                st.dataframe(output_data.head())
                                
                                # Options for setting data in session state
                                st.write("#### Set Data for Analysis")
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    # Set as processed data (for prediction)
                                    if st.button(f"Use as Processed Data", key=f"use_processed_{selected_output}"):
                                        st.session_state.processed_data = output_data.copy()
                                        st.success(f"'{selected_output}' set as processed data for prediction!")
                                
                                with col2:
                                    # Set as interpolated data (for advanced processing)
                                    if st.button(f"Use for Advanced Processing", key=f"use_advanced_{selected_output}"):
                                        st.session_state.interpolated_data = output_data.copy()
                                        st.success(f"'{selected_output}' set as data for advanced processing!")
                                        
                            # Export options
                            st.write("#### Export Options")
                            
                            export_output = st.selectbox(
                                "Select output to export:",
                                list(st.session_state.basic_processed_outputs.keys()),
                                key="export_output"
                            )
                            
                            if export_output:
                                export_format = st.radio(
                                    "Export format:",
                                    ["CSV", "Excel"],
                                    key="export_format"
                                )
                                
                                output_data = st.session_state.basic_processed_outputs[export_output]
                                
                                try:
                                    if export_format == "CSV":
                                        data_bytes = data_handler.export_data(output_data, format='csv')
                                        st.download_button(
                                            label=f"Download {export_output} as CSV",
                                            data=data_bytes,
                                            file_name=f"{export_output}.csv",
                                            mime="text/csv"
                                        )
                                    else:  # Excel
                                        data_bytes = data_handler.export_data(output_data, format='excel')
                                        st.download_button(
                                            label=f"Download {export_output} as Excel",
                                            data=data_bytes,
                                            file_name=f"{export_output}.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                                except Exception as e:
                                    st.error(f"Error preparing download: {e}")
                            
                            # Save to database
                            st.write("#### Save to Database")
                            
                            db_output = st.selectbox(
                                "Select output to save to database:",
                                list(st.session_state.basic_processed_outputs.keys()),
                                key="db_output"
                            )
                            
                            if db_output:
                                save_name = st.text_input(
                                    "Dataset name:",
                                    value=f"{db_output}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}",
                                    key="save_name"
                                )
                                
                                save_desc = st.text_area(
                                    "Description (optional):",
                                    key="save_desc"
                                )
                                
                                if st.button("Save to Database", key="save_to_db"):
                                    try:
                                        output_data = st.session_state.basic_processed_outputs[db_output]
                                        
                                        # Save to database
                                        result = db_handler.save_dataset(
                                            output_data,
                                            name=save_name,
                                            description=save_desc,
                                            data_type="processed"
                                        )
                                        
                                        if result:
                                            st.success(f"Successfully saved '{save_name}' to database with ID: {result}")
                                        else:
                                            st.error("Failed to save dataset to database")
                                    
                                    except Exception as e:
                                        st.error(f"Error saving to database: {e}")
                        else:
                            st.info("No processed outputs available. Please perform processing operations in the other tabs first.")
            
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
                    interpolated_data = None
                    
                    # Get original data
                    if original_data is None:
                        st.warning("Original data is not available. Please import original data from the Data Import tab.")
                    
                    # Check if we have previously interpolated result
                    has_previous_result = 'interpolated_result' in st.session_state and st.session_state.interpolated_result is not None
                    has_interpolated_data = 'interpolated_data' in st.session_state and st.session_state.interpolated_data is not None
                    
                    if has_previous_result or has_interpolated_data:
                        # Choose data source options based on availability
                        source_options = []
                        if has_previous_result:
                            source_options.append("Use previously interpolated result")
                        if has_interpolated_data:
                            source_options.append("Use imported data for interpolation")
                        
                        if len(source_options) > 0:
                            interpolation_source = st.radio(
                                "Interpolated Data Source",
                                source_options,
                                index=0,
                                key="interpolation_source"
                            )
                            
                            if interpolation_source == "Use previously interpolated result" and has_previous_result:
                                interpolated_data = st.session_state.interpolated_result
                                st.success("Using previously interpolated result for analysis.")
                            elif has_interpolated_data:
                                interpolated_data = st.session_state.interpolated_data
                                st.success("Using imported data with missing values for interpolation.")
                    else:
                        st.info("No data available for interpolation. Please import data with missing values in the Data Import tab or create artificial missing values below.")
                    
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
                            
                            # Add control for the number of datasets to fill
                            st.write("#### Multiple Dataset Generation")
                            st.write("Generate multiple interpolated datasets with identical parameters for convergence testing:")
                            generate_multiple = st.checkbox("Generate multiple datasets", value=False)
                            
                            if generate_multiple:
                                num_datasets = st.slider("Number of datasets to generate", min_value=1, max_value=10, value=3, step=1)
                                dataset_prefix = st.text_input("Dataset name prefix", value="MCMC_Interpolation")
                        
                        # Run MCMC interpolation button
                        if st.button("Run MCMC Interpolation", key="mcmc_btn"):
                            try:
                                # First check if interpolated_data is None
                                if interpolated_data is None:
                                    st.error("No data available for interpolation. Please import or create data with missing values first.")
                                # Then check if we have missing values to interpolate
                                elif not interpolated_data.isna().any().any():
                                    st.warning("No missing values detected in the data. MCMC interpolation requires missing values.")
                                else:
                                    # Initialize datasets to store multiple results if needed
                                    generated_datasets = []
                                    
                                    # Determine how many datasets to generate
                                    iterations = num_datasets if generate_multiple else 1
                                    
                                    with st.spinner(f"Running MCMC interpolation for {iterations} dataset(s)... (this may take a while)"):
                                        # Generate the requested number of datasets
                                        for i in range(iterations):
                                            # Show progress for multiple datasets
                                            if generate_multiple:
                                                st.text(f"Generating dataset {i+1} of {iterations}...")
                                                
                                            # Run MCMC interpolation
                                            interpolated_result = advanced_processor.mcmc_interpolation(
                                                interpolated_data,
                                                num_samples=num_samples,
                                                chains=chains
                                            )
                                            
                                            # For the first dataset (or if only generating one), store as main result
                                            if i == 0:
                                                st.session_state.interpolated_result = interpolated_result
                                            
                                            # If generating multiple, add each to the results list with metadata
                                            if generate_multiple:
                                                dataset_info = {
                                                    'id': len(st.session_state.convergence_datasets) + i + 1,
                                                    'data': interpolated_result.copy(),
                                                    'convergence_scores': {},
                                                    'timestamp': pd.Timestamp.now(),
                                                    'name': f"{dataset_prefix}_{i+1}"
                                                }
                                                generated_datasets.append(dataset_info)
                                        
                                        # Show success message
                                        if generate_multiple:
                                            st.success(f"MCMC interpolation completed successfully for {iterations} datasets!")
                                        else:
                                            st.success("MCMC interpolation completed successfully!")
                                        
                                        # Add generated datasets to convergence analysis if requested
                                        if generate_multiple:
                                            if 'convergence_datasets' not in st.session_state:
                                                st.session_state.convergence_datasets = []
                                                st.session_state.convergence_iterations = 0
                                            
                                            # Add the datasets to the session state
                                            for dataset in generated_datasets:
                                                st.session_state.convergence_datasets.append(dataset)
                                                st.session_state.convergence_iterations += 1
                                            
                                            st.info(f"Added {iterations} datasets to the Multiple Imputation Analysis. Please proceed to that tab for analysis.")
                                        
                                        # Display side-by-side comparison of before and after
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            st.write("Before Interpolation:")
                                            st.dataframe(interpolated_data.head())
                                            
                                        with col2:
                                            st.write("After Interpolation:")
                                            st.dataframe(st.session_state.interpolated_result.head())
                                        
                                        # Show missing value counts before and after
                                        missing_before = interpolated_data.isna().sum().sum()
                                        missing_after = st.session_state.interpolated_result.isna().sum().sum()
                                        
                                        st.write(f"Missing values before: {missing_before}")
                                        st.write(f"Missing values after: {missing_after}")
                                        
                                        # Option to set as active dataset
                                        if st.button("Use Interpolated Result for Analysis", key="use_mcmc_result"):
                                            st.session_state.data = st.session_state.interpolated_result
                                            st.success("Interpolated result set as active dataset for analysis.")
                                        
                                        # Add download button for interpolated data
                                        try:
                                            data_bytes = data_handler.export_data(st.session_state.interpolated_result, format='csv')
                                            st.download_button(
                                                label="Download Interpolated Data as CSV",
                                                data=data_bytes,
                                                file_name="mcmc_interpolated_data.csv",
                                                mime="text/csv"
                                            )
                                        except Exception as e:
                                            st.error(f"Error preparing download: {e}")
                            except Exception as e:
                                st.error(f"Error during MCMC interpolation: {e}")
                    
                    # 2. MULTIPLE IMPUTATION ANALYSIS TAB
                    with advanced_options[1]:
                        st.write("### Multiple Imputation Analysis")
                        st.write("""
                        This tab performs advanced analysis on the interpolated dataset to evaluate convergence and 
                        statistical properties. The analysis includes three core methods:
                        1. Cluster Analysis (K-Means)
                        2. Regression Analysis (Linear Regression)
                        3. Factor Analysis (Principal Component Analysis - PCA)
                        """)
                        
                        # Check if we have MCMC interpolated result
                        if 'interpolated_result' not in st.session_state:
                            st.info("Please run MCMC interpolation first before performing multiple imputation analysis.")
                        else:
                            # Initialize convergence status
                            if 'convergence_status' not in st.session_state:
                                st.session_state.convergence_status = "Not evaluated"
                                st.session_state.convergence_iterations = 0
                                st.session_state.convergence_datasets = []
                                st.session_state.closest_convergence_dataset = None
                                
                            # Show current status
                            status_col1, status_col2 = st.columns(2)
                            with status_col1:
                                st.write("**Current Analysis Status**")
                                st.write(f"Convergence Status: {st.session_state.convergence_status}")
                                st.write(f"Iterations Completed: {st.session_state.convergence_iterations}")
                            
                            with status_col2:
                                st.write("**Datasets Information**")
                                if st.session_state.convergence_datasets:
                                    st.write(f"Number of Analyzed Datasets: {len(st.session_state.convergence_datasets)}")
                                    if st.session_state.closest_convergence_dataset is not None:
                                        st.write("Closest to Convergence: Dataset Available")
                                    else:
                                        st.write("Closest to Convergence: None Identified Yet")
                                else:
                                    st.write("No datasets analyzed yet.")

                            # Add current dataset to analysis
                            if st.button("Add Current Interpolated Dataset to Analysis", key="add_dataset_btn"):
                                if st.session_state.interpolated_result is not None:
                                    # Create a copy to avoid reference issues
                                    dataset_copy = st.session_state.interpolated_result.copy()
                                    
                                    # Add dataset to convergence analysis list
                                    dataset_info = {
                                        'id': len(st.session_state.convergence_datasets) + 1,
                                        'data': dataset_copy,
                                        'convergence_scores': {},
                                        'timestamp': pd.Timestamp.now()
                                    }
                                    
                                    st.session_state.convergence_datasets.append(dataset_info)
                                    st.session_state.convergence_iterations += 1
                                    
                                    st.success(f"Dataset {dataset_info['id']} added to analysis.")
                                    st.info("Please proceed with analysis methods below.")
                                else:
                                    st.error("No interpolated dataset available to add.")
                            
                            # Only show analysis options if we have datasets to analyze
                            if st.session_state.convergence_datasets:
                                # Add control module for consecutive analysis
                                with st.expander("Consecutive Analysis Control", expanded=True):
                                    st.write("""
                                    ### Run All Analysis Steps
                                    Use this control to run all analysis steps consecutively with a single click. 
                                    This will perform K-Means clustering, Linear Regression, and PCA analysis on all datasets, 
                                    followed by convergence evaluation.
                                    """)
                                    
                                    run_all = st.checkbox("Enable consecutive analysis", value=False)
                                    
                                    if run_all:
                                        datasets_to_analyze = st.multiselect(
                                            "Select datasets to analyze",
                                            options=[f"Dataset {ds['id']}" for ds in st.session_state.convergence_datasets],
                                            default=[f"Dataset {ds['id']}" for ds in st.session_state.convergence_datasets]
                                        )
                                        
                                        consecutive_btn = st.button("Run All Analysis Steps", key="run_all_btn")
                                        if consecutive_btn:
                                            st.session_state.consecutive_analysis = True
                                            st.session_state.datasets_to_analyze = datasets_to_analyze
                                            st.session_state.current_analysis_step = 0
                                            st.info("Starting consecutive analysis. Please wait while all steps are executed...")
                                    
                                # Analysis methods in tabs
                                analysis_tabs = st.tabs(["Dataset Selection", "Cluster Analysis (K-Means)", 
                                                        "Regression Analysis", "Factor Analysis (PCA)", 
                                                        "Convergence Evaluation"])
                                
                                # Initialize consecutive analysis if it doesn't exist
                                if 'consecutive_analysis' not in st.session_state:
                                    st.session_state.consecutive_analysis = False
                                    st.session_state.current_analysis_step = 0
                                    st.session_state.datasets_to_analyze = []
                                
                                # Handle consecutive analysis logic
                                if st.session_state.consecutive_analysis:
                                    # Get current step
                                    current_step = st.session_state.current_analysis_step
                                    
                                    # Execute analysis steps in order
                                    if current_step == 0:
                                        # Start with cluster analysis
                                        st.warning("Consecutive analysis: Running K-Means clustering analysis...")
                                        # This will be executed in the Cluster Analysis tab, so increase step
                                        st.session_state.current_analysis_step = 1
                                        # Redirect to that tab
                                        # Using rerun() here would be ideal but leads to an infinite loop
                                        # Instead, we'll let each tab check the step to see if processing is needed
                                    
                                    elif current_step == 1:
                                        # Move to regression analysis
                                        st.warning("Consecutive analysis: Running Regression analysis...")
                                        st.session_state.current_analysis_step = 2
                                    
                                    elif current_step == 2:
                                        # Move to PCA analysis
                                        st.warning("Consecutive analysis: Running PCA factor analysis...")
                                        st.session_state.current_analysis_step = 3
                                    
                                    elif current_step == 3:
                                        # Final step - convergence evaluation
                                        st.warning("Consecutive analysis: Running convergence evaluation...")
                                        st.session_state.current_analysis_step = 4
                                    
                                    elif current_step == 4:
                                        # Analysis complete
                                        st.success("Consecutive analysis completed successfully!")
                                        st.session_state.consecutive_analysis = False
                                        st.session_state.current_analysis_step = 0
                                
                                # 1. Dataset Selection Tab
                                with analysis_tabs[0]:
                                    st.write("### Dataset Selection and Comparison")
                                    st.write("Select datasets to analyze or compare:")
                                    
                                    # Create radio buttons to select dataset
                                    dataset_options = ["Original Data"] + [f"Interpolated Dataset {ds['id']}" for ds in st.session_state.convergence_datasets]
                                    selected_dataset = st.radio("Select dataset to view:", dataset_options)
                                    
                                    # Display selected dataset
                                    if selected_dataset == "Original Data":
                                        if original_data is not None:
                                            st.write("**Original Data Statistics**")
                                            st.dataframe(original_data.describe())
                                            
                                            # Show sample of data
                                            st.write("**Sample Data**")
                                            st.dataframe(original_data.head())
                                        else:
                                            st.warning("Original data not available.")
                                    else:
                                        # Extract dataset ID and find the corresponding dataset
                                        dataset_id = int(selected_dataset.split()[-1])
                                        dataset = next((ds for ds in st.session_state.convergence_datasets if ds['id'] == dataset_id), None)
                                        
                                        if dataset:
                                            st.write(f"**Interpolated Dataset {dataset_id} Statistics**")
                                            st.dataframe(dataset['data'].describe())
                                            
                                            # Show sample of data
                                            st.write("**Sample Data**")
                                            st.dataframe(dataset['data'].head())
                                            
                                            # Show timestamp
                                            st.write(f"Added on: {dataset['timestamp']}")
                                        else:
                                            st.error(f"Dataset {dataset_id} not found.")
                                    
                                    # Dataset comparison
                                    st.write("### Dataset Comparison")
                                    
                                    # Select datasets to compare
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        compare_ds1 = st.selectbox("First dataset:", dataset_options, key="compare_ds1")
                                    with col2:
                                        compare_ds2 = st.selectbox("Second dataset:", dataset_options, index=min(1, len(dataset_options)-1), key="compare_ds2")
                                    
                                    if st.button("Compare Datasets"):
                                        # Get data for the first dataset
                                        if compare_ds1 == "Original Data":
                                            data1 = original_data
                                            label1 = "Original Data"
                                        else:
                                            ds1_id = int(compare_ds1.split()[-1])
                                            ds1 = next((ds for ds in st.session_state.convergence_datasets if ds['id'] == ds1_id), None)
                                            data1 = ds1['data'] if ds1 else None
                                            label1 = f"Dataset {ds1_id}"
                                        
                                        # Get data for the second dataset
                                        if compare_ds2 == "Original Data":
                                            data2 = original_data
                                            label2 = "Original Data"
                                        else:
                                            ds2_id = int(compare_ds2.split()[-1])
                                            ds2 = next((ds for ds in st.session_state.convergence_datasets if ds['id'] == ds2_id), None)
                                            data2 = ds2['data'] if ds2 else None
                                            label2 = f"Dataset {ds2_id}"
                                        
                                        # Perform comparison if both datasets are available
                                        if data1 is not None and data2 is not None:
                                            # Common numeric columns
                                            numeric_cols = list(set(data1.select_dtypes(include=np.number).columns) & 
                                                            set(data2.select_dtypes(include=np.number).columns))
                                            
                                            if numeric_cols:
                                                # Let user select a column to compare
                                                selected_col = st.selectbox("Select column to compare:", numeric_cols)
                                                
                                                # Create histogram comparison
                                                fig = plt.figure(figsize=(10, 6))
                                                plt.hist(data1[selected_col].dropna(), alpha=0.5, label=label1)
                                                plt.hist(data2[selected_col].dropna(), alpha=0.5, label=label2)
                                                plt.xlabel(selected_col)
                                                plt.ylabel('Frequency')
                                                plt.title(f'Distribution Comparison for {selected_col}')
                                                plt.legend()
                                                st.pyplot(fig)
                                                
                                                # Calculate statistics for comparison
                                                if selected_col:
                                                    stats_df = pd.DataFrame({
                                                        'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Median'],
                                                        label1: [
                                                            data1[selected_col].mean(),
                                                            data1[selected_col].std(),
                                                            data1[selected_col].min(),
                                                            data1[selected_col].max(),
                                                            data1[selected_col].median()
                                                        ],
                                                        label2: [
                                                            data2[selected_col].mean(),
                                                            data2[selected_col].std(),
                                                            data2[selected_col].min(),
                                                            data2[selected_col].max(),
                                                            data2[selected_col].median()
                                                        ]
                                                    })
                                                    st.dataframe(stats_df)
                                            else:
                                                st.warning("No common numeric columns found between the selected datasets.")
                                        else:
                                            st.error("One or both of the selected datasets are not available.")
                                
                                # 2. Cluster Analysis (K-Means) Tab
                                with analysis_tabs[1]:
                                    st.write("### Cluster Analysis (K-Means)")
                                    st.write("""
                                    K-Means clustering groups data points into k clusters based on similarity.
                                    This analysis helps evaluate if the interpolated data preserves the 
                                    cluster structure of the original data.
                                    """)
                                    
                                    # Check if we're in consecutive analysis mode
                                    consecutive_mode = False
                                    if ('consecutive_analysis' in st.session_state and 
                                        st.session_state.consecutive_analysis and 
                                        st.session_state.current_analysis_step == 1):
                                        consecutive_mode = True
                                        st.info("Running K-Means clustering as part of consecutive analysis...")
                                    
                                    # Select dataset to analyze
                                    dataset_options = ["Original Data"] + [f"Interpolated Dataset {ds['id']}" for ds in st.session_state.convergence_datasets]
                                    
                                    # In consecutive mode, analyze all selected datasets
                                    if consecutive_mode and st.session_state.datasets_to_analyze:
                                        selected_datasets = st.session_state.datasets_to_analyze
                                        st.write(f"Analyzing datasets: {', '.join(selected_datasets)}")
                                        
                                        # Process one dataset now, others will be processed in a loop
                                        # Default to first dataset for parameters
                                        selected_dataset = selected_datasets[0] if selected_datasets else dataset_options[0]
                                    else:
                                        # Regular mode - select a single dataset
                                        selected_dataset = st.selectbox("Select dataset to analyze:", dataset_options, key="kmeans_dataset")
                                    
                                    # Get the selected dataset for parameters
                                    if selected_dataset == "Original Data":
                                        analysis_data = original_data
                                        dataset_label = "Original Data"
                                    else:
                                        dataset_id = int(selected_dataset.split()[-1])
                                        dataset = next((ds for ds in st.session_state.convergence_datasets if ds['id'] == dataset_id), None)
                                        analysis_data = dataset['data'] if dataset else None
                                        dataset_label = f"Dataset {dataset_id}"
                                    
                                    if analysis_data is not None:
                                        # Parameters for K-Means
                                        with st.expander("Clustering Parameters", expanded=True):
                                            # Select features for clustering
                                            numeric_cols = analysis_data.select_dtypes(include=np.number).columns.tolist()
                                            selected_features = st.multiselect(
                                                "Select features for clustering:",
                                                numeric_cols,
                                                default=numeric_cols[:min(3, len(numeric_cols))]
                                            )
                                            
                                            # Number of clusters
                                            k_clusters = st.slider("Number of clusters (k):", min_value=2, max_value=10, value=3)
                                            
                                            # Max iterations
                                            max_iter = st.slider("Maximum iterations:", min_value=100, max_value=1000, value=300, step=100)
                                            
                                            # Random state for reproducibility
                                            random_state = st.slider("Random state:", min_value=0, max_value=100, value=42)
                                        
                                        # Run clustering
                                        run_button_clicked = False
                                        if consecutive_mode:
                                            # In consecutive mode, automatically trigger analysis
                                            run_button_clicked = st.session_state.current_analysis_step == 1
                                            if run_button_clicked:
                                                st.success("Automatically running K-Means clustering for all selected datasets...")
                                        else:
                                            # In regular mode, user has to click button
                                            run_button_clicked = st.button("Run K-Means Clustering", key="run_kmeans_btn")
                                            
                                        if run_button_clicked:
                                            if len(selected_features) < 2:
                                                st.error("Please select at least 2 features for clustering.")
                                                if consecutive_mode:
                                                    # Even in error, move to next step in consecutive mode
                                                    st.session_state.current_analysis_step = 2
                                            else:
                                                try:
                                                    # In consecutive mode with multiple datasets selected
                                                    if consecutive_mode and st.session_state.datasets_to_analyze and len(st.session_state.datasets_to_analyze) > 0:
                                                        # Process each selected dataset in turn
                                                        datasets_to_process = st.session_state.datasets_to_analyze
                                                        
                                                        # Create dictionary to store results for all datasets
                                                        all_kmeans_results = {}
                                                        
                                                        with st.spinner(f"Running K-Means clustering on {len(datasets_to_process)} datasets..."):
                                                            for selected_ds in datasets_to_process:
                                                                st.write(f"Processing {selected_ds}...")
                                                                
                                                                # Get data for this dataset
                                                                if selected_ds == "Original Data":
                                                                    curr_data = original_data.copy()
                                                                    curr_label = "Original Data"
                                                                else:
                                                                    ds_id = int(selected_ds.split()[-1])
                                                                    dataset = next((ds for ds in st.session_state.convergence_datasets if ds['id'] == ds_id), None)
                                                                    if dataset is not None:
                                                                        curr_data = dataset['data'].copy()
                                                                        curr_label = f"Dataset {ds_id}"
                                                                    else:
                                                                        st.warning(f"Dataset {selected_ds} not found, skipping...")
                                                                        continue
                                                                
                                                                # Import required libraries
                                                                from sklearn.cluster import KMeans
                                                                from sklearn.preprocessing import StandardScaler
                                                                
                                                                # Prepare data for clustering
                                                                cluster_data = curr_data[selected_features].dropna()
                                                                
                                                                if len(cluster_data) < k_clusters:
                                                                    st.warning(f"{curr_label}: Not enough data points ({len(cluster_data)}) for {k_clusters} clusters after removing missing values.")
                                                                    continue
                                                                
                                                                # Scale the data
                                                                scaler = StandardScaler()
                                                                scaled_data = scaler.fit_transform(cluster_data)
                                                                
                                                                # Run K-Means
                                                                kmeans = KMeans(n_clusters=k_clusters, max_iter=max_iter, random_state=random_state)
                                                                clusters = kmeans.fit_predict(scaled_data)
                                                                
                                                                # Add cluster labels to the data
                                                                cluster_data['Cluster'] = clusters
                                                                
                                                                # Calculate convergence metrics
                                                                inertia = kmeans.inertia_  # Sum of squared distances to the nearest centroid
                                                                
                                                                # Store clustering results in dataset info if not original data
                                                                if selected_ds != "Original Data":
                                                                    ds_id = int(selected_ds.split()[-1])
                                                                    for ds in st.session_state.convergence_datasets:
                                                                        if ds['id'] == ds_id:
                                                                            ds['convergence_scores']['kmeans_inertia'] = inertia
                                                                            ds['convergence_scores']['kmeans_cluster_counts'] = cluster_data['Cluster'].value_counts().sort_index().to_dict()
                                                                            break
                                                                
                                                                # Store results for later display if needed
                                                                all_kmeans_results[curr_label] = {
                                                                    'inertia': inertia,
                                                                    'data': cluster_data,
                                                                    'kmeans': kmeans,
                                                                    'scaler': scaler
                                                                }
                                                            
                                                            # Report completion
                                                            if all_kmeans_results:
                                                                st.success(f"K-Means clustering completed for {len(all_kmeans_results)} datasets.")
                                                                
                                                                # Show summary of results
                                                                st.write("### Clustering Results Summary")
                                                                summary_data = {label: {'Inertia': results['inertia']} 
                                                                            for label, results in all_kmeans_results.items()}
                                                                
                                                                summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
                                                                st.dataframe(summary_df)
                                                                
                                                                # Show the first dataset's visualization as an example
                                                                if len(selected_features) >= 2:
                                                                    example_label = list(all_kmeans_results.keys())[0]
                                                                    example_results = all_kmeans_results[example_label]
                                                                    
                                                                    st.write(f"#### Example Visualization: {example_label}")
                                                                    viz_features = selected_features[:2]
                                                                    
                                                                    fig, ax = plt.subplots(figsize=(10, 8))
                                                                    scatter = ax.scatter(
                                                                        example_results['data'][viz_features[0]], 
                                                                        example_results['data'][viz_features[1]], 
                                                                        c=example_results['data']['Cluster'], 
                                                                        cmap='viridis', 
                                                                        alpha=0.6,
                                                                        s=50
                                                                    )
                                                                    
                                                                    # Plot centroids
                                                                    centroids = example_results['scaler'].inverse_transform(example_results['kmeans'].cluster_centers_)
                                                                    ax.scatter(
                                                                        centroids[:, 0], 
                                                                        centroids[:, 1],
                                                                        marker='X',
                                                                        s=200,
                                                                        linewidths=2,
                                                                        color='red',
                                                                        label='Centroids'
                                                                    )
                                                                    
                                                                    ax.set_xlabel(viz_features[0])
                                                                    ax.set_ylabel(viz_features[1])
                                                                    ax.set_title(f'K-Means Clustering ({k_clusters} clusters)')
                                                                    ax.legend()
                                                                    plt.colorbar(scatter, label='Cluster')
                                                                    st.pyplot(fig)
                                                            
                                                            # Progress to next step
                                                            if consecutive_mode:
                                                                st.session_state.current_analysis_step = 2
                                                                st.success("K-Means clustering complete for all datasets. Moving to Regression Analysis...")
                                                                # Force a rerun to update the UI for the next step
                                                                st.rerun()
                                                    else:
                                                        # Regular mode - process single dataset
                                                        with st.spinner("Running K-Means clustering..."):
                                                            # Prepare data for clustering
                                                            from sklearn.cluster import KMeans
                                                            from sklearn.preprocessing import StandardScaler
                                                            
                                                            # Get data without missing values
                                                            cluster_data = analysis_data[selected_features].dropna()
                                                            
                                                            if len(cluster_data) < k_clusters:
                                                                st.error(f"Not enough data points ({len(cluster_data)}) for {k_clusters} clusters after removing missing values.")
                                                            else:
                                                                # Scale the data
                                                                scaler = StandardScaler()
                                                                scaled_data = scaler.fit_transform(cluster_data)
                                                                
                                                                # Run K-Means
                                                                kmeans = KMeans(n_clusters=k_clusters, max_iter=max_iter, random_state=random_state)
                                                                clusters = kmeans.fit_predict(scaled_data)
                                                                
                                                                # Add cluster labels to the data
                                                                cluster_data['Cluster'] = clusters
                                                                
                                                                # Calculate convergence metrics
                                                                inertia = kmeans.inertia_  # Sum of squared distances to the nearest centroid
                                                                
                                                                # Display results
                                                                st.success(f"K-Means clustering completed successfully for {dataset_label}.")
                                                                
                                                                # Visualization
                                                                st.write("### Clustering Results")
                                                                st.write(f"Inertia (Sum of squared distances): {inertia:.2f}")
                                                                
                                                                # Display cluster distribution
                                                                st.write("#### Cluster Distribution")
                                                                cluster_counts = cluster_data['Cluster'].value_counts().sort_index()
                                                                fig = plt.figure(figsize=(10, 6))
                                                                plt.bar(cluster_counts.index, cluster_counts.values)
                                                                plt.xlabel('Cluster')
                                                                plt.ylabel('Number of Data Points')
                                                                plt.title('Distribution of Data Points Across Clusters')
                                                                plt.xticks(range(k_clusters))
                                                                st.pyplot(fig)
                                                                
                                                                # Store clustering results in dataset info
                                                                if selected_dataset != "Original Data":
                                                                    dataset_id = int(selected_dataset.split()[-1])
                                                                    for ds in st.session_state.convergence_datasets:
                                                                        if ds['id'] == dataset_id:
                                                                            ds['convergence_scores']['kmeans_inertia'] = inertia
                                                                            ds['convergence_scores']['kmeans_cluster_counts'] = cluster_counts.to_dict()
                                                                            break
                                                                
                                                                # Create a 2D visualization if possible
                                                                if len(selected_features) >= 2:
                                                                    st.write("#### 2D Visualization of Clusters")
                                                                    
                                                                    # Select two features for visualization
                                                                    viz_features = selected_features[:2]
                                                                    
                                                                    fig, ax = plt.subplots(figsize=(10, 8))
                                                                    scatter = ax.scatter(
                                                                        cluster_data[viz_features[0]], 
                                                                        cluster_data[viz_features[1]], 
                                                                        c=cluster_data['Cluster'], 
                                                                        cmap='viridis', 
                                                                        alpha=0.6,
                                                                        s=50
                                                                    )
                                                                    
                                                                    # Plot centroids
                                                                    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
                                                                    ax.scatter(
                                                                        centroids[:, 0], 
                                                                        centroids[:, 1],
                                                                        marker='X',
                                                                        s=200,
                                                                        linewidths=2,
                                                                        color='red',
                                                                        label='Centroids'
                                                                    )
                                                                    
                                                                    ax.set_xlabel(viz_features[0])
                                                                    ax.set_ylabel(viz_features[1])
                                                                    ax.set_title(f'K-Means Clustering ({k_clusters} clusters)')
                                                                    ax.legend()
                                                                    plt.colorbar(scatter, label='Cluster')
                                                                    st.pyplot(fig)
                                                                
                                                                # If in consecutive mode, progress to next step
                                                                if consecutive_mode:
                                                                    st.session_state.current_analysis_step = 2
                                                                    st.success("K-Means clustering complete. Moving to Regression Analysis...")
                                                                    # Force a rerun to update the UI for the next step
                                                                    st.rerun()
                                                
                                                except Exception as e:
                                                    st.error(f"Error during K-Means clustering: {e}")
                                                    # Even on error, if in consecutive mode, move to next step
                                                    if consecutive_mode:
                                                        st.session_state.current_analysis_step = 2
                                    else:
                                        st.error(f"Dataset {selected_dataset} is not available.")
                                
                                # 3. Regression Analysis Tab
                                with analysis_tabs[2]:
                                    st.write("### Regression Analysis")
                                    st.write("""
                                    Linear regression analyzes the relationship between variables.
                                    We'll use it to evaluate if interpolated data maintains the same variable relationships as the original data.
                                    """)
                                    
                                    # Select dataset to analyze
                                    dataset_options = ["Original Data"] + [f"Interpolated Dataset {ds['id']}" for ds in st.session_state.convergence_datasets]
                                    
                                    # In consecutive mode, analyze all selected datasets
                                    if consecutive_mode and st.session_state.datasets_to_analyze:
                                        selected_datasets = st.session_state.datasets_to_analyze
                                        st.write(f"Analyzing datasets: {', '.join(selected_datasets)}")
                                        
                                        # Process one dataset now, others will be processed in a loop
                                        # Default to first dataset for parameters
                                        selected_dataset = selected_datasets[0] if selected_datasets else dataset_options[0]
                                    else:
                                        # Regular mode - select a single dataset
                                        selected_dataset = st.selectbox("Select dataset to analyze:", dataset_options, key="regression_dataset")
                                    
                                    # Get the selected dataset for parameters
                                    if selected_dataset == "Original Data":
                                        analysis_data = original_data
                                        dataset_label = "Original Data"
                                    else:
                                        dataset_id = int(selected_dataset.split()[-1])
                                        dataset = next((ds for ds in st.session_state.convergence_datasets if ds['id'] == dataset_id), None)
                                        analysis_data = dataset['data'] if dataset else None
                                        dataset_label = f"Dataset {dataset_id}"
                                    
                                    if analysis_data is not None:
                                        # Parameters for regression
                                        with st.expander("Regression Parameters", expanded=True):
                                            # Get all numeric columns
                                            numeric_cols = analysis_data.select_dtypes(include=np.number).columns.tolist()
                                            
                                            # Select dependent variable (y)
                                            dependent_var = st.selectbox(
                                                "Select dependent variable (y):",
                                                numeric_cols
                                            )
                                            
                                            # Select independent variables (X)
                                            independent_vars = st.multiselect(
                                                "Select independent variables (X):",
                                                [col for col in numeric_cols if col != dependent_var],
                                                default=[col for col in numeric_cols[:min(3, len(numeric_cols))] if col != dependent_var]
                                            )
                                            
                                            # Test size
                                            test_size = st.slider("Test set size (%):", min_value=10, max_value=50, value=20) / 100
                                            
                                            # Random state for reproducibility
                                            random_state = st.slider("Random state:", min_value=0, max_value=100, value=42, key="reg_random_state")
                                        
                                        # Check if we're in consecutive analysis mode
                                        consecutive_mode = False
                                        if ('consecutive_analysis' in st.session_state and 
                                            st.session_state.consecutive_analysis and 
                                            st.session_state.current_analysis_step == 2):
                                            consecutive_mode = True
                                            st.info("Running Regression Analysis as part of consecutive analysis...")
                                        
                                        # Run regression
                                        run_button_clicked = False
                                        if consecutive_mode:
                                            # In consecutive mode, automatically trigger analysis
                                            run_button_clicked = st.session_state.current_analysis_step == 2
                                            if run_button_clicked:
                                                st.success("Automatically running Linear Regression for selected datasets...")
                                        else:
                                            # In regular mode, user has to click button
                                            run_button_clicked = st.button("Run Linear Regression", key="run_regression_btn")
                                            
                                        if run_button_clicked:
                                            if not independent_vars:
                                                st.error("Please select at least one independent variable.")
                                                if consecutive_mode:
                                                    # Even in error, move to next step
                                                    st.session_state.current_analysis_step = 3
                                            else:
                                                try:
                                                    # In consecutive mode with multiple datasets selected
                                                    if consecutive_mode and st.session_state.datasets_to_analyze and len(st.session_state.datasets_to_analyze) > 0:
                                                        # Process each selected dataset in turn
                                                        datasets_to_process = st.session_state.datasets_to_analyze
                                                        
                                                        # Create dictionary to store results for all datasets
                                                        all_regression_results = {}
                                                        
                                                        with st.spinner(f"Running Linear Regression on {len(datasets_to_process)} datasets..."):
                                                            for selected_ds in datasets_to_process:
                                                                st.write(f"Processing {selected_ds}...")
                                                                
                                                                # Get data for this dataset
                                                                if selected_ds == "Original Data":
                                                                    curr_data = original_data.copy()
                                                                    curr_label = "Original Data"
                                                                else:
                                                                    ds_id = int(selected_ds.split()[-1])
                                                                    dataset = next((ds for ds in st.session_state.convergence_datasets if ds['id'] == ds_id), None)
                                                                    if dataset is not None:
                                                                        curr_data = dataset['data'].copy()
                                                                        curr_label = f"Dataset {ds_id}"
                                                                    else:
                                                                        st.warning(f"Dataset {selected_ds} not found, skipping...")
                                                                        continue
                                                                
                                                                # Import required libraries
                                                                from sklearn.linear_model import LinearRegression
                                                                from sklearn.metrics import mean_squared_error, r2_score
                                                                from sklearn.model_selection import train_test_split
                                                                
                                                                # Prepare data for regression
                                                                reg_data = curr_data[independent_vars + [dependent_var]].dropna()
                                                                
                                                                if len(reg_data) < 10:  # Minimum sample size
                                                                    st.warning(f"{curr_label}: Not enough data points ({len(reg_data)}) after removing missing values.")
                                                                    continue
                                                                # Split data for this dataset
                                                                X = reg_data[independent_vars]
                                                                y = reg_data[dependent_var]
                                                                
                                                                X_train, X_test, y_train, y_test = train_test_split(
                                                                    X, y, test_size=test_size, random_state=random_state
                                                                )
                                                                
                                                                # Fit model
                                                                model = LinearRegression()
                                                                model.fit(X_train, y_train)
                                                                
                                                                # Predictions
                                                                y_train_pred = model.predict(X_train)
                                                                y_test_pred = model.predict(X_test)
                                                                
                                                                # Metrics
                                                                train_mse = mean_squared_error(y_train, y_train_pred)
                                                                test_mse = mean_squared_error(y_test, y_test_pred)
                                                                train_r2 = r2_score(y_train, y_train_pred)
                                                                test_r2 = r2_score(y_test, y_test_pred)
                                                                
                                                                # Store regression results in dataset info if not original data
                                                                if selected_ds != "Original Data":
                                                                    ds_id = int(selected_ds.split()[-1])
                                                                    for ds in st.session_state.convergence_datasets:
                                                                        if ds['id'] == ds_id:
                                                                            ds['convergence_scores']['regression_test_r2'] = test_r2
                                                                            ds['convergence_scores']['regression_test_mse'] = test_mse
                                                                            ds['convergence_scores']['regression_coefficients'] = {
                                                                                'intercept': float(model.intercept_),
                                                                                'coef': {feat: float(coef) for feat, coef in zip(independent_vars, model.coef_)}
                                                                            }
                                                                            break
                                                                
                                                                # Store results for later display
                                                                all_regression_results[curr_label] = {
                                                                    'model': model,
                                                                    'train_mse': train_mse,
                                                                    'test_mse': test_mse,
                                                                    'train_r2': train_r2,
                                                                    'test_r2': test_r2,
                                                                    'y_train': y_train,
                                                                    'y_train_pred': y_train_pred,
                                                                    'y_test': y_test,
                                                                    'y_test_pred': y_test_pred,
                                                                    'independent_vars': independent_vars,
                                                                    'coef': model.coef_,
                                                                    'intercept': model.intercept_
                                                                }
                                                            
                                                            # Report completion
                                                            if all_regression_results:
                                                                st.success(f"Linear Regression completed for {len(all_regression_results)} datasets.")
                                                                
                                                                # Show summary of results
                                                                st.write("### Regression Results Summary")
                                                                summary_data = {
                                                                    label: {
                                                                        'Test R²': f"{results['test_r2']:.4f}",
                                                                        'Test MSE': f"{results['test_mse']:.4f}",
                                                                    }
                                                                    for label, results in all_regression_results.items()
                                                                }
                                                                
                                                                summary_df = pd.DataFrame.from_dict(summary_data, orient='index')
                                                                st.dataframe(summary_df)
                                                                
                                                                # Show the first dataset's visualization as an example
                                                                example_label = list(all_regression_results.keys())[0]
                                                                example_results = all_regression_results[example_label]
                                                                
                                                                st.write(f"#### Example Visualization: {example_label}")
                                                                
                                                                # Model coefficients
                                                                st.write("##### Model Coefficients")
                                                                coef_df = pd.DataFrame({
                                                                    'Feature': example_results['independent_vars'],
                                                                    'Coefficient': example_results['coef']
                                                                })
                                                                st.dataframe(coef_df)
                                                                st.write(f"Intercept: {example_results['intercept']:.4f}")
                                                                
                                                                # Visualize predictions vs actual
                                                                st.write("##### Predictions vs Actual")
                                                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                                                
                                                                # Training set
                                                                ax1.scatter(example_results['y_train'], example_results['y_train_pred'], alpha=0.5)
                                                                y_min = min(example_results['y_train'].min(), example_results['y_train_pred'].min())
                                                                y_max = max(example_results['y_train'].max(), example_results['y_train_pred'].max())
                                                                ax1.plot([y_min, y_max], [y_min, y_max], 'r--')
                                                                ax1.set_xlabel('Actual')
                                                                ax1.set_ylabel('Predicted')
                                                                ax1.set_title(f'Training Set (R² = {example_results["train_r2"]:.4f})')
                                                                
                                                                # Test set
                                                                ax2.scatter(example_results['y_test'], example_results['y_test_pred'], alpha=0.5)
                                                                y_min = min(example_results['y_test'].min(), example_results['y_test_pred'].min())
                                                                y_max = max(example_results['y_test'].max(), example_results['y_test_pred'].max())
                                                                ax2.plot([y_min, y_max], [y_min, y_max], 'r--')
                                                                ax2.set_xlabel('Actual')
                                                                ax2.set_ylabel('Predicted')
                                                                ax2.set_title(f'Test Set (R² = {example_results["test_r2"]:.4f})')
                                                                
                                                                plt.tight_layout()
                                                                st.pyplot(fig)
                                                            
                                                            # Progress to next step
                                                            if consecutive_mode:
                                                                st.session_state.current_analysis_step = 3
                                                                st.success("Regression analysis complete for all datasets. Moving to PCA Analysis...")
                                                                # Force a rerun to update the UI for the next step
                                                                st.rerun()
                                                    else:
                                                        # Regular mode - process single dataset
                                                        with st.spinner("Running linear regression..."):
                                                            from sklearn.linear_model import LinearRegression
                                                            from sklearn.metrics import mean_squared_error, r2_score
                                                            from sklearn.model_selection import train_test_split
                                                            
                                                            # Prepare data
                                                            reg_data = analysis_data[independent_vars + [dependent_var]].dropna()
                                                            
                                                            if len(reg_data) < 10:  # Minimum sample size
                                                                st.error(f"Not enough data points ({len(reg_data)}) after removing missing values.")
                                                            else:
                                                                # Split data
                                                                X = reg_data[independent_vars]
                                                                y = reg_data[dependent_var]
                                                                
                                                                X_train, X_test, y_train, y_test = train_test_split(
                                                                    X, y, test_size=test_size, random_state=random_state
                                                                )
                                                                
                                                                # Fit model
                                                                model = LinearRegression()
                                                                model.fit(X_train, y_train)
                                                                
                                                                # Predictions
                                                                y_train_pred = model.predict(X_train)
                                                                y_test_pred = model.predict(X_test)
                                                                
                                                                # Metrics
                                                                train_mse = mean_squared_error(y_train, y_train_pred)
                                                                test_mse = mean_squared_error(y_test, y_test_pred)
                                                                train_r2 = r2_score(y_train, y_train_pred)
                                                                test_r2 = r2_score(y_test, y_test_pred)
                                                                
                                                                # Display results
                                                                st.success(f"Linear regression completed successfully for {dataset_label}.")
                                                                
                                                                # Model coefficients
                                                                st.write("### Model Coefficients")
                                                                coef_df = pd.DataFrame({
                                                                    'Feature': independent_vars,
                                                                    'Coefficient': model.coef_
                                                                })
                                                                st.dataframe(coef_df)
                                                                st.write(f"Intercept: {model.intercept_:.4f}")
                                                                
                                                                # Model performance
                                                                st.write("### Model Performance")
                                                                metrics_df = pd.DataFrame({
                                                                    'Metric': ['Mean Squared Error (MSE)', 'R² Score'],
                                                                    'Training Set': [train_mse, train_r2],
                                                                    'Test Set': [test_mse, test_r2]
                                                                })
                                                                st.dataframe(metrics_df)
                                                                
                                                                # Store regression results in dataset info
                                                                if selected_dataset != "Original Data":
                                                                    dataset_id = int(selected_dataset.split()[-1])
                                                                    for ds in st.session_state.convergence_datasets:
                                                                        if ds['id'] == dataset_id:
                                                                            ds['convergence_scores']['regression_test_r2'] = test_r2
                                                                            ds['convergence_scores']['regression_test_mse'] = test_mse
                                                                            ds['convergence_scores']['regression_coefficients'] = {
                                                                                'intercept': float(model.intercept_),
                                                                                'coef': {feat: float(coef) for feat, coef in zip(independent_vars, model.coef_)}
                                                                            }
                                                                            break
                                                                
                                                                # Visualize predictions vs actual
                                                                st.write("### Predictions vs Actual")
                                                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                                                
                                                                # Training set
                                                                ax1.scatter(y_train, y_train_pred, alpha=0.5)
                                                                ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
                                                                ax1.set_xlabel('Actual')
                                                                ax1.set_ylabel('Predicted')
                                                                ax1.set_title('Training Set')
                                                                
                                                                # Test set
                                                                ax2.scatter(y_test, y_test_pred, alpha=0.5)
                                                                ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
                                                                ax2.set_xlabel('Actual')
                                                                ax2.set_ylabel('Predicted')
                                                                ax2.set_title('Test Set')
                                                            
                                                                plt.tight_layout()
                                                                st.pyplot(fig)
                                                                
                                                                # Display regression equation
                                                                eq = f"y = {model.intercept_:.4f}"
                                                                for i, var in enumerate(independent_vars):
                                                                    eq += f" + ({model.coef_[i]:.4f} × {var})"
                                                                
                                                                st.write("### Regression Equation")
                                                                st.write(eq)
                                                            
                                                                # If in consecutive mode, progress to next step
                                                                if consecutive_mode:
                                                                    st.session_state.current_analysis_step = 3
                                                                    st.success("Linear Regression complete. Moving to PCA Factor Analysis...")
                                                                    # Force a rerun to update the UI for the next step
                                                                    st.rerun()
                                                
                                                except Exception as e:
                                                    st.error(f"Error during linear regression: {e}")
                                                    # Even on error, if in consecutive mode, move to next step
                                                    if consecutive_mode:
                                                        st.session_state.current_analysis_step = 3
                                    else:
                                        st.error(f"Dataset {selected_dataset} is not available.")
                                
                                # 4. Factor Analysis (PCA) Tab
                                with analysis_tabs[3]:
                                    st.write("### Factor Analysis (PCA)")
                                    st.write("""
                                    Principal Component Analysis (PCA) reduces dimensionality while preserving variance.
                                    This analysis helps evaluate if the interpolated data maintains the same underlying factors.
                                    """)
                                    
                                    # Select dataset to analyze
                                    dataset_options = ["Original Data"] + [f"Interpolated Dataset {ds['id']}" for ds in st.session_state.convergence_datasets]
                                    selected_dataset = st.selectbox("Select dataset to analyze:", dataset_options, key="pca_dataset")
                                    
                                    # Get the selected dataset
                                    if selected_dataset == "Original Data":
                                        analysis_data = original_data
                                        dataset_label = "Original Data"
                                    else:
                                        dataset_id = int(selected_dataset.split()[-1])
                                        dataset = next((ds for ds in st.session_state.convergence_datasets if ds['id'] == dataset_id), None)
                                        analysis_data = dataset['data'] if dataset else None
                                        dataset_label = f"Dataset {dataset_id}"
                                    
                                    if analysis_data is not None:
                                        # Parameters for PCA
                                        with st.expander("PCA Parameters", expanded=True):
                                            # Get all numeric columns
                                            numeric_cols = analysis_data.select_dtypes(include=np.number).columns.tolist()
                                            
                                            # Select features for PCA
                                            selected_features = st.multiselect(
                                                "Select features for PCA:",
                                                numeric_cols,
                                                default=numeric_cols[:min(5, len(numeric_cols))]
                                            )
                                            
                                            # Number of components
                                            n_components = st.slider(
                                                "Number of principal components:",
                                                min_value=2,
                                                max_value=min(len(selected_features), 10),
                                                value=min(3, len(selected_features))
                                            )
                                            
                                            # Random state
                                            random_state = st.slider("Random state:", min_value=0, max_value=100, value=42, key="pca_random_state")
                                        
                                        # Check if we're in consecutive analysis mode
                                        consecutive_mode = False
                                        if ('consecutive_analysis' in st.session_state and 
                                            st.session_state.consecutive_analysis and 
                                            st.session_state.current_analysis_step == 3):
                                            consecutive_mode = True
                                            st.info("Running PCA Factor Analysis as part of consecutive analysis...")
                                        
                                        # Run PCA
                                        run_button_clicked = False
                                        if consecutive_mode:
                                            # In consecutive mode, automatically trigger analysis
                                            run_button_clicked = st.session_state.current_analysis_step == 3
                                            if run_button_clicked:
                                                st.success("Automatically running PCA for selected datasets...")
                                        else:
                                            # In regular mode, user has to click button
                                            run_button_clicked = st.button("Run PCA", key="run_pca_btn")
                                        
                                        if run_button_clicked:
                                            if len(selected_features) < 2:
                                                st.error("Please select at least 2 features for PCA.")
                                                if consecutive_mode:
                                                    # Even in error, move to next step
                                                    st.session_state.current_analysis_step = 4
                                            else:
                                                try:
                                                    with st.spinner("Running PCA..."):
                                                        from sklearn.decomposition import PCA
                                                        from sklearn.preprocessing import StandardScaler
                                                        
                                                        # Prepare data
                                                        pca_data = analysis_data[selected_features].dropna()
                                                        
                                                        if len(pca_data) < n_components:
                                                            st.error(f"Not enough data points ({len(pca_data)}) for {n_components} components after removing missing values.")
                                                        else:
                                                            # Scale the data
                                                            scaler = StandardScaler()
                                                            scaled_data = scaler.fit_transform(pca_data)
                                                            
                                                            # Run PCA
                                                            pca = PCA(n_components=n_components, random_state=random_state)
                                                            principal_components = pca.fit_transform(scaled_data)
                                                            
                                                            # Create DataFrame with principal components
                                                            pca_df = pd.DataFrame(
                                                                data=principal_components,
                                                                columns=[f'PC{i+1}' for i in range(n_components)]
                                                            )
                                                            
                                                            # Display results
                                                            st.success(f"PCA completed successfully for {dataset_label}.")
                                                            
                                                            # Explained variance
                                                            st.write("### Explained Variance")
                                                            explained_variance = pca.explained_variance_ratio_ * 100
                                                            cumulative_variance = np.cumsum(explained_variance)
                                                            
                                                            variance_df = pd.DataFrame({
                                                                'Principal Component': [f'PC{i+1}' for i in range(n_components)],
                                                                'Explained Variance (%)': explained_variance,
                                                                'Cumulative Variance (%)': cumulative_variance
                                                            })
                                                            st.dataframe(variance_df)
                                                            
                                                            # Visualization of explained variance
                                                            fig, ax = plt.subplots(figsize=(10, 6))
                                                            ax.bar(range(1, n_components + 1), explained_variance, alpha=0.7, label='Individual')
                                                            ax.step(range(1, n_components + 1), cumulative_variance, where='mid', label='Cumulative')
                                                            ax.set_xlabel('Principal Components')
                                                            ax.set_ylabel('Explained Variance (%)')
                                                            ax.set_title('Explained Variance by Principal Components')
                                                            ax.set_xticks(range(1, n_components + 1))
                                                            ax.legend()
                                                            st.pyplot(fig)
                                                            
                                                            # Store PCA results
                                                            if selected_dataset != "Original Data":
                                                                dataset_id = int(selected_dataset.split()[-1])
                                                                for ds in st.session_state.convergence_datasets:
                                                                    if ds['id'] == dataset_id:
                                                                        ds['convergence_scores']['pca_explained_variance'] = explained_variance.tolist()
                                                                        ds['convergence_scores']['pca_cumulative_variance'] = cumulative_variance.tolist()
                                                                        break
                                                            
                                                            # Component loadings
                                                            st.write("### Component Loadings")
                                                            loadings = pca.components_.T
                                                            loadings_df = pd.DataFrame(
                                                                data=loadings,
                                                                columns=[f'PC{i+1}' for i in range(n_components)],
                                                                index=selected_features
                                                            )
                                                            st.dataframe(loadings_df)
                                                            
                                                            # Visualization of first two principal components
                                                            if n_components >= 2:
                                                                st.write("### PCA Scatter Plot (First Two Components)")
                                                                fig, ax = plt.subplots(figsize=(10, 8))
                                                                scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.7)
                                                                ax.set_xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
                                                                ax.set_ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
                                                                ax.set_title('PCA: First Two Principal Components')
                                                                
                                                                # Add a grid
                                                                ax.grid(True, linestyle='--', alpha=0.7)
                                                                
                                                                # If there are enough data points, add density contours
                                                                if len(pca_df) > 20:
                                                                    from scipy.stats import gaussian_kde
                                                                    
                                                                    # Calculate the point density
                                                                    xy = np.vstack([pca_df['PC1'], pca_df['PC2']])
                                                                    z = gaussian_kde(xy)(xy)
                                                                    
                                                                    # Sort the points by density, so that the densest points are plotted last
                                                                    idx = z.argsort()
                                                                    x, y, z = pca_df['PC1'][idx], pca_df['PC2'][idx], z[idx]
                                                                    
                                                                    plt.scatter(x, y, c=z, s=50, alpha=0.8, cmap='viridis')
                                                                    plt.colorbar(label='Density')
                                                                
                                                                st.pyplot(fig)
                                                                
                                                                # Biplot
                                                                st.write("### PCA Biplot")
                                                                fig, ax = plt.subplots(figsize=(12, 10))
                                                                
                                                                # Plot data points
                                                                ax.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.5)
                                                                
                                                                # Plot feature vectors
                                                                for i, feature in enumerate(selected_features):
                                                                    ax.arrow(0, 0, loadings[i, 0] * max(principal_components[:, 0]), 
                                                                            loadings[i, 1] * max(principal_components[:, 1]),
                                                                            head_width=0.05, head_length=0.05, fc='red', ec='red')
                                                                    plt.text(loadings[i, 0] * max(principal_components[:, 0]) * 1.15, 
                                                                            loadings[i, 1] * max(principal_components[:, 1]) * 1.15, 
                                                                            feature, color='red')
                                                                
                                                                ax.set_xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
                                                                ax.set_ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
                                                                ax.set_title('PCA Biplot')
                                                                
                                                                # Set axis limits
                                                                xlim = np.max(np.abs(principal_components[:, 0])) * 1.2
                                                                ylim = np.max(np.abs(principal_components[:, 1])) * 1.2
                                                                ax.set_xlim(-xlim, xlim)
                                                                ax.set_ylim(-ylim, ylim)
                                                                
                                                                # Add a grid
                                                                ax.grid(True, linestyle='--', alpha=0.7)
                                                                
                                                                # Add a unit circle
                                                                circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='grey', alpha=0.5)
                                                                ax.add_patch(circle)
                                                                
                                                                st.pyplot(fig)
                                                                
                                                                # If in consecutive mode, progress to next step
                                                                if consecutive_mode:
                                                                    st.session_state.current_analysis_step = 4
                                                                    st.success("PCA Factor Analysis complete. Moving to Convergence Evaluation...")
                                                                    # Force a rerun to update the UI for the next step
                                                                    st.rerun()
                                                
                                                except Exception as e:
                                                    st.error(f"Error during PCA: {e}")
                                                    # Even on error, if in consecutive mode, move to next step
                                                    if consecutive_mode:
                                                        st.session_state.current_analysis_step = 4
                                    else:
                                        st.error(f"Dataset {selected_dataset} is not available.")
                                
                                # 5. Convergence Evaluation Tab
                                with analysis_tabs[4]:
                                    st.write("### Convergence Evaluation")
                                    st.write("""
                                    This tab evaluates convergence across all analyzed datasets based on the three analysis methods.
                                    When results converge, it indicates that the interpolation has stabilized.
                                    """)
                                    
                                    # Check if we have enough datasets analyzed
                                    analyzed_datasets = [ds for ds in st.session_state.convergence_datasets 
                                                        if 'convergence_scores' in ds and ds['convergence_scores']]
                                    
                                    if not analyzed_datasets:
                                        st.warning("No analyzed datasets found. Please perform analysis on at least one dataset.")
                                    elif len(analyzed_datasets) < 2:
                                        st.info("Only one dataset has been analyzed. Please analyze at least one more dataset to evaluate convergence.")
                                    else:
                                        # Show convergence status
                                        st.write("#### Current Convergence Status")
                                        st.write(f"Status: {st.session_state.convergence_status}")
                                        st.write(f"Iterations: {st.session_state.convergence_iterations}")
                                        
                                        # Check if we're in consecutive analysis mode
                                        consecutive_mode = False
                                        if ('consecutive_analysis' in st.session_state and 
                                            st.session_state.consecutive_analysis and 
                                            st.session_state.current_analysis_step == 4):
                                            consecutive_mode = True
                                            st.info("Running Convergence Evaluation as part of consecutive analysis...")
                                        
                                        # Evaluate convergence
                                        run_button_clicked = False
                                        if consecutive_mode:
                                            # In consecutive mode, automatically trigger analysis
                                            run_button_clicked = st.session_state.current_analysis_step == 4
                                            if run_button_clicked:
                                                st.success("Automatically evaluating convergence for all datasets...")
                                        else:
                                            # In regular mode, user has to click button
                                            run_button_clicked = st.button("Evaluate Convergence", key="evaluate_convergence_btn")
                                            
                                        if run_button_clicked:
                                            # Define convergence thresholds
                                            kmeans_inertia_threshold = 0.1  # 10% change
                                            regression_r2_threshold = 0.05  # 5% change
                                            pca_variance_threshold = 0.05  # 5% change
                                            
                                            # Track convergence metrics for each dataset
                                            convergence_metrics = []
                                            
                                            # Evaluate each dataset
                                            for ds in analyzed_datasets:
                                                dataset_convergence = {
                                                    'id': ds['id'],
                                                    'metrics': {}
                                                }
                                                
                                                scores = ds['convergence_scores']
                                                
                                                # Check K-Means convergence
                                                if 'kmeans_inertia' in scores:
                                                    dataset_convergence['metrics']['kmeans'] = True
                                                else:
                                                    dataset_convergence['metrics']['kmeans'] = False
                                                
                                                # Check Regression convergence
                                                if 'regression_test_r2' in scores:
                                                    dataset_convergence['metrics']['regression'] = True
                                                else:
                                                    dataset_convergence['metrics']['regression'] = False
                                                
                                                # Check PCA convergence
                                                if 'pca_explained_variance' in scores:
                                                    dataset_convergence['metrics']['pca'] = True
                                                else:
                                                    dataset_convergence['metrics']['pca'] = False
                                                
                                                # Overall dataset convergence
                                                methods_present = sum(dataset_convergence['metrics'].values())
                                                methods_total = len(dataset_convergence['metrics'])
                                                dataset_convergence['convergence_score'] = methods_present / methods_total if methods_total > 0 else 0
                                                
                                                convergence_metrics.append(dataset_convergence)
                                            
                                            # Compare datasets for stability (convergence)
                                            if len(convergence_metrics) >= 2:
                                                # Sort by ID (iteration order)
                                                convergence_metrics.sort(key=lambda x: x['id'])
                                                
                                                # Check pairwise convergence between consecutive datasets
                                                pairwise_convergence = []
                                                for i in range(len(convergence_metrics) - 1):
                                                    ds1 = next((ds for ds in analyzed_datasets if ds['id'] == convergence_metrics[i]['id']), None)
                                                    ds2 = next((ds for ds in analyzed_datasets if ds['id'] == convergence_metrics[i+1]['id']), None)
                                                    
                                                    if ds1 and ds2:
                                                        pair_conv = {
                                                            'pair': f"{ds1['id']} and {ds2['id']}",
                                                            'metrics': {}
                                                        }
                                                        
                                                        # K-Means convergence
                                                        if ('kmeans_inertia' in ds1['convergence_scores'] and 
                                                            'kmeans_inertia' in ds2['convergence_scores']):
                                                            inertia1 = ds1['convergence_scores']['kmeans_inertia']
                                                            inertia2 = ds2['convergence_scores']['kmeans_inertia']
                                                            relative_change = abs(inertia1 - inertia2) / abs(inertia1) if inertia1 != 0 else float('inf')
                                                            pair_conv['metrics']['kmeans'] = relative_change <= kmeans_inertia_threshold
                                                        else:
                                                            pair_conv['metrics']['kmeans'] = False
                                                        
                                                        # Regression convergence
                                                        if ('regression_test_r2' in ds1['convergence_scores'] and 
                                                            'regression_test_r2' in ds2['convergence_scores']):
                                                            r2_1 = ds1['convergence_scores']['regression_test_r2']
                                                            r2_2 = ds2['convergence_scores']['regression_test_r2']
                                                            relative_change = abs(r2_1 - r2_2) / abs(r2_1) if r2_1 != 0 else float('inf')
                                                            pair_conv['metrics']['regression'] = relative_change <= regression_r2_threshold
                                                        else:
                                                            pair_conv['metrics']['regression'] = False
                                                        
                                                        # PCA convergence
                                                        if ('pca_explained_variance' in ds1['convergence_scores'] and 
                                                            'pca_explained_variance' in ds2['convergence_scores']):
                                                            var1 = ds1['convergence_scores']['pca_explained_variance']
                                                            var2 = ds2['convergence_scores']['pca_explained_variance']
                                                            if len(var1) == len(var2):
                                                                # Calculate average change in explained variance
                                                                changes = [abs(v1 - v2) / abs(v1) if v1 != 0 else float('inf') 
                                                                        for v1, v2 in zip(var1, var2)]
                                                                avg_change = sum(changes) / len(changes) if changes else float('inf')
                                                                pair_conv['metrics']['pca'] = avg_change <= pca_variance_threshold
                                                            else:
                                                                pair_conv['metrics']['pca'] = False
                                                        else:
                                                            pair_conv['metrics']['pca'] = False
                                                        
                                                        # Overall pair convergence
                                                        methods_converged = sum(pair_conv['metrics'].values())
                                                        methods_total = len(pair_conv['metrics'])
                                                        pair_conv['convergence_score'] = methods_converged / methods_total if methods_total > 0 else 0
                                                        
                                                        pairwise_convergence.append(pair_conv)
                                                
                                                # Display pairwise convergence results
                                                st.write("#### Pairwise Convergence Analysis")
                                                
                                                for pair in pairwise_convergence:
                                                    st.write(f"**Datasets {pair['pair']}**")
                                                    
                                                    metrics_df = pd.DataFrame({
                                                        'Analysis Method': ['K-Means Clustering', 'Linear Regression', 'PCA'],
                                                        'Converged': [
                                                            '✅ Yes' if pair['metrics'].get('kmeans', False) else '❌ No',
                                                            '✅ Yes' if pair['metrics'].get('regression', False) else '❌ No',
                                                            '✅ Yes' if pair['metrics'].get('pca', False) else '❌ No'
                                                        ]
                                                    })
                                                    st.dataframe(metrics_df)
                                                    
                                                    # Show convergence percentage
                                                    st.metric("Convergence Score", f"{pair['convergence_score']*100:.1f}%")
                                                    st.write("---")
                                                
                                                # Overall convergence decision
                                                # If any pair has converged on all methods, we consider the process converged
                                                converged_pairs = [p for p in pairwise_convergence if p['convergence_score'] >= 0.8]
                                                
                                                if converged_pairs:
                                                    st.success("✅ **CONVERGENCE ACHIEVED!** The interpolation process has stabilized.")
                                                    st.session_state.convergence_status = "Converged"
                                                    
                                                    # Identify the best converged pair and dataset
                                                    best_pair = max(converged_pairs, key=lambda x: x['convergence_score'])
                                                    best_dataset_id = int(best_pair['pair'].split(' and ')[1])
                                                    best_dataset = next((ds for ds in st.session_state.convergence_datasets if ds['id'] == best_dataset_id), None)
                                                    
                                                    if best_dataset:
                                                        st.write(f"**Best converged dataset: Dataset {best_dataset_id}**")
                                                        st.session_state.closest_convergence_dataset = best_dataset
                                                        
                                                    # If in consecutive mode, mark the analysis as complete
                                                    if consecutive_mode:
                                                        st.session_state.current_analysis_step = 5  # Complete
                                                        st.success("🎉 Consecutive analysis complete! All steps finished successfully with convergence achieved.")
                                                else:
                                                    # Find closest to convergence
                                                    if pairwise_convergence:
                                                        closest_pair = max(pairwise_convergence, key=lambda x: x['convergence_score'])
                                                        closest_score = closest_pair['convergence_score']
                                                        
                                                        st.warning(f"⚠️ Not yet converged. Closest pair has convergence score of {closest_score*100:.1f}%")
                                                        st.info("Continue iteration by taking the last dataset and re-interpolating it.")
                                                        
                                                        # Find the latest dataset
                                                        latest_dataset_id = max(analyzed_datasets, key=lambda x: x['id'])['id']
                                                        latest_dataset = next((ds for ds in st.session_state.convergence_datasets if ds['id'] == latest_dataset_id), None)
                                                        
                                                        # If in consecutive mode, let the user know we need another iteration
                                                        if consecutive_mode:
                                                            st.session_state.current_analysis_step = 5  # Mark as complete even though not converged
                                                            st.info("Consecutive analysis completed all steps, but convergence was not achieved. Additional iterations may be required.")
                                                        
                                                        if latest_dataset:
                                                            st.write(f"**Dataset to re-interpolate: Dataset {latest_dataset_id}**")
                                                            st.session_state.closest_convergence_dataset = latest_dataset
                                                            st.session_state.convergence_status = "Iterating"
                                                        
                                                        # Option to set last dataset for re-interpolation
                                                        if st.button("Set Latest Dataset for Re-interpolation"):
                                                            if latest_dataset:
                                                                st.session_state.interpolated_data = latest_dataset['data'].copy()
                                                                st.success(f"Dataset {latest_dataset_id} set as data to be re-interpolated.")
                                                                st.info("Please go back to Step 1: MCMC Interpolation to continue the iteration.")
                                                    else:
                                                        st.error("Cannot evaluate convergence. Not enough analyzed pairs.")
                                                        # Even if insufficient data, mark consecutive mode as complete
                                                        if consecutive_mode:
                                                            st.session_state.current_analysis_step = 5  # Complete
                                                            st.warning("Consecutive analysis could not be fully completed due to insufficient analyzed pairs.")
                                            else:
                                                st.warning("Not enough analyzed datasets to evaluate convergence.")
                                                # Even if insufficient data, mark consecutive mode as complete
                                                if consecutive_mode:
                                                    st.session_state.current_analysis_step = 5  # Complete
                                                    st.warning("Consecutive analysis could not be fully completed due to insufficient datasets.")
                                                
                                        # Display convergence history
                                        if 'convergence_datasets' in st.session_state and st.session_state.convergence_datasets:
                                            st.write("#### Convergence Metrics History")
                                            
                                            # Create metrics tracking for visualization
                                            datasets = [ds for ds in st.session_state.convergence_datasets if 'convergence_scores' in ds]
                                            
                                            if datasets:
                                                # Track metrics across iterations
                                                iteration_data = {
                                                    'Dataset ID': [],
                                                    'K-Means Inertia': [],
                                                    'Regression R²': [],
                                                    'PCA Explained Variance (PC1)': []
                                                }
                                                
                                                for ds in sorted(datasets, key=lambda x: x['id']):
                                                    scores = ds['convergence_scores']
                                                    iteration_data['Dataset ID'].append(ds['id'])
                                                    
                                                    # K-Means
                                                    if 'kmeans_inertia' in scores:
                                                        iteration_data['K-Means Inertia'].append(scores['kmeans_inertia'])
                                                    else:
                                                        iteration_data['K-Means Inertia'].append(None)
                                                    
                                                    # Regression
                                                    if 'regression_test_r2' in scores:
                                                        iteration_data['Regression R²'].append(scores['regression_test_r2'])
                                                    else:
                                                        iteration_data['Regression R²'].append(None)
                                                    
                                                    # PCA
                                                    if 'pca_explained_variance' in scores and len(scores['pca_explained_variance']) > 0:
                                                        iteration_data['PCA Explained Variance (PC1)'].append(scores['pca_explained_variance'][0])
                                                    else:
                                                        iteration_data['PCA Explained Variance (PC1)'].append(None)
                                                
                                                # Plot convergence metrics
                                                if len(iteration_data['Dataset ID']) > 1:
                                                    # Create subplots
                                                    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
                                                    
                                                    # K-Means Inertia
                                                    valid_kmeans = [i for i, v in enumerate(iteration_data['K-Means Inertia']) if v is not None]
                                                    if valid_kmeans:
                                                        axes[0].plot([iteration_data['Dataset ID'][i] for i in valid_kmeans], 
                                                                    [iteration_data['K-Means Inertia'][i] for i in valid_kmeans], 
                                                                    'o-', color='blue')
                                                        axes[0].set_ylabel('Inertia')
                                                        axes[0].set_title('K-Means Inertia (lower is better)')
                                                        axes[0].grid(True, linestyle='--', alpha=0.7)
                                                    else:
                                                        axes[0].text(0.5, 0.5, 'No K-Means data available', 
                                                                    horizontalalignment='center', verticalalignment='center',
                                                                    transform=axes[0].transAxes)
                                                    
                                                    # Regression R²
                                                    valid_reg = [i for i, v in enumerate(iteration_data['Regression R²']) if v is not None]
                                                    if valid_reg:
                                                        axes[1].plot([iteration_data['Dataset ID'][i] for i in valid_reg], 
                                                                    [iteration_data['Regression R²'][i] for i in valid_reg], 
                                                                    'o-', color='green')
                                                        axes[1].set_ylabel('R² Score')
                                                        axes[1].set_title('Regression R² Score (higher is better)')
                                                        axes[1].grid(True, linestyle='--', alpha=0.7)
                                                    else:
                                                        axes[1].text(0.5, 0.5, 'No Regression data available', 
                                                                    horizontalalignment='center', verticalalignment='center',
                                                                    transform=axes[1].transAxes)
                                                    
                                                    # PCA Explained Variance
                                                    valid_pca = [i for i, v in enumerate(iteration_data['PCA Explained Variance (PC1)']) if v is not None]
                                                    if valid_pca:
                                                        axes[2].plot([iteration_data['Dataset ID'][i] for i in valid_pca], 
                                                                    [iteration_data['PCA Explained Variance (PC1)'][i] for i in valid_pca], 
                                                                    'o-', color='purple')
                                                        axes[2].set_ylabel('Explained Variance (%)')
                                                        axes[2].set_title('PCA First Component Explained Variance (stability is better)')
                                                        axes[2].grid(True, linestyle='--', alpha=0.7)
                                                    else:
                                                        axes[2].text(0.5, 0.5, 'No PCA data available', 
                                                                    horizontalalignment='center', verticalalignment='center',
                                                                    transform=axes[2].transAxes)
                                                    
                                                    # X-axis label
                                                    axes[2].set_xlabel('Dataset ID (Iteration)')
                                                    
                                                    plt.tight_layout()
                                                    st.pyplot(fig)
                                                
                                                # Create table of metrics
                                                st.write("#### Metrics Table")
                                                metrics_df = pd.DataFrame(iteration_data)
                                                st.dataframe(metrics_df)
                    
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