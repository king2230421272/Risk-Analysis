import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from modules.data_processing import DataProcessor
from modules.prediction import Predictor
from modules.risk_assessment import RiskAssessor
from utils.data_handler import DataHandler
from utils.visualization import Visualizer

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

# Initialize modules
data_handler = DataHandler()
data_processor = DataProcessor()
predictor = Predictor()
risk_assessor = RiskAssessor()
visualizer = Visualizer()

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1️⃣ Data Import", 
        "2️⃣ Data Processing", 
        "3️⃣ Prediction", 
        "4️⃣ Risk Assessment", 
        "5️⃣ Visualization"
    ])
    
    # 1. DATA IMPORT TAB
    with tab1:
        st.header("Data Import")
        
        # Data import section with two columns for original and interpolated data
        st.subheader("Import Multiple Datasets")
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
                    selected_col = st.selectbox("Select column to compare:", common_cols)
                    
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
        
        if st.session_state.data is None:
            st.warning("No data available. Please import data in the Data Import tab.")
        else:
            # Data preview
            st.subheader("Data Preview")
            st.dataframe(st.session_state.data.head())
            
            # Data processing options
            st.subheader("Data Processing Options")
            
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
            st.session_state.target_column = st.selectbox(
                "Target Variable",
                ["None"] + all_columns,
                index=0 if st.session_state.target_column is None else all_columns.index(st.session_state.target_column) + 1
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
            if st.button("Process Data"):
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
                        ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
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
                    ["Prediction Intervals", "Error Distribution", "Outlier Detection"]
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