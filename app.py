import streamlit as st
import pandas as pd
import numpy as np
import os
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
if 'data' not in st.session_state:
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

# Initialize modules
data_handler = DataHandler()
data_processor = DataProcessor()
predictor = Predictor()
risk_assessor = RiskAssessor()
visualizer = Visualizer()

# Application title and description
st.title("Modular Data Analysis Platform")
st.markdown("""
    This platform provides tools for data processing, prediction, and risk assessment.
    Use the navigation sidebar to access different modules.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose a module",
    ["Data Import/Export", "Data Processing", "Prediction", "Risk Assessment", "Visualization"]
)

# Data Import/Export module
if app_mode == "Data Import/Export":
    st.header("Data Import/Export")
    
    # Data Import section
    st.subheader("Import Data")
    
    upload_method = st.radio(
        "Select import method:",
        ["Upload File", "Sample Dataset"]
    )
    
    if upload_method == "Upload File":
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                st.session_state.data = data_handler.import_data(uploaded_file)
                st.success(f"Successfully imported data with {st.session_state.data.shape[0]} rows and {st.session_state.data.shape[1]} columns.")
                
                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(st.session_state.data.head())
                
                # Display basic statistics
                st.subheader("Data Statistics")
                st.write(st.session_state.data.describe())
                
            except Exception as e:
                st.error(f"Error importing data: {e}")
    
    elif upload_method == "Sample Dataset":
        st.info("Please upload your own data. No sample datasets are available.")
    
    # Data Export section
    if st.session_state.data is not None:
        st.subheader("Export Data")
        
        data_to_export = st.radio(
            "Select data to export:",
            ["Original Data", "Processed Data", "Predictions", "Risk Assessment"],
            disabled=not st.session_state.data is not None
        )
        
        export_format = st.selectbox(
            "Select export format:",
            ["CSV", "Excel"],
            disabled=not st.session_state.data is not None
        )
        
        if st.button("Export", disabled=not st.session_state.data is not None):
            try:
                if data_to_export == "Original Data" and st.session_state.data is not None:
                    export_data = st.session_state.data
                elif data_to_export == "Processed Data" and st.session_state.processed_data is not None:
                    export_data = st.session_state.processed_data
                elif data_to_export == "Predictions" and st.session_state.predictions is not None:
                    export_data = st.session_state.predictions
                elif data_to_export == "Risk Assessment" and st.session_state.risk_assessment is not None:
                    export_data = st.session_state.risk_assessment
                else:
                    st.warning(f"No {data_to_export.lower()} available to export.")
                    export_data = None
                
                if export_data is not None:
                    export_success, download_link = data_handler.export_data(export_data, export_format.lower())
                    if export_success:
                        st.markdown(download_link, unsafe_allow_html=True)
                    else:
                        st.error("Failed to export data.")
            except Exception as e:
                st.error(f"Error exporting data: {e}")

# Data Processing module
elif app_mode == "Data Processing":
    st.header("Data Processing")
    
    if st.session_state.data is None:
        st.warning("Please import data first in the Data Import/Export module.")
    else:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data.head())
        
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
        
        # Data cleaning options
        st.subheader("Data Cleaning")
        handle_missing = st.checkbox("Handle missing values")
        missing_method = None
        if handle_missing:
            missing_method = st.radio(
                "Method for handling missing values:",
                ["Remove rows", "Mean imputation", "Median imputation", "Mode imputation"]
            )
        
        remove_duplicates = st.checkbox("Remove duplicate rows")
        
        # Data transformation options
        st.subheader("Data Transformation")
        normalize_data = st.checkbox("Normalize numerical features")
        norm_method = None
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

# Prediction module
elif app_mode == "Prediction":
    st.header("Prediction")
    
    if st.session_state.processed_data is None:
        st.warning("Please process data first in the Data Processing module.")
    else:
        st.subheader("Processed Data Preview")
        st.dataframe(st.session_state.processed_data.head())
        
        st.subheader("Prediction Configuration")
        
        # Verify target column is selected
        if st.session_state.target_column is None or st.session_state.target_column == "None":
            st.warning("Please select a target column in the Data Processing module.")
        else:
            # Model selection
            model_type = st.selectbox(
                "Select prediction model:",
                ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
            )
            
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
            
            # Train-test split
            test_size = st.slider("Test set size (%)", 10, 50, 20) / 100
            
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

# Risk Assessment module
elif app_mode == "Risk Assessment":
    st.header("Risk Assessment")
    
    if st.session_state.predictions is None:
        st.warning("Please generate predictions first in the Prediction module.")
    else:
        st.subheader("Predictions Preview")
        st.dataframe(st.session_state.predictions.head())
        
        st.subheader("Risk Assessment Configuration")
        
        # Risk assessment method
        assessment_method = st.selectbox(
            "Select risk assessment method:",
            ["Prediction Intervals", "Error Distribution", "Outlier Detection"]
        )
        
        # Confidence level for prediction intervals
        if assessment_method == "Prediction Intervals":
            confidence_level = st.slider("Confidence level (%)", 70, 99, 95)
        
        # Threshold for outlier detection
        if assessment_method == "Outlier Detection":
            threshold = st.slider("Outlier threshold (standard deviations)", 1.0, 5.0, 3.0)
        
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
                st.write(risk_summary)
                
            except Exception as e:
                st.error(f"Error during risk assessment: {e}")

# Visualization module
elif app_mode == "Visualization":
    st.header("Data Visualization")
    
    if st.session_state.data is None:
        st.warning("Please import data first in the Data Import/Export module.")
    else:
        # Select data to visualize
        data_options = ["Original Data"]
        if st.session_state.processed_data is not None:
            data_options.append("Processed Data")
        if st.session_state.predictions is not None:
            data_options.append("Predictions")
        if st.session_state.risk_assessment is not None:
            data_options.append("Risk Assessment")
        
        data_to_visualize = st.selectbox("Select data to visualize:", data_options)
        
        # Get the selected dataset
        if data_to_visualize == "Original Data":
            viz_data = st.session_state.data
        elif data_to_visualize == "Processed Data":
            viz_data = st.session_state.processed_data
        elif data_to_visualize == "Predictions":
            viz_data = st.session_state.predictions
        elif data_to_visualize == "Risk Assessment":
            viz_data = st.session_state.risk_assessment
        
        # Preview the data
        st.subheader("Data Preview")
        st.dataframe(viz_data.head())
        
        # Visualization options
        st.subheader("Visualization Options")
        
        viz_type = st.selectbox(
            "Select visualization type:",
            ["Histogram", "Scatter Plot", "Line Chart", "Bar Chart", "Correlation Matrix", "Box Plot"]
        )
        
        if viz_type in ["Histogram", "Box Plot"]:
            column = st.selectbox("Select column:", viz_data.select_dtypes(include=np.number).columns)
            
            if st.button("Generate Visualization"):
                st.subheader(f"{viz_type} for {column}")
                
                if viz_type == "Histogram":
                    fig = visualizer.plot_histogram(viz_data, column)
                    st.pyplot(fig)
                elif viz_type == "Box Plot":
                    fig = visualizer.plot_box(viz_data, column)
                    st.pyplot(fig)
        
        elif viz_type in ["Scatter Plot"]:
            x_column = st.selectbox("Select X column:", viz_data.select_dtypes(include=np.number).columns)
            y_column = st.selectbox("Select Y column:", 
                                   [col for col in viz_data.select_dtypes(include=np.number).columns if col != x_column],
                                   index=0 if len(viz_data.select_dtypes(include=np.number).columns) > 1 else None)
            
            if st.button("Generate Visualization"):
                st.subheader(f"{viz_type}: {x_column} vs {y_column}")
                fig = visualizer.plot_scatter(viz_data, x_column, y_column)
                st.pyplot(fig)
        
        elif viz_type in ["Line Chart"]:
            if viz_data.select_dtypes(include=np.number).shape[1] == 0:
                st.warning("No numerical columns available for visualization.")
            else:
                columns = st.multiselect(
                    "Select columns:",
                    viz_data.select_dtypes(include=np.number).columns
                )
                
                if st.button("Generate Visualization") and columns:
                    st.subheader(f"{viz_type}")
                    fig = visualizer.plot_line(viz_data, columns)
                    st.pyplot(fig)
        
        elif viz_type in ["Bar Chart"]:
            if viz_data.select_dtypes(include=np.number).shape[1] == 0:
                st.warning("No numerical columns available for visualization.")
            else:
                y_column = st.selectbox("Select value column:", viz_data.select_dtypes(include=np.number).columns)
                x_column = st.selectbox("Select category column:", viz_data.columns)
                
                if st.button("Generate Visualization"):
                    st.subheader(f"{viz_type}: {y_column} by {x_column}")
                    fig = visualizer.plot_bar(viz_data, x_column, y_column)
                    st.pyplot(fig)
        
        elif viz_type == "Correlation Matrix":
            if viz_data.select_dtypes(include=np.number).shape[1] < 2:
                st.warning("Need at least two numerical columns for correlation matrix.")
            else:
                if st.button("Generate Visualization"):
                    st.subheader("Correlation Matrix")
                    fig = visualizer.plot_correlation(viz_data)
                    st.pyplot(fig)

# Show info about current state
st.sidebar.header("Current State")
if st.session_state.data is not None:
    st.sidebar.success("✅ Data imported")
else:
    st.sidebar.warning("❌ No data imported")

if st.session_state.processed_data is not None:
    st.sidebar.success("✅ Data processed")
else:
    st.sidebar.warning("❌ No processed data")

if st.session_state.predictions is not None:
    st.sidebar.success("✅ Predictions generated")
else:
    st.sidebar.warning("❌ No predictions")

if st.session_state.risk_assessment is not None:
    st.sidebar.success("✅ Risk assessment completed")
else:
    st.sidebar.warning("❌ No risk assessment")
