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
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'visualization_type' not in st.session_state:
    st.session_state.visualization_type = None

# Initialize modules
data_handler = DataHandler()
data_processor = DataProcessor()
predictor = Predictor()
risk_assessor = RiskAssessor()
visualizer = Visualizer()

# Application title and description
st.title("Data Analysis Pipeline")
st.markdown("""
    This platform provides a step-by-step workflow for data analysis, from data import to risk assessment.
    Follow the steps in sequence to complete your analysis.
""")

# Progress bar
total_steps = 5
progress = (st.session_state.current_step - 1) / total_steps
st.progress(progress)

# Step indicator
st.markdown(f"### Step {st.session_state.current_step} of {total_steps}")

# Define step names for reference
step_names = {
    1: "Data Import",
    2: "Data Processing",
    3: "Prediction",
    4: "Risk Assessment",
    5: "Visualization"
}

# Display current step name
st.subheader(f"Current Step: {step_names[st.session_state.current_step]}")

# Functions to navigate between steps
def go_to_next_step():
    if st.session_state.current_step < total_steps:
        st.session_state.current_step += 1
    st.rerun()

def go_to_previous_step():
    if st.session_state.current_step > 1:
        st.session_state.current_step -= 1
    st.rerun()

# Current step
current_step = st.session_state.current_step

# Content for each step
if current_step == 1:  # Data Import
    st.header("Step 1: Data Import")
    
    # Data Import section
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
                
                # Enable proceeding to next step once data is loaded
                st.button("Continue to Data Processing", on_click=go_to_next_step)
                
            except Exception as e:
                st.error(f"Error importing data: {e}")
    
    elif upload_method == "Sample Dataset":
        st.info("Please upload your own data. No sample datasets are available.")

elif current_step == 2:  # Data Processing
    st.header("Step 2: Data Processing")
    
    if st.session_state.data is None:
        st.warning("No data available. Please go back to import data first.")
        st.button("Go back to Data Import", on_click=go_to_previous_step)
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
        col1, col2 = st.columns(2)
        with col1:
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
                    
        # Navigation buttons
        st.subheader("Navigation")
        col1, col2 = st.columns(2)
        with col1:
            st.button("Back to Data Import", on_click=go_to_previous_step)
        with col2:
            if st.session_state.processed_data is not None:
                st.button("Continue to Prediction", on_click=go_to_next_step)
            else:
                st.warning("Please process the data before continuing to the next step.")

elif current_step == 3:  # Prediction
    st.header("Step 3: Prediction")
    
    if st.session_state.processed_data is None:
        st.warning("No processed data available. Please go back to process data first.")
        st.button("Go back to Data Processing", on_click=go_to_previous_step)
    else:
        # Data preview
        st.subheader("Processed Data Preview")
        st.dataframe(st.session_state.processed_data.head())
        
        # Check if target column is selected
        if st.session_state.target_column is None or st.session_state.target_column == "None":
            st.warning("No target column selected. Please go back and select a target column.")
            st.button("Go back to Data Processing", on_click=go_to_previous_step)
        else:
            st.subheader("Prediction Configuration")
            
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
            col1, col2 = st.columns(2)
            with col1:
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
            
            # Navigation buttons
            st.subheader("Navigation")
            col1, col2 = st.columns(2)
            with col1:
                st.button("Back to Data Processing", on_click=go_to_previous_step)
            with col2:
                if st.session_state.predictions is not None:
                    st.button("Continue to Risk Assessment", on_click=go_to_next_step)
                else:
                    st.warning("Please generate predictions before continuing to the next step.")

elif current_step == 4:  # Risk Assessment
    st.header("Step 4: Risk Assessment")
    
    if st.session_state.predictions is None:
        st.warning("No predictions available. Please go back to generate predictions first.")
        st.button("Go back to Prediction", on_click=go_to_previous_step)
    else:
        # Predictions preview
        st.subheader("Predictions Preview")
        st.dataframe(st.session_state.predictions.head())
        
        st.subheader("Risk Assessment Configuration")
        
        # Risk assessment method
        assessment_method = st.selectbox(
            "Select risk assessment method:",
            ["Prediction Intervals", "Error Distribution", "Outlier Detection"]
        )
        
        # Method-specific parameters
        if assessment_method == "Prediction Intervals":
            confidence_level = st.slider("Confidence level (%)", 70, 99, 95)
        elif assessment_method == "Outlier Detection":
            threshold = st.slider("Outlier threshold (standard deviations)", 1.0, 5.0, 3.0)
        
        # Perform risk assessment
        col1, col2 = st.columns(2)
        with col1:
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
        
        # Navigation buttons
        st.subheader("Navigation")
        col1, col2 = st.columns(2)
        with col1:
            st.button("Back to Prediction", on_click=go_to_previous_step)
        with col2:
            if st.session_state.risk_assessment is not None:
                st.button("Continue to Visualization", on_click=go_to_next_step)
            else:
                st.warning("Please complete risk assessment before continuing to the next step.")

elif current_step == 5:  # Visualization
    st.header("Step 5: Visualization")
    
    if st.session_state.data is None:
        st.warning("No data available. Please go back to import data first.")
        st.button("Go back to Data Import", on_click=lambda: setattr(st.session_state, 'current_step', 1))
    else:
        # Data selection
        st.subheader("Select Data to Visualize")
        data_options = ["Original Data"]
        
        if st.session_state.processed_data is not None:
            data_options.append("Processed Data")
        if st.session_state.predictions is not None:
            data_options.append("Predictions")
        if st.session_state.risk_assessment is not None:
            data_options.append("Risk Assessment")
        
        data_to_visualize = st.radio("Select dataset:", data_options)
        
        # Get the selected data
        if data_to_visualize == "Original Data":
            viz_data = st.session_state.data
        elif data_to_visualize == "Processed Data":
            viz_data = st.session_state.processed_data
        elif data_to_visualize == "Predictions":
            viz_data = st.session_state.predictions
        elif data_to_visualize == "Risk Assessment":
            viz_data = st.session_state.risk_assessment
        
        # Visualization type
        st.subheader("Select Visualization Type")
        viz_type = st.selectbox(
            "Visualization type:",
            ["Data Overview", "Histogram", "Scatter Plot", "Line Chart", 
             "Bar Chart", "Correlation Matrix", "Box Plot"]
        )
        
        # Store visualization type in session state
        st.session_state.visualization_type = viz_type
        
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
            # Select column for histogram
            column = st.selectbox("Select column for histogram:", viz_data.columns)
            
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(viz_data[column]):
                fig = visualizer.plot_histogram(viz_data, column)
                st.pyplot(fig)
            else:
                st.warning("Histograms can only be plotted for numeric columns.")
            
        elif viz_type == "Scatter Plot":
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
            # Select columns for line chart
            columns = st.multiselect("Select columns for line chart:", viz_data.select_dtypes(include=np.number).columns)
            
            if columns:
                fig = visualizer.plot_line(viz_data, columns)
                st.pyplot(fig)
            else:
                st.warning("Please select at least one column for the line chart.")
            
        elif viz_type == "Bar Chart":
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
        
        # Navigation buttons
        st.subheader("Navigation")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button("Back to Risk Assessment", on_click=go_to_previous_step)
        with col2:
            if st.button("Start Over"):
                # Reset all session state variables except the data
                for key in list(st.session_state.keys()):
                    if key not in ['data']:
                        del st.session_state[key]
                # Initialize current step
                st.session_state.current_step = 1
                st.rerun()
        with col3:
            st.button("Export Final Results", on_click=lambda: export_final_results())
            
# Function to export final results            
def export_final_results():
    # Check if we have processed data
    if st.session_state.processed_data is not None:
        try:
            export_format = "csv"
            export_success, download_link = data_handler.export_data(st.session_state.processed_data, export_format)
            if export_success:
                st.success("Final processed data exported successfully!")
                st.markdown(download_link, unsafe_allow_html=True)
            else:
                st.error("Failed to export data.")
        except Exception as e:
            st.error(f"Error exporting data: {e}")
    else:
        st.warning("No processed data available to export.")