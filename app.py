import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        st.write("This is the data import tab.")
    
    # 2. DATA PROCESSING TAB  
    with tab2:
        st.header("Data Processing")
        st.write("This is the data processing tab.")

    # 3. PREDICTION TAB
    with tab3:
        st.header("Prediction")
        
        # Create subtabs for prediction sections that are always visible
        pred_tab1, pred_tab2, pred_tab3, pred_tab4 = st.tabs([
            "Train Model", 
            "Results", 
            "Quality Analysis",
            "Verification"
        ])
        
        # Train Model tab
        with pred_tab1:
            st.subheader("Train Prediction Model")
            
            # Model selection interface - always visible
            st.markdown("### Select Model Type")
            model_type = st.selectbox(
                "Choose a model type",
                [
                    "Linear Regression", 
                    "Decision Tree", 
                    "Random Forest", 
                    "Gradient Boosting", 
                    "Neural Network",
                    "LSTM Network"
                ],
                key="prediction_model_type"
            )
            
            # Dataset selection - always visible
            st.markdown("### Select Datasets")
            
            # Create two columns for dataset selection
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Original Dataset**")
                st.info("Please load an original dataset in the Data Import tab")
                
            with col2:
                st.markdown("**Interpolated Dataset**")
                st.info("Please load an interpolated dataset in the Data Import tab")
            
            # Feature and target selection - always visible
            st.markdown("### Select Features and Target")
            
            # Display a message if no dataset is loaded
            if 'original_data' not in st.session_state and 'interpolated_data' not in st.session_state:
                st.warning("Please load datasets in the Data Import tab to select features and targets")
            
            # Training parameters - always visible
            st.markdown("### Training Parameters")
            test_size = st.slider("Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            
            # Model specific parameters - shown conditionally but container is always visible
            st.markdown("### Model Specific Parameters")
            model_params_container = st.container()
            
            with model_params_container:
                if model_type == "Linear Regression":
                    st.write("No specific parameters for Linear Regression")
                    
                elif model_type == "Decision Tree":
                    max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5)
                    min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2)
                    
                elif model_type == "Random Forest":
                    n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, value=100, step=10)
                    max_depth = st.slider("Max Depth", min_value=1, max_value=20, value=5)
                    
                elif model_type == "Gradient Boosting":
                    n_estimators = st.slider("Number of Estimators", min_value=10, max_value=200, value=100, step=10)
                    learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
                    
                elif model_type == "Neural Network":
                    hidden_layers = st.slider("Hidden Layers", min_value=1, max_value=5, value=2)
                    neurons_per_layer = st.slider("Neurons per Layer", min_value=8, max_value=128, value=64, step=8)
                    epochs = st.slider("Training Epochs", min_value=10, max_value=500, value=100, step=10)
                    
                elif model_type == "LSTM Network":
                    hidden_dim = st.slider("Hidden Dimension", min_value=8, max_value=128, value=64, step=8)
                    num_layers = st.slider("Number of LSTM Layers", min_value=1, max_value=3, value=1)
                    sequence_length = st.slider("Sequence Length", min_value=1, max_value=10, value=3)
            
            # Train button - always visible but disabled if no data is loaded
            train_button = st.button("Train Model", disabled=('original_data' not in st.session_state and 'interpolated_data' not in st.session_state))
        
        # Results tab
        with pred_tab2:
            st.subheader("Model Results")
            
            # Performance metrics section - always visible
            st.markdown("### Performance Metrics")
            
            if 'trained_models' not in st.session_state:
                st.info("Train a model to see performance metrics here")
            else:
                # Placeholder for metrics visualization
                st.write("When a model is trained, metrics will be displayed here.")
            
            # Prediction visualization - always visible
            st.markdown("### Prediction Visualization")
            
            if 'trained_models' not in st.session_state:
                st.info("Train a model to visualize predictions")
            else:
                # Placeholder for prediction visualization
                st.write("When a model is trained, predictions will be visualized here.")
            
            # Model parameters - always visible
            st.markdown("### Model Parameters")
            
            if 'trained_models' not in st.session_state:
                st.info("Train a model to see its parameters")
            else:
                # Placeholder for model parameters
                st.write("When a model is trained, its parameters will be displayed here.")
        
        # Quality Analysis tab
        with pred_tab3:
            st.subheader("Model Quality Analysis")
            
            # Feature importance - always visible
            st.markdown("### Feature Importance")
            
            if 'trained_models' not in st.session_state:
                st.info("Train a model to see feature importance")
            else:
                # Placeholder for feature importance
                st.write("When a model is trained, feature importance will be displayed here.")
            
            # Residual analysis - always visible
            st.markdown("### Residual Analysis")
            
            if 'trained_models' not in st.session_state:
                st.info("Train a model to see residual analysis")
            else:
                # Placeholder for residual analysis
                st.write("When a model is trained, residual analysis will be displayed here.")
            
            # SHAP analysis - always visible
            st.markdown("### SHAP Analysis")
            
            if 'trained_models' not in st.session_state:
                st.info("Train a model to see SHAP analysis")
            else:
                # Placeholder for SHAP analysis
                st.write("When a model is trained, SHAP analysis will be displayed here.")
        
        # Verification tab
        with pred_tab4:
            st.subheader("Model Verification")
            
            # Parameter testing - always visible
            st.markdown("### Parameter Testing")
            
            if 'trained_models' not in st.session_state:
                st.info("Train a model to test parameters")
            else:
                # Placeholder for parameter testing
                st.write("When a model is trained, parameter testing tools will be available here.")
            
            # Cross-validation - always visible
            st.markdown("### Cross-Validation")
            
            if 'trained_models' not in st.session_state:
                st.info("Train a model to perform cross-validation")
            else:
                # Placeholder for cross-validation
                st.write("When a model is trained, cross-validation results will be displayed here.")
            
            # Out-of-sample testing - always visible
            st.markdown("### Out-of-Sample Testing")
            
            if 'trained_models' not in st.session_state:
                st.info("Train a model to perform out-of-sample testing")
            else:
                # Placeholder for out-of-sample testing
                st.write("When a model is trained, out-of-sample testing will be available here.")
    
    # 4. RISK ASSESSMENT TAB
    with tab4:
        st.header("Risk Assessment")
        
        # Create tabs for the three risk assessment methods
        ra_tab1, ra_tab2, ra_tab3 = st.tabs([
            "概率损失法 (Probability Loss)", 
            "IAHP-CRITIC-GT法", 
            "动态贝叶斯网络法 (Dynamic Bayes)"
        ])
        
        # 概率损失法 tab
        with ra_tab1:
            st.subheader("概率损失法 (Probability Loss Method)")
            
            # Model selection - always visible
            st.markdown("### Select Model")
            
            if 'trained_models' not in st.session_state:
                st.info("Train a model in the Prediction tab or load models from the Database tab")
                model_selection = st.selectbox(
                    "Model Selection", 
                    ["No models available"], 
                    key="prob_loss_model_select"
                )
            else:
                model_selection = st.selectbox(
                    "Model Selection", 
                    ["Select a model..."] + list(st.session_state.trained_models.keys()), 
                    key="prob_loss_model_select"
                )
            
            # Risk parameters - always visible
            st.markdown("### Risk Parameters")
            
            # Confidence level slider - always visible
            confidence_level = st.slider(
                "Confidence Level", 
                min_value=0.8, 
                max_value=0.99, 
                value=0.95, 
                step=0.01,
                key="prob_loss_confidence"
            )
            
            # Loss threshold - always visible
            loss_threshold = st.number_input(
                "Loss Threshold", 
                min_value=0.0, 
                value=1.0, 
                step=0.1,
                key="prob_loss_threshold"
            )
            
            # Run analysis button - always visible but disabled when no model is available
            run_analysis_button = st.button(
                "Run Probability Loss Analysis", 
                disabled=('trained_models' not in st.session_state),
                key="run_prob_loss"
            )
            
            # Results section - always visible
            st.markdown("### Analysis Results")
            
            if 'prob_loss_results' not in st.session_state:
                st.info("Run the analysis to see results here")
            else:
                # Placeholder for results
                st.write("When analysis is run, results will be displayed here.")
        
        # IAHP-CRITIC-GT法 tab
        with ra_tab2:
            st.subheader("IAHP-CRITIC-GT Method")
            
            # LLM model selection - always visible
            st.markdown("### LLM Model Selection")
            llm_model = st.selectbox(
                "Select LLM Model", 
                ["DeepSeek", "Anthropic Claude", "OpenAI GPT-4o"],
                key="iahp_llm_model"
            )
            
            # Risk weights section - always visible
            st.markdown("### Risk Weights Configuration")
            
            # IAHP weights - always visible
            st.markdown("#### IAHP Weights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                iahp_technical = st.slider(
                    "Technical Risk Weight", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.33, 
                    step=0.01,
                    key="iahp_technical"
                )
                
                iahp_environmental = st.slider(
                    "Environmental Risk Weight", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.33, 
                    step=0.01,
                    key="iahp_environmental"
                )
            
            with col2:
                iahp_management = st.slider(
                    "Management Risk Weight", 
                    min_value=0.0, 
                    max_value=1.0, 
                    value=0.34, 
                    step=0.01,
                    key="iahp_management"
                )
                
                # Display total to ensure it's 1.0
                total = iahp_technical + iahp_environmental + iahp_management
                st.metric("Total Weight", f"{total:.2f}", delta=f"{total-1.0:.2f}")
            
            # Run analysis button - always visible
            run_iahp_button = st.button(
                "Run IAHP-CRITIC-GT Analysis", 
                key="run_iahp"
            )
            
            # Results section - always visible
            st.markdown("### Analysis Results")
            
            if 'iahp_results' not in st.session_state:
                st.info("Run the analysis to see results here")
            else:
                # Placeholder for results
                st.write("When analysis is run, results will be displayed here.")
        
        # 动态贝叶斯网络法 tab
        with ra_tab3:
            st.subheader("动态贝叶斯网络法 (Dynamic Bayesian Network)")
            
            # Network configuration - always visible
            st.markdown("### Network Configuration")
            
            # Node selection - always visible
            st.markdown("#### Select Risk Nodes")
            
            # This would typically come from the data, but since we don't have data loaded yet, show placeholders
            if 'original_data' not in st.session_state and 'interpolated_data' not in st.session_state:
                st.info("Load data to select risk nodes")
                available_nodes = ["No data available"]
                selected_nodes = st.multiselect(
                    "Select Risk Factors", 
                    available_nodes,
                    disabled=True,
                    key="bayes_nodes"
                )
            else:
                # Placeholder for when data is available
                available_nodes = ["Factor 1", "Factor 2", "Factor 3"]  # This would be replaced with actual column names
                selected_nodes = st.multiselect(
                    "Select Risk Factors", 
                    available_nodes,
                    key="bayes_nodes"
                )
            
            # Edge configuration - always visible
            st.markdown("#### Configure Node Relationships")
            
            # This interface would let users define relationships between nodes
            if len(selected_nodes) < 2 or 'original_data' not in st.session_state:
                st.info("Select at least two nodes to configure relationships")
            else:
                st.write("Interface for configuring node relationships would appear here")
            
            # Prior probabilities - always visible
            st.markdown("### Prior Probabilities")
            
            if len(selected_nodes) == 0 or 'original_data' not in st.session_state:
                st.info("Select nodes to configure prior probabilities")
            else:
                st.write("Interface for setting prior probabilities would appear here")
            
            # Run analysis button - always visible but disabled when conditions aren't met
            run_bayes_button = st.button(
                "Run Bayesian Network Analysis", 
                disabled=(len(selected_nodes) < 2 or 'original_data' not in st.session_state),
                key="run_bayes"
            )
            
            # Results section - always visible
            st.markdown("### Analysis Results")
            
            if 'bayes_results' not in st.session_state:
                st.info("Run the analysis to see results here")
            else:
                # Placeholder for results
                st.write("When analysis is run, results will be displayed here.")
    
    # 5. VISUALIZATION TAB
    with tab5:
        st.header("Visualization")
        st.write("This is the visualization tab.")
    
    # 6. DATABASE TAB
    with tab6:
        st.header("Database Management")
        
        # Create database management tabs
        db_tab1, db_tab2, db_tab3 = st.tabs([
            "Dataset Management", 
            "Model Management", 
            "Export/Import"
        ])
        
        # Dataset Management tab
        with db_tab1:
            st.subheader("Dataset Management")
            
            # Saved datasets section
            st.markdown("### Saved Datasets")
            
            # Check for saved datasets
            # This would be replaced with actual database query
            if 'saved_datasets' not in st.session_state:
                st.session_state.saved_datasets = []
            
            if not st.session_state.saved_datasets:
                st.info("No datasets saved in the database")
            else:
                # Display saved datasets with multiselect for deletion
                selected_datasets = st.multiselect(
                    "Select datasets",
                    st.session_state.saved_datasets,
                    key="db_dataset_select"
                )
                
                # Dataset operations
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    load_button = st.button(
                        "Load Selected Datasets", 
                        disabled=not selected_datasets,
                        key="db_load_datasets"
                    )
                
                with col2:
                    delete_button = st.button(
                        "Delete Selected Datasets", 
                        disabled=not selected_datasets,
                        key="db_delete_datasets"
                    )
                
                with col3:
                    view_details_button = st.button(
                        "View Dataset Details", 
                        disabled=len(selected_datasets) != 1,
                        key="db_view_dataset_details"
                    )
            
            # Save current datasets section
            st.markdown("### Save Current Datasets")
            
            # Check if datasets are in session state
            has_datasets = ('original_data' in st.session_state or 'interpolated_data' in st.session_state)
            
            if not has_datasets:
                st.info("No datasets available to save")
            else:
                # Dataset selection
                col1, col2 = st.columns(2)
                
                with col1:
                    save_original = st.checkbox(
                        "Save Original Dataset", 
                        value=('original_data' in st.session_state),
                        disabled=('original_data' not in st.session_state),
                        key="db_save_original"
                    )
                
                with col2:
                    save_interpolated = st.checkbox(
                        "Save Interpolated Dataset", 
                        value=('interpolated_data' in st.session_state),
                        disabled=('interpolated_data' not in st.session_state),
                        key="db_save_interpolated"
                    )
                
                # Dataset name input
                dataset_name = st.text_input(
                    "Dataset Name", 
                    placeholder="Enter a name for the dataset",
                    key="db_dataset_name"
                )
                
                # Save button
                save_button = st.button(
                    "Save Datasets", 
                    disabled=not (dataset_name and (save_original or save_interpolated)),
                    key="db_save_datasets"
                )
        
        # Model Management tab
        with db_tab2:
            st.subheader("Model Management")
            
            # Saved models section
            st.markdown("### Saved Models")
            
            # Check for saved models
            # This would be replaced with actual database query
            if 'saved_models' not in st.session_state:
                st.session_state.saved_models = []
            
            if not st.session_state.saved_models:
                st.info("No models saved in the database")
            else:
                # Display saved models with multiselect
                selected_models = st.multiselect(
                    "Select models",
                    st.session_state.saved_models,
                    key="db_model_select"
                )
                
                # Model operations
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    load_model_button = st.button(
                        "Load Selected Models", 
                        disabled=not selected_models,
                        key="db_load_models"
                    )
                
                with col2:
                    delete_model_button = st.button(
                        "Delete Selected Models", 
                        disabled=not selected_models,
                        key="db_delete_models"
                    )
                
                with col3:
                    view_model_button = st.button(
                        "View Model Details", 
                        disabled=len(selected_models) != 1,
                        key="db_view_model_details"
                    )
            
            # Save current model section
            st.markdown("### Save Current Models")
            
            # Check if models are in session state
            has_models = 'trained_models' in st.session_state and st.session_state.trained_models
            
            if not has_models:
                st.info("No trained models available to save")
            else:
                # Model selection
                available_models = list(st.session_state.trained_models.keys())
                models_to_save = st.multiselect(
                    "Select models to save",
                    available_models,
                    key="db_models_to_save"
                )
                
                # Model name input
                col1, col2 = st.columns(2)
                
                with col1:
                    model_name = st.text_input(
                        "Model Name", 
                        placeholder="Enter a name for the model",
                        key="db_model_name"
                    )
                
                with col2:
                    model_description = st.text_input(
                        "Model Description", 
                        placeholder="Enter a description (optional)",
                        key="db_model_description"
                    )
                
                # Save button
                save_model_button = st.button(
                    "Save Models", 
                    disabled=not (model_name and models_to_save),
                    key="db_save_models"
                )
        
        # Export/Import tab
        with db_tab3:
            st.subheader("Export and Import")
            
            # Export section
            st.markdown("### Export Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Export Datasets**")
                
                has_datasets = ('original_data' in st.session_state or 
                               'interpolated_data' in st.session_state or 
                               ('saved_datasets' in st.session_state and st.session_state.saved_datasets))
                
                if not has_datasets:
                    st.info("No datasets available to export")
                else:
                    export_format = st.selectbox(
                        "Export Format",
                        ["CSV", "Excel", "JSON"],
                        key="db_export_format"
                    )
                    
                    export_button = st.button(
                        "Export Datasets", 
                        key="db_export_datasets"
                    )
            
            with col2:
                st.markdown("**Export Models**")
                
                has_models = ('trained_models' in st.session_state and st.session_state.trained_models) or \
                            ('saved_models' in st.session_state and st.session_state.saved_models)
                
                if not has_models:
                    st.info("No models available to export")
                else:
                    export_model_button = st.button(
                        "Export Models", 
                        key="db_export_models"
                    )
            
            # Import section
            st.markdown("### Import Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Import Datasets**")
                
                dataset_file = st.file_uploader(
                    "Upload Dataset File", 
                    type=["csv", "xlsx", "json"],
                    key="db_dataset_file"
                )
                
                if dataset_file is not None:
                    import_dataset_button = st.button(
                        "Import Dataset", 
                        key="db_import_dataset"
                    )
            
            with col2:
                st.markdown("**Import Models**")
                
                model_file = st.file_uploader(
                    "Upload Model File", 
                    type=["pkl", "joblib", "json"],
                    key="db_model_file"
                )
                
                if model_file is not None:
                    import_model_button = st.button(
                        "Import Model", 
                        key="db_import_model"
                    )