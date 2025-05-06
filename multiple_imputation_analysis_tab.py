import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def multiple_imputation_analysis_tab(analysis_tabs_index, datasets_to_analyze=None):
    """
    Multiple Imputation Analysis Tab implementation
    
    This tab sequentially runs all three analysis methods (K-Means, Regression, PCA),
    evaluates convergence, and if convergence is achieved, outputs datasets for CGAN Analysis.
    """
    with analysis_tabs_index:
        st.write("### Multiple Imputation Analysis")
        st.write("""
        This analysis sequentially runs all three methods (K-Means Clustering, Regression Analysis, PCA Factor Analysis), 
        evaluates convergence, and if all converged, outputs the datasets for CGAN Analysis.
        """)
        
        # Check if we have datasets to analyze
        if 'convergence_datasets' not in st.session_state or not st.session_state.convergence_datasets:
            st.warning("No datasets available for analysis. Please interpolate data first.")
            return
        
        # Create a button to start the sequential analysis
        if st.button("Run Multiple Imputation Analysis", key="multiple_imputation_btn"):
            st.info("Starting Multiple Imputation Analysis... This will run all analysis steps in sequence.")
            
            # Get available datasets
            dataset_options = ["Original Data"] + [f"Interpolated Dataset {ds['id']}" for ds in st.session_state.convergence_datasets]
            
            # Let user select datasets or use the provided ones
            if datasets_to_analyze is None:
                st.write("### Select datasets to analyze:")
                selected_datasets = []
                
                # Display checkboxes for dataset selection
                for ds_option in dataset_options:
                    selected = st.checkbox(ds_option, key=f"multi_imp_{ds_option}")
                    if selected:
                        selected_datasets.append(ds_option)
                
                # Validate selection
                if not selected_datasets:
                    st.warning("Please select at least one dataset to analyze.")
                    return
                    
                st.session_state.datasets_to_analyze = selected_datasets
            else:
                selected_datasets = datasets_to_analyze
                st.session_state.datasets_to_analyze = selected_datasets
                st.write(f"Analyzing datasets: {', '.join(selected_datasets)}")
            
            # Initialize the sequential analysis
            st.session_state.consecutive_analysis = True
            st.session_state.current_analysis_step = 0
            
            # Start with the first step
            st.warning("Step 1: Running K-Means clustering analysis...")
            st.session_state.current_analysis_step = 1
            
            # Show progress and next steps
            progress_container = st.empty()
            progress_container.info("Starting analysis with K-Means Clustering. Please go to the Cluster Analysis tab to see progress.")
            
            # Instruct user to navigate to the appropriate tab
            st.info("Please go to the 'Cluster Analysis (K-Means)' tab to see the analysis progress.")
        
        # Display the current state of sequential analysis
        if 'consecutive_analysis' in st.session_state and st.session_state.consecutive_analysis:
            current_step = st.session_state.current_analysis_step
            
            # Show progress based on the current step
            if current_step == 0:
                st.info("Sequential analysis not started. Press the button above to begin.")
            elif current_step == 1:
                st.info("Step 1 in progress: Running K-Means clustering on selected datasets...")
            elif current_step == 2:
                st.info("Step 2 in progress: Running Regression Analysis on selected datasets...")
            elif current_step == 3:
                st.info("Step 3 in progress: Running PCA Factor Analysis on selected datasets...")
            elif current_step == 4:
                st.info("Step 4 in progress: Evaluating convergence across all analyzed datasets...")
            elif current_step == 5:
                # Check if convergence was achieved
                if 'all_converged' in st.session_state and st.session_state.all_converged:
                    st.success("✅ Convergence achieved! Running CGAN Analysis on the best converged dataset.")
                    st.info("Please go to the 'CGAN Analysis' tab to see the final results.")
                else:
                    st.warning("❌ Convergence not achieved. The process needs more iterations.")
                    st.info("Please go back to Step 1: MCMC Interpolation to continue the iteration.")
            elif current_step == 6:
                st.success("🎉 Multiple Imputation Analysis completed successfully!")
                
                # Show the result and next steps
                if 'cgan_analysis_complete' in st.session_state and st.session_state.cgan_analysis_complete:
                    st.write("### Analysis Results Summary:")
                    st.write("1. All three analysis methods completed successfully")
                    st.write("2. Convergence evaluation passed")
                    st.write("3. CGAN Analysis completed on the best converged dataset")
                    
                    # Offer to view results
                    st.info("Please check the CGAN Analysis tab to view and download the detailed results.")
                else:
                    st.info("CGAN Analysis is the final step. Please check the CGAN Analysis tab to complete the process.")
        
        # Option to reset the sequential analysis
        if 'consecutive_analysis' in st.session_state and st.session_state.consecutive_analysis:
            if st.button("Reset Multiple Imputation Analysis", key="reset_multi_imp"):
                st.session_state.consecutive_analysis = False
                st.session_state.current_analysis_step = 0
                if 'datasets_to_analyze' in st.session_state:
                    del st.session_state.datasets_to_analyze
                st.success("Multiple Imputation Analysis process has been reset.")
                st.rerun()