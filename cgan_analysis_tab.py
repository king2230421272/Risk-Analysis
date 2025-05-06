import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from modules.advanced_data_processing import AdvancedDataProcessor

def cgan_analysis_tab(analysis_tabs_index, original_data, data_handler, advanced_processor):
    """
    CGAN Analysis Tab implementation
    """
    with analysis_tabs_index:
        st.write("### CGAN Analysis")
        st.write("""
        Conditional Generative Adversarial Network (CGAN) analysis evaluates if the interpolated data
        follows the same distribution as the original data by learning the conditional distribution.
        """)
        
        # Check if we have data to analyze and if convergence has been achieved
        if 'all_converged' not in st.session_state or not st.session_state.all_converged:
            st.warning("⚠️ Convergence has not been achieved. Please run the convergence evaluation first.")
            st.info("Run the convergence evaluation tab to check if your datasets have converged before running CGAN Analysis.")
            return
            
        # Check if we have best dataset available for analysis
        if 'best_converged_dataset' not in st.session_state:
            st.warning("No best converged dataset available. Please run convergence evaluation first.")
            return
            
        # Get the best dataset
        best_dataset = st.session_state.best_converged_dataset
        st.success(f"Using best converged dataset: Dataset {best_dataset['id']}")
        
        # Check if we have original data
        if original_data is None:
            st.error("Original data not available. Please import it first.")
            return
        
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
                # Check if in consecutive mode
                auto_run = False
                if ('consecutive_analysis' in st.session_state and 
                    st.session_state.consecutive_analysis and 
                    st.session_state.current_analysis_step == 5):
                    auto_run = True
                    st.info("Running CGAN analysis as part of consecutive analysis...")
                
                if st.button("Run CGAN Analysis", key="cgan_btn") or auto_run:
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
                                best_dataset['data'],
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
                                    best_dataset['data'][col_to_visualize],
                                    cgan_results[f'{col_to_visualize}_mean'],
                                    alpha=0.5
                                )
                                plt.xlabel(f'Interpolated {col_to_visualize}')
                                plt.ylabel(f'CGAN Predicted {col_to_visualize}')
                                plt.title(f'Interpolated vs CGAN Predicted: {col_to_visualize}')
                                
                                # Add perfect prediction line
                                min_val = min(best_dataset['data'][col_to_visualize].min(),
                                            cgan_results[f'{col_to_visualize}_mean'].min())
                                max_val = max(best_dataset['data'][col_to_visualize].max(),
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
                            
                            # Mark as complete if in consecutive mode
                            if auto_run:
                                st.session_state.current_analysis_step = 6  # Move to next step
                                st.session_state.cgan_analysis_complete = True
                                st.success("CGAN analysis completed as part of consecutive process.")
                    except Exception as e:
                        st.error(f"Error during CGAN analysis: {e}")
            else:
                st.warning("Please select at least one condition column and one target column.")