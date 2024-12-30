
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Page config
st.set_page_config(
    page_title="DFI Prediction Tool",
    page_icon="🧬",
    layout="wide"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open('ensemble_model.pkl', 'rb') as model_file:
        return pickle.load(model_file)

try:
    model = load_model()
    
    # App title and description
    st.title('🧬 DFI Prediction Tool')
    st.write('Enter sperm parameters to predict DNA Fragmentation Index (DFI)')
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('Motility Parameters')
        progressive = st.number_input('Progressive (%)', 
                                    min_value=0.0, 
                                    max_value=100.0, 
                                    value=50.0, 
                                    step=0.1,
                                    format="%.1f")
        non_progressive = st.number_input('Non-progressive (%)', 
                                        min_value=0.0, 
                                        max_value=100.0, 
                                        value=10.0, 
                                        step=0.1,
                                        format="%.1f")
        immotile = st.number_input('Immotile (%)', 
                                  min_value=0.0, 
                                  max_value=100.0, 
                                  value=40.0, 
                                  step=0.1,
                                  format="%.1f")
    
    with col2:
        st.subheader('Other Parameters')
        concentration = st.number_input('Concentration (million/mL)', 
                                      min_value=0.0, 
                                      max_value=300.0, 
                                      value=50.0, 
                                      step=0.1,
                                      format="%.1f")
        normal_sperm = st.number_input('Normal Morphology (%)', 
                                      min_value=0.0, 
                                      max_value=100.0, 
                                      value=14.0, 
                                      step=0.1,
                                      format="%.1f")
    
    # Add validation for percentages
    total_percentage = progressive + non_progressive + immotile
    if total_percentage != 100.0:
        st.warning(f'Total motility percentages should equal 100%. Current total: {total_percentage:.1f}%')
    
    # Add a predict button
    if st.button('Predict DFI', type='primary', disabled=(total_percentage != 100.0)):
        # Create input array
        input_features = np.array([[progressive, immotile, non_progressive, concentration, normal_sperm]])
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        
        # Display results
        st.markdown('---')
        st.subheader('Prediction Results')
        
        # Create columns for displaying results
        res_col1, res_col2, res_col3 = st.columns(3)
        
        with res_col2:
            st.metric(label="Predicted DFI", value=f"{prediction:.1f}%")
        
        # Add interpretation
        st.markdown('---')
        st.subheader('Interpretation Guide')
        if prediction < 15:
            st.success('DFI < 15%: Generally considered normal/good fertility potential')
        elif prediction < 25:
            st.warning('DFI 15-25%: Moderate fertility impact, may affect pregnancy outcomes')
        else:
            st.error('DFI > 25%: Higher impact on fertility, may indicate need for additional evaluation')

except Exception as e:
    st.error(f'An error occurred: {str(e)}')
    st.info('Please make sure all model files are properly loaded and try again.')
