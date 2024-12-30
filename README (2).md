
# DFI Prediction Tool

A Streamlit web application for predicting DNA Fragmentation Index (DFI) based on sperm parameters.

## Features
- Input sperm motility and morphology parameters
- Predict DFI using a trained Random Forest model
- Interpretation guide for DFI values

## Usage Instructions

1. Clone this repository to your local machine.
2. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit app locally:
   ```
   streamlit run dfi_prediction_app.py
   ```
4. Enter the sperm parameters in the sidebar and click 'Predict DFI' to get the results.

## Deployment to Streamlit Cloud

1. Push this repository to GitHub.
2. Connect your GitHub repository to Streamlit Cloud.
3. Deploy the app through the Streamlit Cloud dashboard.

## Files in the Repository
- `dfi_prediction_app.py`: The Streamlit app script.
- `ensemble_model.pkl`: The trained Random Forest model.
- `requirements.txt`: List of dependencies.
- `README.md`: This file with usage and deployment instructions.

