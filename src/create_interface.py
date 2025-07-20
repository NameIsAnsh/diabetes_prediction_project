#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Web Interface for Diabetes Prediction System
"""

import joblib  # Replaces pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import streamlit as st
import shutil
# The user's original code included this import.
# It's assumed to exist in the project structure.
# from utils import predict_diabetes

# Set page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Model Loading ---
# FIX: Use Streamlit's caching to load the model robustly and only once.
# This is the recommended practice for loading heavy resources.
@st.cache_resource
def load_model():
    """
    Loads the saved model pipeline.
    NOTE: This assumes the saved file is the model/pipeline object itself,
    not a function. This is the standard practice to avoid loading errors.
    If 'predict_function.pkl' contains the model object, it will work.
    """
    try:
        models_dir = Path('models')
        # We load the file that contains the trained model/pipeline object.
        model_pipeline = joblib.load(models_dir / 'predict_function.pkl')
        return model_pipeline
    except FileNotFoundError:
        st.error("Model file ('predict_function.pkl') not found in the 'models' directory.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.info("This error can occur if the saved file is a function instead of a model object. The recommended solution is to save only the trained scikit-learn pipeline/model.")
        return None

# Load the model using the cached function
model_pipeline = load_model()

# --- Prediction Logic ---
# FIX: The prediction logic is now defined directly within the Streamlit app.
# This makes the app self-contained and avoids dependency issues from the saved file.
def run_prediction(input_data, pipeline):
    """
    Takes user input and the loaded model pipeline to return a prediction.
    """
    if pipeline is None:
        st.error("Model is not loaded. Cannot make a prediction.")
        return None

    # Convert the input dictionary to a DataFrame for the model
    input_df = pd.DataFrame([input_data])
    
    # Use the loaded pipeline to predict
    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]  # Probability of '1' (diabetes)

    # Determine risk level based on probability
    if probability > 0.7:
        risk_level = 'High'
    elif probability > 0.3:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'
    
    return {'prediction': prediction, 'probability': probability, 'risk_level': risk_level}


# Define the app
def main():
    # Sidebar
    st.sidebar.image("https://img.freepik.com/free-vector/diabetes-round-concept_1284-37921.jpg", width=200)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Prediction Tool", "Model Performance", "About"])
    
    if page == "Home":
        show_home()
    elif page == "Prediction Tool":
        show_prediction_tool()
    elif page == "Model Performance":
        show_model_performance()
    else:
        show_about()

def show_home():
    st.title("Diabetes Risk Prediction System")
    st.markdown("""
    ## Welcome to the Diabetes Risk Assessment Tool
    
    This application uses machine learning to predict the risk of diabetes based on several health indicators.
    
    ### How to use this tool:
    
    1. Navigate to the **Prediction Tool** page using the sidebar
    2. Enter your health information
    3. Click "Predict" to see your diabetes risk assessment
    4. Review the results and recommendations
    
    ### About Diabetes
    
    Diabetes is a chronic health condition that affects how your body turns food into energy. 
    If you have diabetes, your body either doesn't make enough insulin or can't use the insulin it makes as well as it should.
    
    Early detection and management of diabetes can prevent complications and improve quality of life.
    """)
    
    st.markdown("""
    ### Dataset Information
    
    This prediction model was trained on the Pima Indians Diabetes Dataset, which includes health metrics from female patients of Pima Indian heritage.
    """)
    
    st.image("http://googleusercontent.com/file_content/1", caption="Correlation between different health metrics and diabetes")
    
    st.markdown("""
    ### Key Risk Factors
    
    Based on our analysis, the following factors have the strongest correlation with diabetes risk:
    
    1. **Glucose Level**: Higher blood glucose levels are strongly associated with diabetes
    2. **BMI (Body Mass Index)**: Higher BMI values indicate increased risk
    3. **Age**: Risk increases with age
    4. **Diabetes Pedigree Function**: Family history of diabetes increases risk
    5. **Number of Pregnancies**: More pregnancies are associated with higher risk in this dataset
    
    Use the prediction tool to assess your personal risk based on these and other factors.
    """)

def show_prediction_tool():
    st.title("Diabetes Risk Prediction Tool")
    st.markdown("Enter your health information below to get a personalized diabetes risk assessment.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
    
    with col2:
        insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=1000, value=80)
        bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=70.0, value=25.0, format="%.1f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
        age = st.number_input("Age (years)", min_value=21, max_value=100, value=30)
    
    # Create input dictionary
    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    
    # Add predict button
    if st.button("Predict Diabetes Risk"):
        # Make prediction using the new self-contained function
        result = run_prediction(input_data, model_pipeline)
        
        if result:
            # Display result
            st.markdown("## Prediction Result")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                if result['prediction'] == 1:
                    st.error("**Prediction: Positive for Diabetes Risk**")
                else:
                    st.success("**Prediction: Negative for Diabetes Risk**")
                
                st.metric("Risk Probability", f"{result['probability']:.2%}")
                
                if result['risk_level'] == 'High':
                    st.markdown("**Risk Level:** 🔴 High")
                elif result['risk_level'] == 'Medium':
                    st.markdown("**Risk Level:** 🟠 Medium")
                else:
                    st.markdown("**Risk Level:** 🟢 Low")
            
            with res_col2:
                st.markdown("### Recommendations")
                
                if result['risk_level'] == 'High':
                    st.markdown("""
                    - **Consult a healthcare provider immediately** for proper diagnosis and treatment
                    - Monitor your blood glucose levels regularly
                    - Adopt a balanced diet low in sugar and refined carbohydrates
                    - Engage in regular physical activity
                    - Maintain a healthy weight
                    """)
                elif result['risk_level'] == 'Medium':
                    st.markdown("""
                    - **Schedule a check-up** with your healthcare provider
                    - Consider getting tested for prediabetes
                    - Reduce sugar and refined carbohydrate intake
                    - Increase physical activity to at least 150 minutes per week
                    - Aim for a healthy weight
                    """)
                else:
                    st.markdown("""
                    - Continue maintaining a healthy lifestyle
                    - Get regular check-ups with your healthcare provider
                    - Stay physically active
                    - Eat a balanced diet
                    - Monitor your weight
                    """)
            
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Diabetes Risk Probability')
            ax.set_xticks([0, 0.3, 0.7, 1])
            ax.set_xticklabels(['0%', '30%', '70%', '100%'])
            ax.set_yticks([])
            
            ax.axvspan(0, 0.3, color='green', alpha=0.3)
            ax.axvspan(0.3, 0.7, color='orange', alpha=0.3)
            ax.axvspan(0.7, 1, color='red', alpha=0.3)
            
            ax.plot(result['probability'], 0.5, 'ko', markersize=12)
            
            st.pyplot(fig)
            
            st.markdown("""
            **Disclaimer:** This tool provides an estimate of diabetes risk based on machine learning models. 
            It is not a medical diagnosis. Always consult with healthcare professionals for proper medical advice and diagnosis.
            """)

def show_model_performance():
    st.title("Model Performance Analysis")
    st.markdown("""
    This page presents the performance metrics and visualizations of our diabetes prediction model.
    """)
    
    st.header("Model Comparison")
    st.markdown("""
    We trained and evaluated several machine learning models to find the best performer for diabetes prediction.
    The K-Nearest Neighbors model achieved the highest F1 score and was selected as our final model.
    """)
    
    st.image("http://googleusercontent.com/file_content/2", 
             caption="Performance comparison of different machine learning models")
    
    st.header("Confusion Matrix")
    st.markdown("""
    The confusion matrix shows the model's prediction performance:
    - **True Negatives:** Correctly predicted non-diabetic cases
    - **False Positives:** Non-diabetic cases incorrectly predicted as diabetic
    - **False Negatives:** Diabetic cases incorrectly predicted as non-diabetic
    - **True Positives:** Correctly predicted diabetic cases
    """)
    
    st.image("http://googleusercontent.com/file_content/5", 
             caption="Confusion Matrix showing prediction performance percentages")
    
    st.header("ROC Curve")
    st.markdown("""
    The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate against the False Positive Rate.
    The Area Under the Curve (AUC) is a measure of the model's ability to distinguish between classes.
    """)
    
    st.image("http://googleusercontent.com/file_content/3", 
             caption="ROC Curve showing model's classification performance")
    
    st.header("Precision-Recall Curve")
    st.markdown("""
    The Precision-Recall curve shows the tradeoff between precision and recall for different thresholds.
    This is particularly useful for imbalanced datasets like ours.
    """)
    
    st.image("http://googleusercontent.com/file_content/4", 
             caption="Precision-Recall Curve")

def show_about():
    st.title("About This Project")
    st.markdown("""
    ## Diabetes Risk Prediction System
    
    This project implements a machine learning approach to predict diabetes risk based on health metrics.
    
    ### Project Components
    
    1. **Data Collection and Analysis**: Using the Pima Indians Diabetes Dataset
    2. **Machine Learning Models**: Implementation of various classification algorithms
    3. **Model Evaluation**: Comprehensive performance assessment
    4. **Web Interface**: User-friendly tool for diabetes risk prediction
    
    ### Technologies Used
    
    - **Python**: Core programming language
    - **Scikit-learn**: Machine learning library
    - **Pandas & NumPy**: Data manipulation
    - **Matplotlib & Seaborn**: Data visualization
    - **Streamlit**: Web interface development
    
    ### Research Background
    
    This project is based on research in machine learning applications for healthcare, specifically in diabetes prediction.
    The approach follows methodologies described in recent literature on predictive modeling for chronic diseases.
    
    ### References
    
    For a complete list of references and research papers that informed this project, please refer to the project documentation.
    
    ### Disclaimer
    
    This tool is for educational and research purposes only. It is not intended to replace professional medical advice, diagnosis, or treatment.
    Always consult qualified healthcare providers with any questions regarding medical conditions.
    """)

if __name__ == "__main__":
    main()
