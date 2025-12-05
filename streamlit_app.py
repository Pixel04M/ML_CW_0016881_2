# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------- Page Config --------------------------
st.set_page_config(
    page_title="Vehicle Crash Injury Severity Predictor",
    page_icon="car",
    layout="wide"
)

# -------------------------- Title & Description --------------------------
st.title("Vehicle Crash Injury Severity Prediction")
st.markdown("""
This app predicts the **Injury Severity** of a vehicle crash using a pre-trained machine learning model  
(based on Montgomery County Crash Reporting - Drivers Data).
""")

# -------------------------- Load Model & Artifacts --------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.pkl")
    scaler = joblib.load("scaler.pkl")
    info = joblib.load("preprocessing_info.pkl")
    return model, scaler, info

try:
    model, scaler, info = load_artifacts()
    st.success("Model and preprocessing artifacts loaded successfully!")
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.info("Make sure `best_model.pkl`, `scaler.pkl`, and `preprocessing_info.pkl` are in the same folder.")
    st.stop()

# Extract saved info
selected_features = info['selected_features']
categorical_features = info['categorical_features']
numerical_features = info['numerical_features']
final_feature_names = info['feature_names']
best_model_name = info['best_model']

st.sidebar.header("Input Crash Details")

# -------------------------- Helper Functions --------------------------
def preprocess_user_input(df_user):
    """Apply exactly the same preprocessing pipeline that was used during training."""
    df = df_user.copy()

    # 1. Feature Engineering (same as training)
    if 'Crash Date/Time' in df.columns:
        df['Crash Date/Time'] = pd.to_datetime(df['Crash Date/Time'], errors='coerce')
        df['Crash Hour'] = df['Crash Date/Time'].dt.hour
        df['Crash DayOfWeek'] = df['Crash Date/Time'].dt.dayofweek
    else:
        df['Crash Hour'] = np.nan
        df['Crash DayOfWeek'] = np.nan

    if 'Vehicle Year' in df.columns:
        current_year = 2025
        df['Vehicle Age'] = current_year - df['Vehicle Year'].replace(0, np.nan)
        df['Vehicle Age'] = df['Vehicle Age'].fillna(df['Vehicle Age'].median())

    if 'Driver Substance Abuse' in df.columns:
        df['Has_Substance_Abuse'] = df['Driver Substance Abuse'].apply(
            lambda x: 1 if pd.notna(x) and ('ALCOHOL' in str(x).upper() or 'SUSPECTED' in str(x).upper()) else 0
        )

    # 2. Select only the features we trained on
    for col in selected_features:
        if col not in df.columns:
        df[col] = np.nan

    X = df[selected_features].copy()

    # 3. Fill missing values (same strategy as training)
    for col in selected_features:
        if col in X.columns:
            if X[col].dtype == 'object':
                X[col] = X[col].fillna('Unknown')
            else:
                X[col] = X[col].fillna(X[col].median() if X[col].notna().sum() > 0 else 0)

    # 4. One-hot encode categorical columns (exactly the same columns & prefixes)
    X_encoded = pd.get_dummies(X[categorical_features], prefix=categorical_features, drop_first=True)

    # 5. Combine numerical + encoded categorical
    X_final = pd.concat([X[numerical_features], X_encoded], axis=1)

    # 6. Ensure we have ALL columns the model expects (missing columns → 0)
    for col in final_feature_names:
        if col not in X_final.columns:
            X_final[col] = 0

    # Reorder columns to match training exactly
    X_final = X_final[final_feature_names]

    # 7. Scale numerical features using the fitted scaler
    X_scaled = scaler.transform(X_final)
    return X_scaled

# -------------------------- Sidebar Inputs --------------------------
def user_input_features():
    data = {}

    # Weather
    data['Weather'] = st.sidebar.selectbox(
        "Weather", options=['CLEAR', 'CLOUDY', 'RAINING', 'SNOW', 'FOG', 'UNKNOWN', 'OTHER'], index=6)

    # Surface Condition
    data['Surface Condition'] = st.sidebar.selectbox(
        "Surface Condition", options=['DRY', 'WET', 'SNOW', 'ICE', 'UNKNOWN'], index=0)

    # Light
    data['Light'] = st.sidebar.selectbox(
        "Light Conditions", options=['DAYLIGHT', 'DARK LIGHTS ON', 'DUSK', 'DAWN', 'DARK NO LIGHTS', 'UNKNOWN'], index=0)

    # Traffic Control
    data['Traffic Control'] = st.sidebar.selectbox(
        "Traffic Control", options=['NO CONTROLS', 'TRAFFIC SIGNAL', 'STOP SIGN', 'YIELD SIGN', 'UNKNOWN'], index=0)

    # Collision Type
    data['Collision Type'] = st.sidebar.selectbox(
        "Collision Type", options=['SINGLE VEHICLE', 'HEAD ON', 'REAR END', 'ANGLE', 'SIDESWIPE', 'OTHER'], index=4)

    # Driver At Fault
    data['Driver At Fault'] = st.sidebar.selectbox("Driver At Fault?", options=['Yes', 'No'])

    # Driver Substance Abuse
    data['Driver Substance Abuse'] = st.sidebar.selectbox(
        "Driver Substance Abuse", options=['None', 'Alcohol Present', 'Alcohol Contributed', 'Drugs', 'Unknown'], index=0)

    # Vehicle Damage Extent
    data['Vehicle Damage Extent'] = st.sidebar.selectbox(
        "Vehicle Damage Extent", options=['NONE', 'MINOR', 'MODERATE', 'SEVERE', 'TOTALED', 'UNKNOWN'], index=5)

    # Vehicle Body Type
    data['Vehicle Body Type'] = st.sidebar.selectbox(
        "Vehicle Body Type", options=['PASSENGER CAR', 'SUV', 'PICKUP TRUCK', 'VAN', 'MOTORCYCLE', 'BUS', 'OTHER'], index=0)

    # Vehicle Movement
    data['Vehicle Movement'] = st.sidebar.selectbox(
        "Vehicle Movement", options=['MOVING', 'STOPPED', 'PARKED', 'SLOWING', 'UNKNOWN'], index=0)

    # Speed Limit
    data['Speed Limit'] = st.sidebar.number_input("Speed Limit (mph)", min_value=0, max_value=100, value=35)

    # Vehicle Year
    data['Vehicle Year'] = st.sidebar.number_input("Vehicle Year", min_value=1900, max_value=2025, value=2018)

    # Route Type
    data['Route Type'] = st.sidebar.selectbox(
        "Route Type", options=['Local Road', 'County Road', 'State Highway', 'Interstate', 'Unknown'], index=0)

    # Crash Date/Time (optional – for hour/day extraction)
    crash_datetime = st.sidebar.text_input(
        "Crash Date/Time (e.g., 2024-05-20 14:30:00)", value="2025-01-01 12:00:00")
    data['Crash Date/Time'] = crash_datetime

    # Convert Yes/No to proper format
    data['Driver At Fault'] = 'Yes' if data['Driver At Fault'] == 'Yes' else 'No'

    return pd.DataFrame([data])

# Get user input
input_df = user_input_features()

# -------------------------- Prediction --------------------------
if st.sidebar.button("Predict Injury Severity"):
    with st.spinner("Processing input and making prediction..."):
        try:
            X_processed = preprocess_user_input(input_df)
            prediction = model.predict(X_processed)[0]
            prediction_proba = model.predict_proba(X_processed)[0]

            # Map class probabilities
            classes = model.classes_
            proba_df = pd.DataFrame({
                'Injury Severity': classes,
                'Probability': prediction_proba
            }).sort_values('Probability', ascending=False)

            st.subheader("Prediction Result")
            st.success(f"**Predicted Injury Severity: {prediction}**")

            st.subheader("Prediction Probabilities")
            st.dataframe(proba_df.style.format({"Probability": "{:.2%}"}))

            # Visual bar chart of probabilities
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=proba_df, x='Probability', y='Injury Severity', palette="viridis", ax=ax)
            ax.set_title("Prediction Confidence per Class")
            ax.set_xlabel("Probability")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.exception(e)

# -------------------------- Additional Info --------------------------
st.sidebar.markdown("---")
st.sidebar.info(f"**Best Model Used:** {best_model_name}")

st.markdown("---")
st.caption("""
Note: This model was trained on historical Montgomery County, MD crash data.  
Predictions are probabilistic and should not be used as the sole basis for safety decisions.
""")