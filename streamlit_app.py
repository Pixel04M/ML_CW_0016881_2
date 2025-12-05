"""
Streamlit Multi-Page Application for Crash Reporting Analysis
Deployment for ML Coursework
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crash Reporting Analysis",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Sidebar navigation
st.sidebar.title("🚗 Crash Analysis Dashboard")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📊 Data Exploration", "🔧 Preprocessing", "🤖 Model Training", "📈 Model Evaluation", "🔮 Prediction"]
)

# Home Page
if page == "🏠 Home":
    st.markdown('<div class="main-header">🚗 Crash Reporting Analysis Dashboard</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ## Welcome to the Crash Reporting Analysis Application
    
    This application provides comprehensive analysis of vehicle crash data to predict injury severity.
    
    ### Features:
    - **Data Exploration**: Interactive exploration of crash reporting data
    - **Preprocessing**: Data cleaning and feature engineering options
    - **Model Training**: Train machine learning models with hyperparameter tuning
    - **Model Evaluation**: Compare and evaluate different ML algorithms
    - **Prediction**: Make predictions on new crash data
    
    ### Dataset Information
    The dataset contains information about vehicle crashes including:
    - Weather conditions
    - Road surface conditions
    - Driver behavior and characteristics
    - Vehicle information
    - Crash details and outcomes
    
    ### Getting Started
    1. Navigate to **Data Exploration** to explore the dataset
    2. Use **Preprocessing** to clean and prepare the data
    3. Go to **Model Training** to train ML models
    4. Check **Model Evaluation** to compare model performance
    5. Use **Prediction** to make predictions on new data
    """)
    
    # Load dataset button
    if st.button("Load Dataset", type="primary"):
        try:
            df = pd.read_csv('Crash_Reporting_-_Drivers_Data.csv')
            st.session_state.data = df
            st.success(f"✅ Dataset loaded successfully! Shape: {df.shape}")
        except FileNotFoundError:
            st.error("❌ Dataset file not found. Please ensure 'Crash_Reporting_-_Drivers_Data.csv' is in the current directory.")
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")

# Data Exploration Page
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load the dataset from the Home page first.")
        if st.button("Load Dataset Now"):
            try:
                df = pd.read_csv('Crash_Reporting_-_Drivers_Data.csv')
                st.session_state.data = df
                st.success("✅ Dataset loaded!")
                st.rerun()
            except:
                st.error("❌ Could not load dataset.")
    else:
        df = st.session_state.data
        
        # Dataset overview
        st.header("Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", f"{df.shape[0]:,}")
        col2.metric("Total Columns", df.shape[1])
        col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        col4.metric("Duplicate Rows", df.duplicated().sum())
        
        # Display data
        st.subheader("Data Preview")
        num_rows = st.slider("Number of rows to display", 5, 100, 10)
        st.dataframe(df.head(num_rows))
        
        # Summary statistics
        st.subheader("Summary Statistics")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe())
        
        # Visualizations
        st.subheader("Visualizations")
        
        # Target variable distribution
        if 'Injury Severity' in df.columns:
            st.write("### Injury Severity Distribution")
            injury_counts = df['Injury Severity'].value_counts()
            fig, ax = plt.subplots(figsize=(10, 6))
            injury_counts.plot(kind='bar', ax=ax)
            ax.set_title('Distribution of Injury Severity')
            ax.set_xlabel('Injury Severity')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

# Preprocessing Page
elif page == "🔧 Preprocessing":
    st.title("🔧 Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please load the dataset from the Home page first.")
    else:
        df = st.session_state.data.copy()
        st.info("💡 Preprocessing steps are demonstrated in the Jupyter notebook. This page shows the preprocessing pipeline.")
        
        st.subheader("Preprocessing Steps Applied:")
        st.markdown("""
        1. **Missing Values**: Filled categorical with 'Unknown', numerical with median
        2. **Scaling**: StandardScaler applied for Logistic Regression
        3. **Error Correction**: Removed duplicates, invalid data
        4. **Feature Engineering**: Created vehicle age, crash hour, day of week, substance abuse binary
        5. **Encoding**: One-hot encoding for categorical variables
        6. **Train-Test Split**: 60% train, 20% validation, 20% test
        """)

# Model Training Page
elif page == "🤖 Model Training":
    st.title("🤖 Model Training")
    
    st.info("💡 Models are trained in the Jupyter notebook with hyperparameter tuning using GridSearchCV.")
    
    st.subheader("Models Trained:")
    st.markdown("""
    1. **Random Forest Classifier** - with GridSearchCV hyperparameter tuning
    2. **Gradient Boosting Classifier** - with GridSearchCV hyperparameter tuning
    3. **Logistic Regression** - with GridSearchCV hyperparameter tuning
    
    All models use 3-fold cross-validation for hyperparameter optimization.
    """)
    
    # Load model if available
    try:
        model = joblib.load('best_model.pkl')
        st.session_state.model = model
        st.success("✅ Best model loaded successfully!")
        
        # Try to load scaler
        try:
            scaler = joblib.load('scaler.pkl')
            st.session_state.scaler = scaler
        except:
            pass
    except:
        st.warning("⚠️ Model file not found. Please run the notebook first to train models.")

# Model Evaluation Page
elif page == "📈 Model Evaluation":
    st.title("📈 Model Evaluation")
    
    st.subheader("Evaluation Metrics Used:")
    st.markdown("""
    - **Accuracy**: Overall correctness of predictions
    - **Precision**: Proportion of positive predictions that are correct
    - **Recall**: Proportion of actual positives correctly identified
    - **F1-Score**: Harmonic mean of precision and recall
    - **Confusion Matrix**: Detailed breakdown of prediction errors
    - **Classification Report**: Per-class metrics
    """)
    
    st.info("💡 Detailed evaluation results are available in the Jupyter notebook, including confusion matrices and classification reports for all three models.")

# Prediction Page
elif page == "🔮 Prediction":
    st.title("🔮 Make Predictions")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please load the model from the Model Training page first.")
        try:
            model = joblib.load('best_model.pkl')
            st.session_state.model = model
            try:
                scaler = joblib.load('scaler.pkl')
                st.session_state.scaler = scaler
            except:
                pass
            st.rerun()
        except:
            st.error("❌ Model file not found. Please run the notebook first.")
    else:
        st.header("Predict Injury Severity")
        st.subheader("Enter Crash Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            weather = st.selectbox("Weather", ["Clear", "Cloudy", "Rain", "Snow", "Other"])
            surface = st.selectbox("Surface Condition", ["Dry", "Wet", "Snow", "Ice", "Other"])
            light = st.selectbox("Light", ["Daylight", "Dark - Lighted", "Dark - Not Lighted", "Dusk", "Dawn"])
            collision_type = st.selectbox("Collision Type", ["Front to Rear", "Front to Front", "Angle", "Sideswipe, Same Direction", "Single Vehicle", "Other"])
        
        with col2:
            driver_at_fault = st.selectbox("Driver At Fault", ["Yes", "No"])
            speed_limit = st.slider("Speed Limit", 0, 100, 40)
            vehicle_year = st.number_input("Vehicle Year", min_value=1900, max_value=2025, value=2015)
            route_type = st.selectbox("Route Type", ["Interstate (State)", "US (State)", "Maryland (State) Route", "County Route", "Other"])
        
        if st.button("Predict Injury Severity", type="primary"):
            st.info("💡 For accurate predictions, the input features must match the exact feature engineering pipeline used during training. Please refer to the notebook for the complete prediction pipeline.")
            st.success("✅ Prediction functionality requires matching feature engineering from training phase. See notebook for full implementation.")

if __name__ == "__main__":
    pass

