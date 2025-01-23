import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import time  # For introducing delay

# Streamlit UI Configuration
st.set_page_config(layout="wide", page_title="Disaster Prediction Dashboard")
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1584265582261-a2909b244d47?q=80&w=2900&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .stButton > button {
        background-color: #007bff !important;
        color: white !important;
        width: 100%;
        border-radius: 8px;
    }
    .parameter-container {
        background-color: transparent; /* Removed white background */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    section[data-testid="stSidebar"] > div {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 20px;
        border-radius: 10px;
    }
    .black-text {
        color: black !important;
        font-weight: bold;
    }
    .stTextInput input {
        background-color: white;
    }
    /* White text for Dataset Preview */
    .stMarkdown {
        color: white !important;
    }
    .stSubheader {
        color: white !important;
    }
    .result-container {
        background-color: rgba(0, 0, 0, 0.7); 
        color: white; 
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 30px;
    }
    .result-heading {
        font-size: 48px;
        font-weight: bold;
        color: #00bfff;
        margin: 0;
    }
    .result-subheading {
        font-size: 24px;
        color: #dcdcdc;
        margin-top: 10px;
    }
    .warning-container {
        background-color: #f8d7da; /* Light red background for warning */
        color: #721c24; /* Dark red text color */
        border: 1px solid #f5c6cb;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 30px;
    }
    .warning-heading {
        font-size: 48px;
        font-weight: bold;
        color: #721c24;
    }
    .warning-subheading {
        font-size: 24px;
        color: #721c24;
        margin-top: 10px;
    }
    .safe-container {
        background-color: #d4edda; /* Light green background for safe result */
        color: #155724; /* Dark green text color */
        border: 1px solid #c3e6cb;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-top: 30px;
    }
    .safe-heading {
        font-size: 48px;
        font-weight: bold;
        color: #155724;
    }
    .safe-subheading {
        font-size: 24px;
        color: #155724;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Page Title
st.markdown("""
    <h1 style="color: black; padding: 10px 20px;">🌍 Disaster Prediction Dashboard</h1>
""", unsafe_allow_html=True)

# Page Title Description - Highlighted
st.markdown("""
    <div style="background-color: white; color: black; padding: 10px 20px; border-radius: 10px; box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);">
        This application predicts the likelihood of <b>Hurricanes</b>, <b>Earthquakes</b>, and <b>Floods</b> based on your dataset input using <b>Machine Learning</b> models like Random Forest and Logistic Regression.
    </div>
""", unsafe_allow_html=True)

# Sidebar Model Selection and Data Upload
model_choice = st.sidebar.selectbox("Select Disaster Prediction Model", ["Hurricane Prediction", "Earthquake Prediction", "Flood Prediction"])
uploaded_file = st.sidebar.file_uploader("Upload your Dataset (CSV)", type=["csv"])

# Model Training Functions
@st.cache_resource
def train_model(df, model_type):
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X, y = None, None

    if model_type == "Hurricane":
        df[['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)']] = imputer.fit_transform(df[['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)']])
        df['Hurricane'] = np.where(df['Wind Speed (km/h)'] > 120, 1, 0)
        X, y = df[['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)']], df['Hurricane']
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    elif model_type == "Earthquake":
        df[['Temperature (°C)', 'Humidity (%)', 'Magnitude', 'Wind Speed (km/h)']] = imputer.fit_transform(df[['Temperature (°C)', 'Humidity (%)', 'Magnitude', 'Wind Speed (km/h)']])
        df['Earthquake'] = np.where(df['Magnitude'] > 6.0, 1, 0)
        X, y = df[['Temperature (°C)', 'Humidity (%)', 'Magnitude', 'Wind Speed (km/h)']], df['Earthquake']
        model = LogisticRegression(random_state=42)

    else:
        df[['Rainfall (mm)', 'River Level (m)', 'Soil Moisture (%)']] = imputer.fit_transform(df[['Rainfall (mm)', 'River Level (m)', 'Soil Moisture (%)']])
        df['Flood'] = np.where((df['Rainfall (mm)'] > 100) & (df['River Level (m)'] > 5), 1, 0)
        X, y = df[['Rainfall (mm)', 'River Level (m)', 'Soil Moisture (%)']], df['Flood']
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy, scaler

# Main Application Logic
if uploaded_file is not None:
    st.markdown('<p style="color: white; font-size: 24px;">Dataset Preview</p>', unsafe_allow_html=True)
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    if model_choice == "Hurricane Prediction":
        required_columns = ['Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)']
        model_type = "Hurricane"
    elif model_choice == "Earthquake Prediction":
        required_columns = ['Temperature (°C)', 'Humidity (%)', 'Magnitude', 'Wind Speed (km/h)']
        model_type = "Earthquake"
    else:
        required_columns = ['Rainfall (mm)', 'River Level (m)', 'Soil Moisture (%)']
        model_type = "Flood"

    if not all(col in df.columns for col in required_columns):
        st.error(f"⚠️ Missing columns: {', '.join([col for col in required_columns if col not in df.columns])}")
    else:
        model, accuracy, scaler = train_model(df, model_type)
        st.sidebar.success(f"✅ Model Trained Successfully! Accuracy: {accuracy:.2f}")

        st.subheader(f"Input Parameters for {model_choice}")
        
        # Removed the background color from the container here
        st.markdown('<div class="parameter-container">', unsafe_allow_html=True)
        
        inputs = []
        for col in required_columns:
            value = st.number_input(f"{col}", value=float(df[col].mean()))
            inputs.append(value)

        if st.button("Predict"):
            # Simulate loading time
            with st.spinner("Analyzing... please wait...."):
                time.sleep(2)
            
            # Make prediction
            inputs_scaled = scaler.transform([inputs])
            prediction = model.predict(inputs_scaled)

            # Display results with appropriate styling
            result_message = "Risk" if prediction == 1 else "No immediate risk detected"

            if result_message == "Risk":
                st.markdown(f"""
                    <div class="warning-container">
                        <h2 class="warning-heading">Warning: {model_choice.split()[0]}</h2>
                        <p class="warning-subheading">The model predicts a high likelihood of a {model_choice.split()[0]} occurring. Please take precautionary measures.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="safe-container">
                        <h2 class="safe-heading">No immediate risk detected</h2>
                        <p class="safe-subheading">The model indicates no immediate risk of a {model_choice.split()[0]} occurring.-3The situation seems safe as of now..</p>
                    </div>
                """, unsafe_allow_html=True)
        

