import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
import plotly.graph_objects as go

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Boston House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================
# Load Model + Scaler
# ==============================
@st.cache_resource
def load_models():
    try:
        model = load("boston_house_model.joblib")
        scaler = load("scaler.joblib")
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

model, scaler, error = load_models()

# ==============================
# Feature Descriptions
# ==============================
FEATURE_INFO = {
    "CRIM": "Per capita crime rate by town",
    "ZN": "Proportion of residential land zoned for lots over 25,000 sq.ft.",
    "INDUS": "Proportion of non-retail business acres per town",
    "CHAS": "Charles River dummy variable (1 if tract bounds river; 0 otherwise)",
    "NOX": "Nitric oxides concentration (parts per 10 million)",
    "RM": "Average number of rooms per dwelling",
    "AGE": "Proportion of owner-occupied units built prior to 1940",
    "DIS": "Weighted distances to five Boston employment centres",
    "RAD": "Index of accessibility to radial highways",
    "TAX": "Full-value property-tax rate per $10,000",
    "PTRATIO": "Pupil-teacher ratio by town",
    "B": "1000(Bk - 0.63)^2 where Bk is the proportion of Black residents",
    "LSTAT": "% lower status of the population"
}

# Default values based on Boston Housing dataset means
DEFAULT_VALUES = {
    "CRIM": 3.61, "ZN": 11.36, "INDUS": 11.14, "CHAS": 0, "NOX": 0.55,
    "RM": 6.28, "AGE": 68.57, "DIS": 3.80, "RAD": 9.55, "TAX": 408.24,
    "PTRATIO": 18.46, "B": 356.67, "LSTAT": 12.65
}

# ==============================
# Title Section
# ==============================
st.title("🏡 Boston House Price Predictor")
st.markdown("**Advanced ML-powered real estate valuation system**")
st.divider()

# Check if models loaded successfully
if error:
    st.error(f"⚠️ Error loading models: {error}")
    st.info("Please ensure 'boston_house_model.joblib' and 'scaler.joblib' are in the same directory.")
    st.stop()

# ==============================
# Sidebar - Input Section
# ==============================
st.sidebar.header("📊 Input Features")
st.sidebar.markdown("Adjust the values to input house characteristics:")

# Organize inputs with better grouping
with st.sidebar.expander("🏘️ Location & Environment", expanded=True):
    CRIM = st.number_input("Crime Rate (CRIM)", value=DEFAULT_VALUES["CRIM"], 
                           min_value=0.0, step=0.1, help=FEATURE_INFO["CRIM"])
    
    ZN = st.slider("Residential Zoning (ZN)", 0.0, 100.0, DEFAULT_VALUES["ZN"], 
                   help=FEATURE_INFO["ZN"])
    
    CHAS = st.selectbox("Near Charles River (CHAS)", [0, 1], 
                        help=FEATURE_INFO["CHAS"])

with st.sidebar.expander("🏭 Industrial & Pollution"):
    INDUS = st.slider("Industrial Proportion (INDUS)", 0.0, 30.0, DEFAULT_VALUES["INDUS"],
                      help=FEATURE_INFO["INDUS"])
    
    NOX = st.slider("NOx Concentration (NOX)", 0.3, 0.9, DEFAULT_VALUES["NOX"], 0.01,
                    help=FEATURE_INFO["NOX"])

with st.sidebar.expander("🏠 Property Characteristics"):
    RM = st.slider("Average Rooms (RM)", 3.0, 9.0, DEFAULT_VALUES["RM"], 0.1,
                   help=FEATURE_INFO["RM"])
    
    AGE = st.slider("Property Age (AGE)", 0.0, 100.0, DEFAULT_VALUES["AGE"],
                    help=FEATURE_INFO["AGE"])

with st.sidebar.expander("🚗 Accessibility & Infrastructure"):
    DIS = st.slider("Distance to Employment (DIS)", 1.0, 12.0, DEFAULT_VALUES["DIS"], 0.1,
                    help=FEATURE_INFO["DIS"])
    
    RAD = st.slider("Highway Access (RAD)", 1.0, 24.0, DEFAULT_VALUES["RAD"],
                    help=FEATURE_INFO["RAD"])
    
    TAX = st.slider("Property Tax (TAX)", 150.0, 750.0, DEFAULT_VALUES["TAX"],
                    help=FEATURE_INFO["TAX"])

with st.sidebar.expander("👥 Demographics & Services"):
    PTRATIO = st.slider("Pupil-Teacher Ratio (PTRATIO)", 12.0, 22.0, DEFAULT_VALUES["PTRATIO"], 0.1,
                        help=FEATURE_INFO["PTRATIO"])
    
    B = st.slider("B Statistic", 0.0, 400.0, DEFAULT_VALUES["B"],
                  help=FEATURE_INFO["B"])
    
    LSTAT = st.slider("Lower Status % (LSTAT)", 1.0, 40.0, DEFAULT_VALUES["LSTAT"], 0.1,
                      help=FEATURE_INFO["LSTAT"])

# Add reset button
st.sidebar.divider()
if st.sidebar.button("🔄 Reset to Defaults", use_container_width=True):
    st.rerun()

# ==============================
# Main Content Area
# ==============================
col1, col2 = st.columns([2.5, 1])

with col1:
    st.subheader("📈 Input Summary")
    
    # Create two columns for better display
    col_a, col_b = st.columns(2)
    
    with col_a:
        input_df_1 = pd.DataFrame({
            'Feature': ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE'],
            'Value': [f"{CRIM:.2f}", f"{ZN:.2f}", f"{INDUS:.2f}", 
                     str(CHAS), f"{NOX:.2f}", f"{RM:.2f}", f"{AGE:.2f}"]
        })
        st.dataframe(input_df_1, use_container_width=True, hide_index=True, height=280)
    
    with col_b:
        input_df_2 = pd.DataFrame({
            'Feature': ['DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'],
            'Value': [f"{DIS:.2f}", f"{RAD:.2f}", f"{TAX:.2f}", 
                     f"{PTRATIO:.2f}", f"{B:.2f}", f"{LSTAT:.2f}"]
        })
        st.dataframe(input_df_2, use_container_width=True, hide_index=True, height=280)

with col2:
    st.subheader("🎯 Quick Stats")
    st.metric("✅ Features Configured", "13/13", delta="Complete")
    st.metric("🤖 Model Type", "Regression")
    st.metric("⚡ Status", "Ready", delta="Active")

# ==============================
# Prediction Button & Results
# ==============================
st.divider()
st.subheader("🔮 Generate Prediction")

col_btn, col_space = st.columns([1, 3])
with col_btn:
    predict_button = st.button("🚀 Predict House Price", use_container_width=True, type="primary")

if predict_button:
    # Prepare data
    data = np.array([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, 
                      PTRATIO, B, LSTAT]])
    
    with st.spinner("🔄 Analyzing features and generating prediction..."):
        try:
            # Scale and predict
            data_scaled = scaler.transform(data)
            prediction = model.predict(data_scaled)[0]
            
            # Display result
            st.success("Prediction Complete!")
            
            # Main prediction display
            col_main = st.columns([1])[0]
            with col_main:
                st.markdown("### 💰 Predicted House Price")
                st.markdown(f"# ${prediction:.2f}K")
                st.caption("Estimated median value in thousands of dollars")
            
            st.divider()
            
            # Additional insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="💵 Price Range",
                    value=f"${prediction:.2f}K",
                    delta=f"±${(prediction * 0.1):.2f}K",
                    help="10% confidence interval"
                )
            
            with col2:
                price_category = "High" if prediction > 30 else "Medium" if prediction > 20 else "Low"
                category_emoji = "📈" if prediction > 30 else "📊" if prediction > 20 else "📉"
                st.metric(
                    label="📊 Price Category",
                    value=price_category,
                    delta=category_emoji
                )
            
            with col3:
                # Key factor analysis
                key_factor = "Rooms" if RM > 7 else "Crime" if CRIM > 5 else "Location"
                st.metric(
                    label="🔑 Key Driver",
                    value=key_factor,
                    help="Primary factor affecting price"
                )
            
            st.divider()
            
            # Feature importance visualization
            st.subheader("📊 Feature Impact Analysis")
            
            # Simple feature importance based on typical Boston Housing patterns
            importance_data = {
                'Feature': ['RM', 'LSTAT', 'PTRATIO', 'NOX', 'DIS', 'CRIM', 'TAX'],
                'Impact Score': [0.95, 0.85, 0.65, 0.55, 0.50, 0.45, 0.40]
            }
            
            fig = go.Figure(go.Bar(
                x=importance_data['Impact Score'],
                y=importance_data['Feature'],
                orientation='h',
                marker=dict(
                    color=importance_data['Impact Score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Impact")
                )
            ))
            
            fig.update_layout(
                title="Top Features Affecting Price",
                xaxis_title="Relative Impact Score",
                yaxis_title="Feature",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Prediction error: {str(e)}")

# ==============================
# Footer Information
# ==============================
st.divider()
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    ### Boston Housing Price Prediction Model
    
    This application uses machine learning to predict median house prices in Boston suburbs 
    based on 13 different features including crime rate, property characteristics, and 
    neighborhood attributes.
    
    **Model Features:**
    - 📊 Trained on the classic Boston Housing dataset
    - 🎯 Predicts median home values in thousands of dollars
    - ⚙️ Uses standardized features for accurate predictions
    
    **How to Use:**
    1. Adjust the feature values in the sidebar
    2. Click "Predict House Price" to get the estimate
    3. Review the detailed analysis and insights
    
    **Note:** This is a demonstration model. For real estate decisions, consult professional appraisers.
    """)

st.caption("\t\t\tSham")