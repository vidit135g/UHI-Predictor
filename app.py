import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ----------------- 🌙 Dark Mode Page Configuration -----------------
st.set_page_config(
    page_title="UHI Index Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)
file_path = "data/processed/UHI_Weather_Building_Sentinel_LST_Featured_Cleaned.csv"
df = pd.read_csv(file_path)
# ----------------- 🎨 Custom Dark Theme Styling -----------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');
        .poppins-thin {
            font-family: "Poppins", serif;
            font-weight: 100;
            font-style: normal;
        }
        html, body, [class*="css"] {
            font-family: 'poppins-thin', serif;
            font-weight:100;
            background: #121212;
            color: #e0e0e0;
        }
        .st-emotion-cache-1lnhyzl {
            background: linear-gradient(to right, #1e1e1e, #292929);
            padding: 15px;
            font-family: 'poppins-thin', serif;
            font-weight:100;
            border-radius: 8px;
            text-align: center;
            font-size: 18px;
            font-weight: bold;
        }
        .stButton>button {
            background: linear-gradient(45deg, #FF6B6B, #FFD166);
            color: white;
            font-size: 16px;
            font-family: 'poppins-thin', serif;
            font-weight:100;
            padding: 10px 24px;
            border-radius: 8px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #FF8E8E, #FFE599);
            font-family: 'poppins-thin', serif;
            font-weight:100;
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- 🏠 Sidebar Navigation -----------------
st.sidebar.title("🔍 Navigation")
nav_option = st.sidebar.radio("Go To:", ["🏠 Home", "📈 Predict UHI", "📊 Visualizations", "📂 Upload Dataset", "📖 Insights & Recommendations", "ℹ️ About"])

# ----------------- 📌 Load Models -----------------
@st.cache_resource
def load_models():
    return {
        "rf_model": joblib.load("models/rf_model.pkl"),
        "xgb_model": joblib.load("models/xgb_model.pkl"),
        "encoder": joblib.load("models/encoder.pkl"),
        "feature_names": joblib.load("models/feature_names.pkl")
    }

models = load_models()

# ----------------- 🏠 Home Page -----------------
if nav_option == "🏠 Home":
    file_path = "data/processed/UHI_Weather_Building_Sentinel_LST_Featured_Cleaned.csv"
    df = pd.read_csv(file_path)
    st.title("🏙️ Urban Heat Island (UHI) Index Predictor")
    st.markdown("**A Smart Approach to Understanding Urban Heat 🌎**")
    st.markdown("""
    - **🌡️ Predict UHI Index:** Enter parameters and get instant results.
    - **📊 Explore Visualizations:** Compare trends, patterns & heatmaps.
    - **📂 Upload Data:** Retrain models with custom datasets.
    - **📖 Learn More:** Insights on how UHI affects cities.
    """)
    st.image("assets/urban_heat.jpg", use_column_width=True)

    file_path = "data/processed/UHI_Weather_Building_Sentinel_LST_Featured_Cleaned.csv"
    df = pd.read_csv(file_path)

# ----------------- 🔍 Predict UHI -----------------
elif nav_option == "📈 Predict UHI":
    st.title("📌 Predict UHI Index")

    # 🌍 Satellite Data
    tab1, tab2, tab3 = st.tabs(["🌍 Satellite Data", "🌡️ Weather Factors", "🏢 Urban Features"])
    
    with tab1:
        st.subheader("🌍 Satellite Data")
        user_inputs = {
            'land_surface_temp': st.slider("🌡️ Land Surface Temp (°C)", 20.0, 50.0, 35.0, 0.1),
            'band1': st.slider("🌍 Band 1", 500, 2500, 1200),
            'band2': st.slider("🌍 Band 2", 500, 2500, 1100),
            'band3': st.slider("🌍 Band 3", 500, 2500, 1500),
            'band4': st.slider("🌍 Band 4", 500, 2500, 1800)
        }

    with tab2:
        st.subheader("🌡️ Weather Factors")
        user_inputs.update({
            'air_temp_at_surface_degc': st.slider("🌡️ Air Temp (°C)", 15.0, 40.0, 30.0, 0.1),
            'relative_humidity_percent': st.slider("💧 Relative Humidity (%)", 0, 100, 60),
            'avg_wind_speed_m_s': st.slider("💨 Avg Wind Speed (m/s)", 0.0, 10.0, 3.2, 0.1),
            'wind_direction_degrees': st.slider("🧭 Wind Direction (°)", 0, 360, 180),
            'solar_flux_w_m2': st.slider("☀️ Solar Flux (W/m²)", 200, 800, 500)
        })

    with tab3:
        st.subheader("🏢 Urban Features")
        user_inputs.update({
            'building_distance_m': st.slider("🏢 Building Distance (m)", 0, 500, 50),
            'building_density_50m': st.slider("🏠 Building Density (50m)", 0, 100, 5),
            'building_density_100m': st.slider("🏠 Building Density (100m)", 0, 100, 10),
            'building_density_200m': st.slider("🏠 Building Density (200m)", 0, 100, 20)
        })

    # ✅ Save user inputs to session state
    st.session_state["user_inputs"] = user_inputs

    df_input = pd.DataFrame([user_inputs])
    
    # 🔥 Fix Feature Name Mismatch
    for col in models["feature_names"]:
        if col not in df_input.columns:
            df_input[col] = 0  # Fill missing features with 0

    df_input = df_input[models["feature_names"]]  # Align with trained model

    # 🔥 Predict Button
    if st.button("🔍 Predict UHI Index"):
        try:
            uhi_rf = models["rf_model"].predict(df_input)[0]
            uhi_xgb = models["xgb_model"].predict(df_input)[0]

            st.success("✅ Prediction Successful!")
            st.metric(label="📌 **Predicted UHI Index (Random Forest)**", value=f"{uhi_rf:.6f}")
            st.metric(label="📌 **Predicted UHI Index (XGBoost)**", value=f"{uhi_xgb:.6f}")

            # ✅ Save prediction in session state
            st.session_state["uhi_value"] = uhi_rf  # Save RF model's UHI prediction for insights

        except ValueError as e:
            st.error(f"⚠️ Prediction Error: {e}")
# ----------------- 📊 Visualizations -----------------
# ----------------- 📊 Visualizations -----------------
elif nav_option == "📊 Visualizations":
    st.title("📊 UHI Data Visualizations")

    # Load dataset
    file_path = "data/processed/UHI_Weather_Building_Sentinel_LST_Featured_Cleaned.csv"
    df = pd.read_csv(file_path)

    # ✅ Drop non-numeric columns
    non_numeric_cols = ["datetime", "hour_category", "wind_direction_category"]
    df_numeric = df.drop(columns=[col for col in non_numeric_cols if col in df.columns], errors="ignore")

    # 🌡️ **1️⃣ Heatmap of UHI vs. Features**
    st.subheader("🌡️ Heatmap of UHI vs. Features")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    # 📈 **2️⃣ Scatter Plot: Land Surface Temp vs. UHI Index**
    st.subheader("📈 Scatter Plot: Land Surface Temp vs. UHI Index")
    fig = px.scatter(df_numeric, x="land_surface_temp", y="uhi_index", color="uhi_index",
                     labels={"uhi_index": "UHI Index", "land_surface_temp": "Land Surface Temperature (°C)"},
                     title="📌 Relationship Between UHI Index and Surface Temperature",
                     template="plotly_dark")
    st.plotly_chart(fig)

    # 📊 **3️⃣ Histogram: UHI Index Distribution**
    st.subheader("📊 Histogram: UHI Index Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df["uhi_index"], bins=30, kde=True, color="blue", ax=ax)
    ax.set_title("Distribution of UHI Index")
    st.pyplot(fig)

    # 📊 **4️⃣ Box Plot: UHI Index by Time of Day**
    st.subheader("📊 Box Plot: UHI Index by Time of Day")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(x=df["hour_category"], y=df["uhi_index"], palette="coolwarm", ax=ax)
    ax.set_title("UHI Index Variation Across Different Times of the Day")
    st.pyplot(fig)

    # 📉 **5️⃣ Pairplot: Key Features vs. UHI Index**
    st.subheader("📉 Pair Plot: Key Features vs. UHI Index")
    selected_features = ["uhi_index", "land_surface_temp", "air_temp_at_surface_degc", "humidity_temp_interaction"]
    fig = sns.pairplot(df[selected_features], diag_kind="kde", plot_kws={"alpha": 0.5})
    st.pyplot(fig)

   

# ----------------- 📂 Upload Dataset -----------------
elif nav_option == "📂 Upload Dataset":
    file_path = "data/processed/UHI_Weather_Building_Sentinel_LST_Featured_Cleaned.csv"
    df = pd.read_csv(file_path)
    st.title("📂 Upload New Dataset")
    uploaded_file = st.file_uploader("📌 Upload a CSV File", type=["csv"])
    if uploaded_file:
        file_path = "data/processed/UHI_Weather_Building_Sentinel_LST_Featured_Cleaned.csv"
        df = pd.read_csv(file_path)
        df_uploaded = pd.read_csv(uploaded_file)
        st.success(f"✅ File Uploaded! Shape: {df_uploaded.shape}")
        st.dataframe(df_uploaded.head())

# 📖 Insights & Recommendations
# 📖 Insights & Recommendations
elif nav_option == "📖 Insights & Recommendations":
    st.title("📖 Real-Time Insights & Recommendations")

    # ✅ Check if predictions exist
    if "user_inputs" not in st.session_state or "uhi_value" not in st.session_state:
        st.warning("⚠️ No prediction found! Please run the 'Predict UHI' tab first.")
        st.stop()

    user_inputs = st.session_state["user_inputs"]
    uhi_value = st.session_state["uhi_value"]

    # ✅ Drop non-numeric columns before calculating correlation
    non_numeric_cols = ["datetime", "hour_category", "wind_direction_category"]
    df_numeric = df.drop(columns=[col for col in non_numeric_cols if col in df.columns], errors="ignore")

    # 🌡️ Correlation with UHI
    st.subheader("📊 Correlation of Features with UHI Index")
    correlation = df_numeric.corr()["uhi_index"].sort_values(ascending=False)
    st.bar_chart(correlation)

    # 📌 Key Insights
    st.subheader(f"🔍 UHI Index: **{uhi_value:.4f}** - Interpretation")

    if uhi_value < 0.3:
        st.success("🌿 Low UHI Impact: Favorable environmental conditions with balanced urban planning.")
        st.markdown("""
        - ✅ **Efficient Urban Cooling Strategies**
        - ✅ **Adequate vegetation & water bodies**
        - ✅ **Reflective surfaces and cool roofs**
        """)
    elif 0.3 <= uhi_value < 0.6:
        st.warning("🌆 Moderate UHI Impact: Consider adding more green spaces & reflective surfaces.")
        st.markdown("""
        - ⚠️ **Increasing tree cover and green roofs**
        - ⚠️ **Using heat-resistant building materials**
        - ⚠️ **Encouraging sustainable urban growth**
        """)
    else:
        st.error("🔥 High UHI Impact: Significant heat absorption, increase urban vegetation & cooling strategies.")
        st.markdown("""
        - 🚨 **High risk of heat stress & energy demand surge**
        - 🚨 **Need for strict climate-adaptive policies**
        - 🚨 **Prioritize heat mitigation strategies**
        """)

    # 🌍 Recommended Actions for Improvement
    st.subheader("📌 Recommended Actions for Mitigation")
    mitigation_tips = {
        "Increase Green Cover": "Plant more trees, vertical gardens, and rooftop greenery.",
        "Improve Building Materials": "Use reflective surfaces, light-colored rooftops, and heat-resistant coatings.",
        "Enhance Water Bodies": "Promote urban lakes, fountains, and water-based cooling solutions.",
        "Implement Smart City Policies": "Encourage climate-responsive infrastructure and urban planning."
    }

    for key, tip in mitigation_tips.items():
        st.markdown(f"✅ **{key}** - {tip}")

    st.markdown("---")
    st.markdown("🚀 **Smarter Cities, Cooler Environments, Sustainable Growth**")

    for key, tip in mitigation_tips.items():
        st.markdown(f"✅ **{key}** - {tip}")

    st.markdown("---")
    st.markdown("🚀 **Smarter Cities, Cooler Environments, Sustainable Growth**")

# ----------------- ℹ️ About Page -----------------

elif nav_option == "ℹ️ About":
    file_path = "data/processed/UHI_Weather_Building_Sentinel_LST_Featured_Cleaned.csv"
    df = pd.read_csv(file_path)
    st.title("ℹ️ About This App")
    st.markdown("🌎 A powerful tool for analyzing Urban Heat Islands.")

file_path = "data/processed/UHI_Weather_Building_Sentinel_LST_Featured_Cleaned.csv"
df = pd.read_csv(file_path)
st.markdown("---")
st.markdown("🚀 Built for Smart Urban Planning | Powered by ML")