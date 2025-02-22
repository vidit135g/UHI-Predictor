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

# ----------------- 🎨 Custom Styling -----------------
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@100;200;300;400;500;600;700&display=swap');

    html, body, [class*="st-"], [class*="css-"] {
        font-family: 'Poppins', sans-serif !important;
    }

    .stApp {
        font-family: 'Poppins', sans-serif !important;
    }

    h1, h2, h3, h4, h5, h6, p, div {
        font-family: 'Poppins', sans-serif !important;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)


# ----------------- 🏠 Sidebar Navigation -----------------
st.sidebar.title("🔍 Navigation")
nav_option = st.sidebar.radio("Go To:", ["🏠 Home", "📈 Predict UHI", "📊 Visualizations", "📖 Insights", "ℹ️ About"])

# ----------------- 📌 Load Models -----------------
@st.cache_resource
def load_models():
    model_files = {
        "rf_model": "models/rf_model.pkl",
        "xgb_model": "models/xgb_model.pkl",
        "encoder": "models/encoder.pkl",
        "feature_names": "models/feature_names.pkl"
    }

    for key, path in model_files.items():
        if not os.path.exists(path):
            st.error(f"❌ Missing model file: {path}")
            raise FileNotFoundError(f"File not found: {path}")

    return {key: joblib.load(path) for key, path in model_files.items()}

models = load_models()

# ----------------- 🏠 Home Page -----------------
if nav_option == "🏠 Home":
    st.title("🏙️ Urban Heat Island (UHI) Index Predictor")
    st.markdown("**A Smart Approach to Understanding Urban Heat 🌎**")
    st.image("assets/urban_heat.jpg", use_container_width=True)

# ----------------- 🔍 Predict UHI -----------------
elif nav_option == "📈 Predict UHI":
    st.title("📌 Predict UHI Index")

    # ✅ Custom Slider Styling (Green Theme)
    st.markdown("""
    <style>
        div[data-baseweb="slider"] > div > div {
            background: linear-gradient(90deg, #F1D7BB, #F1D7BB) !important;
            height: 5px !important;
            border-radius: 10px;
        }
        div[data-baseweb="slider"] > div > div > div {
            background: #129A7D !important;
            width: 25px !important;
            height: 25px !important;
            border-radius: 50% !important;
            border: 2px solid white !important;
        }
        div[data-baseweb="slider"] > div > div > div:hover {
            background: #9BD3CB !important;
            transform: scale(1.5);
        }
    </style>
    """, unsafe_allow_html=True)

    # 🧩 Tabs for Inputs
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
            'building_density_50m': st.slider("🏠 Building Density (50m)", 0, 100, 5),
            'building_density_100m': st.slider("🏠 Building Density (100m)", 0, 100, 10),
            'building_density_200m': st.slider("🏠 Building Density (200m)", 0, 100, 20)
        })

    # ✅ Data Formatting & Feature Matching
    df_input = pd.DataFrame([user_inputs])
    df_input.columns = df_input.columns.str.replace(" ", "_").str.lower()

    # 🔍 Fix Feature Name Case Sensitivity
    model_features = [col for col in models["feature_names"]]
    feature_mapping = {df_input_col.lower(): df_input_col for df_input_col in df_input.columns}
    df_input = df_input.rename(columns={k: v for k, v in feature_mapping.items() if v in model_features})

    # 🔥 Add Missing Features with Default Values
    for col in model_features:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[model_features]

    # 🔥 Predict Button
    if st.button("🔍 Predict UHI Index"):
        try:
            uhi_rf = models["rf_model"].predict(df_input)[0]
            uhi_xgb = models["xgb_model"].predict(df_input)[0]

            st.success("✅ Prediction Successful!")
            st.metric(label="📌 **Predicted UHI Index (Random Forest)**", value=f"{uhi_rf:.6f}")
            st.metric(label="📌 **Predicted UHI Index (XGBoost)**", value=f"{uhi_xgb:.6f}")

            # ✅ Store predictions in session state (Ensure these lines exist)
            st.session_state["uhi_value"] = uhi_rf  # Save RF model's prediction
            st.session_state["user_inputs"] = user_inputs  # Save user inputs

        except ValueError as e:
            st.error(f"⚠️ Prediction Error: {e}")

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

    # 🌡️ **1️⃣ Heatmap of UHI vs. Features**
# 📈 **2️⃣ Dynamic Scatter Plot**
    st.subheader("📈 Relationship Between Features")

    x_feature = st.selectbox("📌 Select X-Axis Feature", df_numeric.columns, index=list(df_numeric.columns).index("land_surface_temp"))
    y_feature = st.selectbox("📌 Select Y-Axis Feature", df_numeric.columns, index=list(df_numeric.columns).index("uhi_index"))

    fig = px.scatter(
        df_numeric, x=x_feature, y=y_feature, trendline="ols",
        labels={x_feature: x_feature, y_feature: y_feature},
        title=f"📌 Relationship Between {x_feature} and {y_feature}"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 🌡️ **1️⃣ Correlation Heatmap with Adjustable Features**
    st.subheader("🌡️ Correlation Heatmap with Selectable Features")
    
    selected_features = st.multiselect(
        "📌 Select Features for Correlation Heatmap",
        options=df_numeric.columns.tolist(),
        default=["uhi_index", "land_surface_temp", "ndvi", "ndbi", "air_temp_at_surface_"]
    )

    correlation_threshold = st.slider("📏 Set Minimum Correlation Threshold", 0.0, 1.0, 0.3, 0.05)

    if selected_features:
        filtered_corr = df_numeric[selected_features].corr()
        filtered_corr = filtered_corr[abs(filtered_corr) > correlation_threshold]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(filtered_corr, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
        st.pyplot(fig)

    # 📈 **2️⃣ Pairplot with Selectable Features**
    st.subheader("📈 Pairplot: Explore Relationships")
    
    pairplot_features = st.multiselect(
        "📌 Select Features for Pairplot",
        options=df_numeric.columns.tolist(),
        default=["uhi_index", "land_surface_temp", "ndvi", "ndbi"]
    )

    if len(pairplot_features) >= 2:
        fig = sns.pairplot(df_numeric[pairplot_features], diag_kind="kde", plot_kws={"alpha": 0.6})
        st.pyplot(fig)
    else:
        st.warning("⚠️ Select at least 2 features for the pairplot.")

    # 🌍 **3️⃣ Interactive 3D Scatter Plot**
    st.subheader("🌍 3D Scatter Plot")

    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis = st.selectbox("📌 Select X-Axis", df_numeric.columns.tolist(), index=df_numeric.columns.get_loc("land_surface_temp"))
    with col2:
        y_axis = st.selectbox("📌 Select Y-Axis", df_numeric.columns.tolist(), index=df_numeric.columns.get_loc("uhi_index"))
    with col3:
        z_axis = st.selectbox("📌 Select Z-Axis", df_numeric.columns.tolist(), index=df_numeric.columns.get_loc("ndvi"))

    fig = px.scatter_3d(df_numeric, x=x_axis, y=y_axis, z=z_axis, color="uhi_index",
                         title="3D Visualization of UHI Factors", template="plotly_dark")
    st.plotly_chart(fig)

   # 🌡️ **1️⃣ Interactive Correlation Heatmap**
    st.subheader("🌡️ Interactive Correlation Heatmap")
    corr_matrix = df_numeric.corr()

    fig = px.imshow(
        corr_matrix,
        labels=dict(x="Features", y="Features", color="Correlation"),
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        color_continuous_scale="RdBu",
        title="📌 Feature Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

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

# ----------------- 📖 Insights & Recommendations -----------------
elif nav_option == "📖 Insights":
    st.title("📖 Real-Time Insights & Recommendations")

    # ✅ Check if session state has predictions stored
    if "uhi_value" not in st.session_state:
        st.warning("⚠️ No prediction found! Please run the 'Predict UHI' tab first.")
        st.stop()

    # ✅ Retrieve stored prediction
    uhi_value = st.session_state["uhi_value"]

    # ✅ Reload dataset for insights
    file_path = "data/processed/UHI_Weather_Building_Sentinel_LST_Featured_Cleaned.csv"
    df = pd.read_csv(file_path)

    non_numeric_cols = ["datetime", "hour_category", "wind_direction_category"]
    df_numeric = df.drop(columns=[col for col in non_numeric_cols if col in df.columns], errors="ignore")

    # 🌡️ Correlation with UHI
    st.subheader("📊 Correlation of Features with UHI Index")

    correlation = df_numeric.corr()["uhi_index"].sort_values(ascending=False)

    fig = px.bar(
        correlation, 
        x=correlation.index, 
        y=correlation.values,
        text=correlation.values.round(2),
        labels={"x": "Features", "y": "Correlation"},
        title="📌 Feature Correlation with UHI Index",
        color=correlation.values, 
        color_continuous_scale="RdBu"
    )

    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

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
# ----------------- ℹ️ About Page -----------------
elif nav_option == "ℹ️ About":
    st.title("ℹ️ About This App")
    
    st.markdown("""
        ## 🌎 Urban Heat Island (UHI) Index Predictor  
        This web application predicts **Urban Heat Island (UHI) Index** values based on environmental, meteorological, and urban landscape data.  
        UHI is a critical environmental phenomenon where urban areas experience significantly **higher temperatures** than surrounding rural regions due to factors like infrastructure, land usage, and population density.  

        This tool helps urban planners, researchers, and environmentalists **analyze** and **mitigate** UHI effects by providing data-driven insights.
    """)

    st.markdown("---")

    st.subheader("🔍 Why Was This App Made?")
    st.markdown("""
    - **📈 Predict UHI Index** → Uses **machine learning models** to forecast UHI intensity at different locations.  
    - **📊 Identify Key Contributors** → Highlights **factors** influencing urban heat hotspots.  
    - **🏙️ Support Smart Urban Planning** → Helps policymakers **make informed decisions** on mitigating urban heat.  
    - **📉 Provide Interactive Visualizations** → Showcases data trends through **heatmaps, scatter plots, and insights**.  
    """)

    st.markdown("---")

    st.subheader("📌 Machine Learning Models Used")
    st.markdown("""
    - 🔹 **Random Forest Regressor** → A tree-based ensemble model, effective for handling complex relationships.  
    - 🔹 **XGBoost Regressor** → A high-performance boosting algorithm optimized for predictive accuracy.  
    """)

    st.markdown("---")

    st.subheader("📦 Libraries & Technologies Used")
    st.markdown("""
    - 🔹 **Machine Learning** → `scikit-learn`, `XGBoost`, `joblib`  
    - 🔹 **Data Processing** → `pandas`, `numpy`, `seaborn`  
    - 🔹 **Visualization** → `matplotlib`, `plotly`, `seaborn`  
    - 🔹 **Web App Development** → `streamlit`  
    """)

    st.markdown("---")

    st.subheader("👨‍💻 Developed By")
    st.markdown("""
    💡 **Vidit Gupta**  
    A passionate **Cybersecurity Engineer & Cloud Security Consultant** with expertise in **Cyber Risk, Cloud Security, DevSecOps, and Compliance Audits**.  
    🚀 Dedicated to **ensuring security-first digital transformations**.
    """)

    st.markdown("---")
    st.markdown("📌 **Built for Smart Urban Planning | Powered by Machine Learning** 🚀")