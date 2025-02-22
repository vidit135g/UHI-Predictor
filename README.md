<html>
<head>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    body {
      font-family: 'Poppins', sans-serif;
    }
  </style>
</head>
<body>
# 🌆 UHI Index Predictor  
![Urban Heat](assets/urban_heat.jpg)  

🔍 **An advanced AI-powered application for predicting Urban Heat Island (UHI) intensity using machine learning models.**  

---

## 🎥 Live Demo  

### 📌 Watch the Full Demo Here:  
Watch the Video: (https://www.youtube.com/watch?v=2-cxQMd-Jrc)  
---

## 📖 Background & Purpose  
Urban Heat Islands (UHI) are regions within cities that experience significantly higher temperatures than surrounding rural areas due to factors such as dense infrastructure, limited vegetation, and anthropogenic heat sources.  

The **UHI Index Predictor** is designed to:  
- **Analyze temperature variations** across urban landscapes.  
- **Predict UHI index** based on environmental, meteorological, and urban features.  
- **Assist policymakers** in designing climate-resilient cities.  
- **Enable real-time insights** for urban planners, sustainability experts, and researchers.  

---

## 🚀 Key Features  
✅ **Accurate UHI Predictions** using advanced ML models (Random Forest & XGBoost).  
✅ **Real-Time Analysis** with customizable environmental inputs.  
✅ **Interactive Visualizations** such as heatmaps, scatter plots, and histograms.  
✅ **Dataset Upload & Processing** for batch predictions.  
✅ **Eco-Friendly UI** with a **greenish, sustainable theme**.  
✅ **Insights & Recommendations** based on **real-time data trends**.  

---

## 📊 Data Used for Model Training  
This project utilizes **real-world near-surface air temperature data**, collected on **July 24, 2021**, covering **the Bronx and Manhattan regions of NYC**.  

**📌 Dataset Includes:**  
- **Geospatial Features:** Latitude, Longitude  
- **Meteorological Factors:** Air Temperature, Relative Humidity, Wind Speed, Wind Direction, Solar Flux  
- **Urban Features:** Building Density (50m, 100m, 200m), Distance to Nearest Building  
- **Satellite Observations:** NDVI, NDBI, Land Surface Temperature (LST)  
- **Temporal Data:** Hour, Day, Month, Weekday/Weekend Categorization  
- **Target Variable:** UHI Index  

---

## 🏗 Technology Stack  
| **Category**          | **Libraries Used** |
|----------------------|-------------------|
| **Web Framework**    | Streamlit |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Data Processing**  | Pandas, NumPy, SciPy |
| **Visualization**    | Matplotlib, Seaborn, Plotly |
| **Geospatial Analysis** | Haversine, cKDTree |

---

## 🔥 Machine Learning Models Used  
### ✅ 1️⃣ **Random Forest Regressor**  
- Trained with **200 estimators** for robust prediction.  
- Handles **non-linear relationships** and provides **high accuracy**.  
- Outputs feature importance scores for **key factor analysis**.  

### ✅ 2️⃣ **XGBoost Regressor**  
- Optimized using **GridSearchCV** for hyperparameter tuning.  
- High efficiency, **fast execution**, and **low error margin**.  
- Handles **missing values** and **multi-dimensional data** effectively.  

Both models are trained and **compared using MAE, RMSE, and R² scores** for optimal selection.  

---

## 📌 How to Run the Application  

### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/your-username/UHI-Predictor.git
cd UHI-Predictor
</body>
</html>