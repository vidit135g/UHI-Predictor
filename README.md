# Urban Heat Island Prediction

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

## Project Overview
Predicting urban heat island hotspots using machine learning and geospatial analysis

## Features
- Spatial feature engineering
- Interpretable ML models
- SHAP value analysis
- Interactive visualizations

# UHI-Predictor

UHI-Predictor is a data science project developed for the EY Open Science AI & Data Challenge. It aims to predict Urban Heat Island (UHI) effects using multiple datasets—including ground temperature, weather, building footprint, and satellite data. The project combines exploratory data analysis (EDA), feature engineering, machine learning model training (using XGBoost), and a prediction pipeline. Although the current model was trained on New York City data, the framework can be adapted for other cities (e.g., testing for Lucknow).

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Data Sources and Licensing](#data-sources-and-licensing)
- [Installation](#installation)
- [Usage](#usage)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Feature Engineering](#feature-engineering)
  - [Model Training](#model-training)
  - [Making Predictions](#making-predictions)
  - [Testing for a Different Location](#testing-for-a-different-location)
- [Optional: Model Deployment](#optional-model-deployment)
- [Future Work](#future-work)
- [References and Credits](#references-and-credits)
- [License](#license)

---

## Project Overview

The goal of UHI-Predictor is to develop a data-driven model that can predict the UHI Index—an indicator of how much hotter a given urban area is relative to the surrounding rural areas. The project leverages the following datasets:

- **Ground Temperature Data (CSV):** Contains columns for Longitude, Latitude, datetime, and UHI Index.
- **Weather Data (Excel):** Contains separate worksheets for two locations (Bronx and Manhattan) with metadata such as temperature readings and timestamps.
- **Building Footprint Data (KML):** Provides spatial information about building footprints in the target area.
- **Satellite Data (Notebooks):** Sample processing of Sentinel-2 and Landsat data for additional remote sensing features.

The project follows these high-level steps:
1. **EDA:** Load and visualize raw datasets.
2. **Feature Engineering:** Merge and process datasets; extract numerical features (e.g., hour, day, month, weekday) from datetime fields.
3. **Model Training:** Train a baseline XGBoost model using the engineered features.
4. **Prediction:** Load the trained model and perform predictions on new input data.
5. **Optional Deployment:** Deploy the model as an API using FastAPI.

---

## Repository Structure


---

## Data Sources and Licensing

**Ground Temperature/UHI Data (CSV):**  
- **Source:** CAPA Strategies / Center for Open Science  
- **License/Terms:** [COS Terms of Use](https://github.com/CenterForOpenScience/cos.io/blob/master/TERMS_OF_USE.md) and [Apache 2.0 License](https://github.com/CenterForOpenScience/cos.io/blob/master/LICENSE)

**Weather Data (Excel):**  
- **Source:** New York State Mesonet  
- **License/Terms:** [Mesonet Data Terms](https://nysmesonet.org/about/data) and [NYS Mesonet Data Access Policy](https://nysmesonet.org/documents/NYS_Mesonet_Data_Access_Policy.pdf)

**Building Footprint Data (KML):**  
- **Source:** NYC Office of Technology and Innovation (NYC Open Data)  
- **License/Terms:** [NYC Data Terms](https://www.nyc.gov/html/data/terms.html) and [Apache 2.0 License for NYC Geo Data](https://github.com/CityOfNewYork/nyc-geo-metadata#Apache-2.0-1-ov-file)

**Satellite Data (Notebooks):**  
- **Sentinel-2:** [Sentinel Data Legal Notice](https://sentinel.esa.int/documents/247904/690755/Sentinel_Data_Legal_Notice) and [CC BY-SA 3.0 IGO License](https://creativecommons.org/licenses/by-sa/3.0/igo/)  
- **Landsat:** Typically public domain (NASA/USGS)

*Make sure to include proper attribution when publishing or redistributing your work.*

---

## Installation

### Prerequisites

- Python 3.13 (or later)
- [Homebrew](https://brew.sh/) (for macOS users, to install additional libraries like libomp if needed)
- Virtual environment tool (venv)

### Steps

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/UHI-Predictor.git
   cd UHI-Predictor

# UHI Predictor - README

## 📌 Project Overview
The **UHI Predictor** is a machine learning model designed to predict the Urban Heat Island (UHI) Index based on various meteorological and geographical parameters. This project has undergone significant improvements to enhance accuracy, compliance, and usability.

---

## 🚀 Key Improvements & Enhancements
### ✅ **Feature Engineering Refinements**
- **Removed Non-Compliant Features**: Longitude and Latitude were eliminated as predictive features to ensure model adaptability across different regions.
- **New Predictive Features Added**:
  - **Air Temperature at Surface (°C)**
  - **Relative Humidity (%)**
  - **Average Wind Speed (m/s)**
  - **Wind Direction (°)**
  - **Solar Flux (W/m²)**
  - **Altitude (m)**
  - **Hour of the Day**
  - **Day of the Month**
  - **Month of the Year**
  - **Weekday Indicator**

### 🎨 **Revamped Web Application UI**
- **Modern Material Design with Pastel Colors**
- **Improved User Experience**:
  - Animated form fields
  - Vibrant buttons with hover effects
  - Background images for enhanced aesthetics
  - Dynamic tooltips explaining each input field
- **Prediction Scale Information**:
  - A floating hover popup provides insights into the UHI Index scale.
  - Explains what low, moderate, and high UHI Index values indicate.

### 📊 **Model Enhancements**
- **Optimized XGBoost Model**:
  - Enabled categorical support for features requiring encoding.
  - Feature selection process refined to exclude non-numeric columns.
- **Dataset Validation**:
  - The training pipeline now checks for missing or invalid values before processing.
  - Enhanced error handling prevents crashes due to incorrect data formats.

### 🛠 **Technical Fixes & Compliance**
- **Resolved Feature Name Mismatches**:
  - Ensured input data matches the expected model features.
  - Implemented data transformation before passing inputs to the model.
- **Handled Missing Data in Feature Engineering**:
  - Implemented forward-fill and backward-fill techniques where applicable.
- **Optimized Data Processing Pipelines**:
  - Improved efficiency of merging datasets (Weather, Building Footprints, UHI Data).
  - Ensured proper datetime format handling across datasets.

---

## 🏗️ How to Run the Project
### **1️⃣ Set Up Environment**
```sh
pip install -r requirements.txt
```

### **2️⃣ Train the Model**
```sh
python models/train.py
```

### **3️⃣ Start the Web Application**
```sh
uvicorn app:app --reload
```

### **4️⃣ Access the Web App**
Open **http://127.0.0.1:8000/** in your browser.

---

## ✅ Compliance with Requirements
| Compliance Aspect | Status |
|-------------------|--------|
| Longitude & Latitude removed from ML model | ✅ Done |
| New weather and environmental features included | ✅ Done |
| Pastel-themed web UI with improved UX | ✅ Done |
| Detailed prediction scale info | ✅ Done |
| Data validation for missing/null values | ✅ Done |
| Model retraining & validation checks | ✅ Done |

---

## 📝 Next Steps & Future Enhancements
- **Improve Feature Selection**: Explore additional environmental factors.
- **Deploy to Cloud**: Deploy the model via AWS/GCP for wider accessibility.
- **Live Data Integration**: Fetch real-time weather data for better predictions.

📌 **Final Note**: This version of UHI Predictor ensures compliance with project rules while improving user experience and model accuracy. 🚀

