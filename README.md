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

