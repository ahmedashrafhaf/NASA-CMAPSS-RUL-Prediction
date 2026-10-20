# BEFORE ZERO

### NASA C-MAPSS — Turbofan Engine Degradation Prognostics & Predictive Maintenance

## Overview

BEFORE ZERO is a predictive maintenance project that uses NASA C-MAPSS engine sensor data to estimate Remaining Useful Life (RUL) and support proactive maintenance decisions.

The goal is to identify engine degradation patterns and predict potential failures before they occur.

## Objectives

* Analyze turbofan engine sensor data.
* Prepare data for machine learning.
* Identify informative sensors and address multicollinearity.
* Build predictive models for Remaining Useful Life (RUL).
* Present results through an interactive dashboard.

## Methodology

* **Data Preprocessing:** Prepare the engine sensor data.
* **Feature Selection:** Remove uninformative sensors.
* **Multicollinearity Analysis:** Use Variance Inflation Factor (VIF).
* **Model Development:** Train a Random Forest model with 250 trees.
* **Evaluation:** Evaluate model performance using regression metrics.
* **Visualization:** Use Plotly and Streamlit to present insights.

## Results

Reported project results:

* **MAE:** 27.40 cycles
* **RMSE:** 36.95 cycles
* **Health Condition Classification Accuracy:** 84.07%

*Results are based on the project's recorded outputs.*

## Tech Stack

* Python
* Pandas & NumPy
* Scikit-learn
* Plotly
* Streamlit

## Dataset

The project uses NASA's Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) dataset.

Dataset source: [NASA Prognostics Center of Excellence](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/)

## Project Structure

```text
NASA-CMAPSS-RUL-Prediction/
├── ml/
│   ├── nasa.py
│   └── readme.txt
├── README.md
└── .gitignore
```

*Update the structure above to match the actual files in your repository.*

## Future Improvements

* Compare additional machine learning models.
* Improve RUL prediction performance.
* Enhance dashboard functionality.

## Author

Ahmed Ashraf Hafez
