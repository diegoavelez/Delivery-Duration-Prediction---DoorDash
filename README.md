# Delivery Duration Prediction for DoorDash

This project aims to predict the delivery duration for DoorDash orders using historical data. Accurate prediction of delivery times is critical to enhancing customer satisfaction and optimizing logistics. This repository contains all the necessary files, scripts, and documentation to reproduce the analysis and models used in this project.

## Table of Contents
- [Project Overview](#project-overview)
- [Data](#data)
- [Methodology](#methodology)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Conclusion](#conclusion)
- [Future Work](#future-work)

---

## Project Overview

This project was developed as part of a take-home assignment for a Data Science role. The objective is to build a model that accurately predicts the delivery duration (in seconds) for DoorDash orders, from the time the order is placed to the time it is delivered.

Two main modeling approaches were explored:
1. **Machine Learning (XGBoost)**: An optimized XGBoost model was developed, which achieved the best performance.
2. **Deep Learning (Neural Network)**: A deep learning model with an optimized architecture was also developed, achieving reasonable performance but slightly below XGBoost.

---

## Data

The dataset provided includes various features related to:
- **Time Features**: Timestamps for order creation and delivery.
- **Order Details**: Number of items, distinct items, and order value.
- **Store Characteristics**: Cuisine type, protocol used for order processing.
- **Market Dynamics**: Number of active dashers, busy dashers, and outstanding orders near the time of order.
- **Predicted Durations**: Estimated times for different stages of delivery, which serve as useful baseline features.

> Note: Some features had outliers, missing values, and required transformation. Outlier handling, one-hot encoding, and feature engineering were performed.

---

## Methodology

### 1. **Exploratory Data Analysis (EDA)**
   - **Distribution Analysis**: Examined the distribution of key features, especially delivery duration.
   - **Outlier Handling**: Marked or capped outliers for important numeric features using IQR methodology.
   - **Feature Engineering**: Created additional features such as `busy_dashers_per_order`, `workload_ratio`, and indicators for rush hours (lunch, dinner).
   - **Encoding**: Categorical features like `store_primary_category` were one-hot encoded for model compatibility.

### 2. **Modeling**
   - **XGBoost Model**: Tuned using GridSearchCV, achieving the best performance with an MAE of approximately 525 seconds.
   - **Deep Learning Model**: Built with Keras, utilizing batch normalization and dropout to reduce overfitting. Achieved an MAE of approximately 610 seconds.

### 3. **Evaluation Metrics**
   - **Mean Absolute Error (MAE)**: Used as the primary metric to evaluate model performance.
   - **Root Mean Squared Error (RMSE)**: Secondary metric to assess prediction stability.

---

## Results

- **XGBoost Model**:
  - **MAE**: 525.25 seconds
  - **RMSE**: 679.89 seconds

- **Deep Learning Model**:
  - **MAE**: 610.14 seconds
  - **RMSE**: ~780 seconds
  
  The XGBoost model provided more accurate predictions and was chosen as the final model for deployment due to its lower error metrics.

---

## Requirements

To reproduce the analysis, install the required libraries:

```bash
pip install -r requirements.txt

The main packages include:

	•	pandas, numpy, scikit-learn: Data manipulation and machine learning
	•	xgboost: For the XGBoost model
	•	tensorflow or keras: For the deep learning model
	•	matplotlib, seaborn: Visualization

Usage

1. Clone the Repository

git clone https://github.com/your-username/delivery-duration-prediction.git
cd delivery-duration-prediction

2. Run the Notebooks

	•	01_EDA.ipynb: Exploratory Data Analysis and feature engineering.
	•	02_Preprocessing.ipynb: Preprocess data and save the final dataset.
	•	03_Modeling_XGBoost.ipynb: Train and evaluate the XGBoost model.
	•	04_Modeling_DeepLearning.ipynb: Train and evaluate the deep learning model.

Each notebook is self-contained and can be executed in sequence. Ensure the preprocessed data file processed_data_final.csv is available before running the modeling notebooks.

Repository Structure

.
├── data/
│   ├── historical_data.csv               # Original dataset
│   ├── processed_data_final.csv           # Final processed dataset
├── notebooks/
│   ├── 01_EDA.ipynb                       # EDA and feature engineering
│   ├── 02_Preprocessing.ipynb             # Preprocessing and outlier handling
│   ├── 03_Modeling_XGBoost.ipynb          # XGBoost model
│   ├── 04_Modeling_DeepLearning.ipynb     # Deep Learning model
├── requirements.txt                       # Required libraries
├── README.md                              # Project documentation
└── LICENSE                                # Project license

Conclusion

The XGBoost model achieved the best performance with an MAE of 525 seconds, providing a reliable prediction within approximately 8-10 minutes. The deep learning model also performed reasonably well but was slightly less accurate.

Future Work

Potential improvements include:

	•	Feature Engineering: Experiment with additional features to capture more complex patterns.
	•	Ensemble Models: Combine XGBoost and Deep Learning models for potentially improved accuracy.
	•	Hyperparameter Tuning: Further tune the deep learning model’s architecture.
	•	Real-time Prediction: Deploy the XGBoost model for real-time predictions in production.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

Thanks to DoorDash for providing the dataset and framing the problem. This project was developed as part of a take-home assignment for a Data Science role.