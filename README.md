# Delivery Duration Prediction for DoorDash

This project aims to build a machine learning model to predict the delivery duration for orders on DoorDash. Accurate delivery time predictions are critical for improving customer experience by setting realistic expectations. This project was developed to showcase skills in data preprocessing, feature engineering, and model training using various regression models, including Random Forest, XGBoost, and Deep Learning.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Models Used](#models-used)
- [Results](#results)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Project Overview
The objective of this project is to predict the total time taken for an order to be delivered based on historical data. This is achieved through various machine learning models, with a focus on performance evaluation and model optimization.

## Dataset
The dataset provided contains historical data on DoorDash orders, with each record representing one order along with various features. Key columns include:
- `created_at`: Timestamp when the order was placed.
- `actual_delivery_time`: Timestamp when the order was delivered.
- Various features related to order items, store details, and market conditions at the time of order creation.

## Features
The following feature engineering steps were implemented:
- **Time-Based Features**: Extracted `day_of_week`, `hour_of_day`, and `is_weekend` from the timestamp data.
- **Ratios and Metrics**: Created features like `busy_dashers_ratio` and `workload_ratio` to capture the state of the delivery environment.
- **Outlier Detection**: Used the Interquartile Range (IQR) method to flag outliers for key numerical columns.
- **Encoding**: Applied one-hot encoding for categorical features such as `store_primary_category` and frequency encoding for high-cardinality features like `store_id`.

## Models Used
The following models were trained and evaluated for this task:
1. **Random Forest Regressor**: A robust ensemble learning method based on decision trees, which provided a good balance of interpretability and performance.
2. **XGBoost Regressor**: An optimized gradient boosting algorithm that excelled in terms of predictive accuracy.
3. **Deep Learning Model**: A neural network implemented using Keras with layers of Dense, Dropout, and BatchNormalization to handle complex patterns in the data.

## Results
Each model was evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) to measure its predictive performance.

- **Random Forest**: 
  - MAE: 613.86
  - RMSE: 788.85

- **XGBoost** (optimized): 
  - MAE: 525.25
  - RMSE: 679.88

- **Deep Learning Model**: 
  - MAE: 610.14 (optimized)

Among these models, the optimized XGBoost model achieved the best performance, followed closely by the Deep Learning model and the Random Forest regressor.

## Requirements
To install the necessary dependencies, you can use `pip` with the provided `requirements.txt` file:

```bash
pip install -r requirements.txt

Usage

1. Clone the Repository

git clone https://github.com/your-username/delivery-duration-prediction.git
cd delivery-duration-prediction

2. Run the Notebooks

   •	00_Assignment.ipyn: Info about the assignment and data dictionary
	•	01_EDA.ipynb: Exploratory Data Analysis and feature engineering.
	•	02_Preprocessing.ipynb: Preprocess data and save the final dataset.
	•	03_Modeling_Training.ipynb: Train and evaluate the Random Forest Regressor and the XGBoost model.
	•	04_Modeling_DeepLearning.ipynb: Train and evaluate the Deep Learning model.

Each notebook is self-contained and can be executed in sequence. Ensure the preprocessed data file processed_data_final.csv is available before running the modeling notebooks.

Project Structure

.
├── data/
│   ├── historical_data.csv                # Original dataset
│   ├── processed_data_final.csv           # Final processed dataset
├── notebooks/
│   ├── 00_Assignment.ipynb                # Info about the assignment and data   
│   ├── 01_EDA.ipynb                       # EDA and feature engineering
│   ├── 02_Preprocessing.ipynb             # Preprocessing and outlier handling
│   ├── 03_Modeling_Training.ipynb         # XGBoost model
│   ├── 04_DeepLearning.ipynb              # Deep Learning model
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