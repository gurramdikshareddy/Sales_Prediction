# Sales Prediction Using Machine Learning and Google Colab

## Overview
This repository contains a machine learning-based sales prediction model built using the BigMart Sales Data. The goal of this project is to predict the sales of products based on various features such as item weight, visibility, price, and store type. Several machine learning algorithms are applied to build a robust predictive model.

## Key Features
- **Sales Prediction**: Predict product sales using multiple machine learning algorithms, including XGBoost, Random Forest, Gradient Boosting, and more.
- **Exploratory Data Analysis (EDA)**: Visualizations and data preprocessing techniques to clean and prepare the data for model training.
- **Google Colab Integration**: The project is developed using Google Colab for easy collaboration and access to computational resources.

## Tools and Technologies
- **Google Colab**: The primary tool used for coding, data manipulation, and model development.
- **Python**: The programming language used for model implementation and analysis.
- **pandas**: Data manipulation and analysis.
- **numpy**: Numerical operations.
- **matplotlib & seaborn**: Data visualization.
- **scikit-learn**: For data preprocessing, model training, and evaluation.
- **XGBoost, RandomForest, GradientBoosting, LGBMRegressor**: Machine learning models used for prediction.
- **Joblib**: For saving and loading models for future predictions.

## Project Structure
├── README.md # Project overview and documentation ├── main_code.ipynb # Jupyter notebook with all code for data processing and model training ├── XGBRegressor.joblib # Saved model for making predictions ├── Data/ │ ├── Train.csv # Training dataset │ └── Test.csv # Test dataset


## Steps and Workflow

### 1. Data Collection and Processing
The dataset is collected and cleaned by performing the following steps:
- **Handling Missing Data**: Missing values in columns like `Item_Weight` and `Outlet_Size` are filled using the mean and mode, respectively.
- **Feature Encoding**: Categorical features such as `Item_Fat_Content`, `Outlet_Type`, and `Item_Identifier` are encoded using `LabelEncoder`.
- **Feature Scaling**: Numerical features such as `Item_Weight`, `Item_Visibility`, and `Item_MRP` are standardized using `StandardScaler` to improve model performance.

### 2. Exploratory Data Analysis (EDA)
Using Google Colab, the following visualizations and analyses were performed:
- **Distribution Plots**: Distribution of numerical features like `Item_Weight`, `Item_Visibility`, etc.
- **Count Plots**: Visualizations to examine the distribution of categorical variables such as `Outlet_Type`, `Item_Fat_Content`, etc.
- **Correlation Analysis**: Correlation matrix and heatmap to understand relationships between features.

### 3. Model Building
The following machine learning models were trained to predict the sales:
- **XGBoost Regressor**: A powerful boosting algorithm that performs well on tabular data.
- **Random Forest Regressor**: An ensemble learning method for regression tasks.
- **Gradient Boosting Regressor**: A popular boosting algorithm used for prediction tasks.
- **LightGBM Regressor**: A gradient boosting framework efficient for large datasets.

### 4. Model Evaluation
The models are evaluated using the following metrics:
- **R-Squared**: Measures how well the model fits the data.
- **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions.
- **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values.

### 5. Saving the Model
The best-performing model (XGBoost Regressor) is saved using `Joblib` for future use in making predictions on new data.

### 6. Predictions on Test Data
Once the model is trained, predictions are made on the test dataset. The final output is a CSV file containing the predicted sales for each `Item_Identifier`.

## Google Colab Integration
Google Colab was used to perform the following tasks:
- Running code on a cloud-based Jupyter notebook with access to free GPU/TPU resources for faster computation.
- Using Colab's built-in visualization tools (matplotlib, seaborn) for generating graphs and exploring data.
- Installing and using Python libraries directly in the notebook environment (e.g., scikit-learn, XGBoost, LightGBM).
- Real-time collaboration: Google Colab enables easy sharing and collaborative work.

## Requirements
To run this project locally, you'll need to install the following Python libraries:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm lazypredict joblib

Alternatively, you can run the provided Jupyter notebook directly on Google Colab by uploading the `main_code.ipynb` notebook.

## How to Use the Model

### 1. Load the Saved Model
To load the saved XGBoost model and make predictions:
```python
from joblib import load
model = load('XGBRegressor.joblib')
predictions = model.predict(test_data)

2. Make Predictions
After loading the model, pass your test data to the model to predict the sales for respective products.

3. Output
The predictions will include the Item_Identifier and predicted sales, which can be saved into a new CSV file for further analysis.

Conclusion
This project demonstrates the complete workflow of building a sales prediction model from scratch. It covers data cleaning, feature engineering, model building, and evaluation. The use of Google Colab facilitates easy collaboration and access to computational resources for training and evaluating the models.

Feel free to explore the code, modify it to suit your needs, and contribute to the project. If you have any questions, don't hesitate to open an issue!
