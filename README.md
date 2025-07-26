# EV Adoption & Charging Demand Forecasting 🔋

*A project for the Shell-Edunet Skills4Future Internship | Jul–Aug 2025, organized by Edunet Foundation, AICTE, and Shell.*

This repository contains the code and analysis for predicting electric vehicle (EV) adoption trends. By leveraging historical data, this project aims to forecast future demand, providing insights that can aid in the strategic development of charging infrastructure.

## 🧠 Objective

The primary goal is to predict demand trends for electric vehicles and charging stations. This forecasting supports smart infrastructure planning and helps stakeholders make data-driven decisions to accommodate the growing EV market.

## 📂 Project Workflow & Methodology

The project follows a structured machine learning workflow, from data preparation to forecasting future trends.

### 1. Data Cleaning and Preprocessing
The initial dataset contained inconsistencies that were addressed as follows:
- **Missing Values**: Null values in the `County` and `State` columns were identified and filled with the placeholder 'Unknown'.
- **Outlier Detection**: The Interquartile Range (IQR) method was used to identify significant outliers in the `Percent Electric Vehicles` column. A total of 2,476 outliers were detected.
- **Outlier Treatment**: To minimize skewness while retaining data, outliers were capped at the upper and lower bounds calculated using the 1.5 * IQR rule. This approach effectively normalized the distribution without removing valuable data points.

### 2. Feature Engineering
To improve the model's predictive power, several new features were created from the existing data:
- **Time-Based Features**: The `Date` column was used to extract `year`, `month`, and a `numeric_date` to capture temporal patterns.
- **Categorical Encoding**: The `County` column was converted into numerical format using `LabelEncoder`, making it suitable for the machine learning model.
- **Lag Features**: To provide historical context, lag features (`ev_total_lag1`, `ev_total_lag2`, `ev_total_lag3`) were created to represent the EV counts from the previous one, two, and three months.
- **Trend and Growth Features**:
  - A **3-month rolling mean** was calculated to smooth out short-term fluctuations.
  - **Percentage change** over 1 and 3 months was computed to capture growth rates.
  - A **6-month growth slope** based on the cumulative EV count was engineered to model the trend's steepness.

### 3. Model Training and Hyperparameter Tuning
- **Model Selection**: A **Random Forest Regressor** was chosen for its robustness and ability to handle non-linear relationships in the data.
- **Data Splitting**: The dataset was split into training (90%) and testing (10%) sets. `shuffle=False` was used to maintain the chronological order of the time-series data.
- **Hyperparameter Tuning**: **RandomizedSearchCV** was employed to find the optimal combination of hyperparameters. This technique randomly samples from a predefined parameter space, efficiently exploring a wide range of configurations. The model was evaluated using the **R² score**, and the best parameters were identified as:
  - `n_estimators`: 200
  - `max_depth`: 15
  - `min_samples_split`: 4
  - `min_samples_leaf`: 1
  - `max_features`: None

### 4. Model Evaluation
The tuned model performed exceptionally well on the test set, achieving the following metrics:
- **Mean Absolute Error (MAE)**: 0.01
- **Root Mean Squared Error (RMSE)**: 0.06
- **R² Score**: 1.00

The R² score of 1.00 indicates a perfect fit on the test data, suggesting the model can explain nearly all the variance in the target variable.

### 5. Forecasting
The trained model was used to forecast EV adoption for the next **36 months (3 years)** for all counties. The forecast was generated iteratively, where each new prediction was used as an input for the subsequent month's forecast.

## 📊 Results and Visualizations

- **Feature Importance**: The analysis revealed that **lag features** (`ev_total_lag2` and `ev_total_lag1`) were the most significant predictors. The **growth slope** also showed high importance, confirming that historical data and recent trends are critical for accurate forecasting.

- **Forecast for Kings County**: A detailed 3-year forecast for Kings County shows a continued linear growth in cumulative EV adoption, suggesting a need for infrastructure expansion.

- **Top 5 Counties Forecast**: A comparative analysis of the top 5 counties (Santa Clara, Fairfax, Orange, Honolulu, and Los Angeles) projects continued strong momentum for Santa Clara and Fairfax, a potential plateau for Orange County, and steady growth for the others.

## ⚙️ Key Technical Concepts

- **Hyperparameter Tuning**: The process of selecting the optimal external configurations for a machine learning model (e.g., the number of trees in a Random Forest). Unlike model parameters, these are not learned from the data. Techniques like `GridSearchCV` and `RandomizedSearchCV` are used to automate this process.

- **Random Forest**: An ensemble learning method that builds multiple decision trees on random subsets of the data and averages their predictions. This approach helps reduce overfitting and improves the model's generalization capabilities.

- **Lag Features**: Historical values of a time series used as input features for a model. For example, the EV count from one month ago (`lag1`) can help predict the count for the current month.

- **Smoothing**: A technique used to reduce noise and reveal underlying patterns in time-series data. The **rolling mean (or moving average)** is a common smoothing method that averages data points over a sliding window.

- **Label Encoding**: A method for converting categorical text data into a numerical format. Each unique category is assigned an integer value, which allows machine learning algorithms to process the data.

## 🛠️ Tech Stack

- **Python**: Core programming language for data analysis and modeling.
- **Pandas**: For data manipulation, cleaning, and feature engineering.
- **NumPy**: For numerical operations, especially in data preprocessing.
- **Matplotlib & Seaborn**: For data visualization and plotting results.
- **Scikit-learn**: For implementing the machine learning model (`RandomForestRegressor`), hyperparameter tuning (`RandomizedSearchCV`), and evaluation metrics.
- **Jupyter Notebook**: For interactive development and analysis.
