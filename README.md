# EV Adoption & Charging Demand Forecasting 🔋

*A project for the Shell-Edunet Skills4Future Internship | Jul–Aug 2025, organized by Edunet Foundation, AICTE, and Shell.*

This project forecasts electric vehicle (EV) adoption trends across counties in Washington State to support smart infrastructure planning, especially for charging stations. It combines data preprocessing, feature engineering, and a trained Random Forest model deployed through a Streamlit web app.

## 🧠 Objective

Forecast cumulative EV adoption over the next 36 months for U.S. counties using historical data, thereby enabling data-driven infrastructure planning for charging stations.

# ⚙️ Features & Capabilities
    -📈 Interactive Forecasting App built with Streamlit
    
    -🏙️ County-level prediction of EV growth trends
    
    -📊 Compare up to 3 counties simultaneously
    
    -📍 Feature importance insights from model training
    
    -🎨 Custom styling, HTML markdown, and real-time visuals

## 🖥️ How to Run the App Locally
      bash
      # Clone the repo
      git clone https://github.com/your-username/ev-forecasting-app.git
      cd ev-forecasting-app
      
      # Create a virtual environment
      python -m venv .venv
      .venv\Scripts\activate  # On Windows
      # or
      source .venv/bin/activate  # On macOS/Linux
      
      # Install dependencies
      pip install -r requirements.txt
      
      # Run the Streamlit app
## 🗂️ Folder Structure
    ├── app.py                       # Main Streamlit deployment script
    ├── forecasting_ev_model.pkl     # Trained Random Forest model
    ├── preprocessed_ev_data.csv     # Cleaned dataset with features
    ├── assets/
    │   └── ev-car-factory.jpg       # EV image used in dashboard
    ├── README.md
    ├── requirements.txt

## 📂 Project Workflow & Methodology

The project follows a structured machine learning workflow, from data preparation to forecasting future trends.

### 1. Data Cleaning and Preprocessing
The initial dataset contained inconsistencies that were addressed as follows:
- **Missing Values**: Null values in the `County` and `State` columns were identified and filled with the placeholder 'Unknown'.
- **Outlier Detection**: The Interquartile Range (IQR) method was used to identify significant outliers in the `Percent Electric Vehicles` column. A total of 2,476 outliers were detected.
- **Outlier Treatment**: To minimize skewness while retaining data, outliers were capped at the upper and lower bounds calculated using the 1.5 * IQR rule. This approach effectively normalized the distribution without removing valuable data points.
<img width="629" height="470" alt="63d86487-5a9e-4bdc-9762-16c28243665e" src="https://github.com/user-attachments/assets/95e41f7a-b8be-4877-af95-8dc8221b3fb7" />
<img width="629" height="470" alt="82c67d51-eede-432b-8688-62a9939a2b35" src="https://github.com/user-attachments/assets/1e56b6d1-9149-4571-9be1-352d419fef4b" />
<img width="789" height="590" alt="ca469a5d-15b7-4406-b6cf-8fc5ad83e1c1" src="https://github.com/user-attachments/assets/da0fe587-21ab-48de-85e6-69f475f8e362" />

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
<img width="989" height="590" alt="4846f20a-fc9b-40b0-a9a7-7eff037f7f69" src="https://github.com/user-attachments/assets/b1570fac-19f5-4cec-87a0-053f1700cba7" />
<img width="808" height="470" alt="8b06955b-a48e-4cc8-bc3a-1ec8e5182f88" src="https://github.com/user-attachments/assets/c8ff73e1-235b-4048-8da2-620e33d8fcc1" />

### 5. Forecasting
The trained model was used to forecast EV adoption for the next **36 months (3 years)** for all counties. The forecast was generated iteratively, where each new prediction was used as an input for the subsequent month's forecast.
<img width="1189" height="590" alt="ae6fec16-236b-4461-97df-ece81e7c040a" src="https://github.com/user-attachments/assets/404feac6-0759-425c-a793-4ea81a5b53f3" />
<img width="1189" height="590" alt="2226b772-5f84-47f3-9965-c39603b33be9" src="https://github.com/user-attachments/assets/321296d4-dee2-41f4-9e77-bce2b54d7b27" />

### 6. 5. Streamlit Deployment
- Real-time selection (selectbox, multiselect)
- Styled layout with HTML markdown
- Dynamic Matplotlib plots for historical vs. forecasted data
![EV Forecast_page-0001](https://github.com/user-attachments/assets/b8f25d07-7c67-40a5-83a8-d3af995bf19e)
![EV Forecast_page-0002](https://github.com/user-attachments/assets/727ef880-d3e4-47d5-8207-a46de227a451)
![EV Forecast_page-0003](https://github.com/user-attachments/assets/efb2a99f-51d5-4fce-ace9-63ea8a566440)



## 📊 Results and Visualizations

- **Feature Importance**: The analysis revealed that **lag features** (`ev_total_lag2` and `ev_total_lag1`) were the most significant predictors. The **growth slope** also showed high importance, confirming that historical data and recent trends are critical for accurate forecasting.

- **Forecast for Kings County**: A detailed 3-year forecast for Kings County shows a continued linear growth in cumulative EV adoption, suggesting a need for infrastructure expansion.

- **Top 5 Counties Forecast**: A comparative analysis of the top 5 counties (Santa Clara, Fairfax, Orange, Honolulu, and Los Angeles) projects continued strong momentum for Santa Clara and Fairfax, a potential plateau for Orange County, and steady growth for the others.
<img width="1389" height="690" alt="d3b6e743-0753-4af5-bea8-bbb98d275edc" src="https://github.com/user-attachments/assets/e9f9b1ee-df58-4669-8945-c8881bc793f9" />

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
