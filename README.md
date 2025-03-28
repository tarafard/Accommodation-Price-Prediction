# Airbnb Price Prediction in New York City

This project aims to predict Airbnb listing prices in New York City using machine learning and deep learning techniques. The dataset includes features such as location, room type, neighborhood, and other listing attributes.

## Project Overview

- **Objective**: Predict Airbnb listing prices based on various features.
- **Dataset**: [AB_NYC_2019.csv](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data) containing 48,895 listings with 16 features.
- **Key Steps**:
  - Exploratory Data Analysis (EDA)
  - Data Cleaning and Preprocessing
  - Feature Engineering
  - Model Training and Evaluation
  - Hyperparameter Tuning


---


## Key Findings

### EDA Highlights
- **Room Types**: 
  - Entire home/apartment (52%) is the most expensive, followed by private rooms (45%) and shared rooms (3%).
- **Neighborhoods**: 
  - Over 85% of listings are in Manhattan and Brooklyn.
  - Manhattan has the highest average price ($196), while the Bronx has the lowest ($87).
- **Price Distribution**: 
  - Right-skewed; log transformation was applied for better modeling.

### Data Preprocessing
- Handled missing values by filling `reviews_per_month` with 0 and dropping `last_review`.
- Removed outliers using IQR for the `price` column.
- Applied log transformation to the target variable (`price`).
- Encoded categorical features (`neighbourhood_group`, `neighbourhood`, `room_type`) using One-Hot Encoding.
- Scaled numerical features using `StandardScaler`.


### Code Structure
- **Notebook**: Contains the full analysis, visualizations, and model training.
- **Sections**:
  1. Data Loading and Initial Exploration
  2. EDA and Visualizations
  3. Data Cleaning and Feature Engineering
  4. Model Training and Evaluation
  5. Hyperparameter Tuning
  6. Neural Network Implementation
 
 ---
 
## Models Used

#### 1. Classical Machine Learning Models

##### Linear Models:
- **Ridge Regression** (`Ridge`)
- **Lasso Regression** (`Lasso`)

##### Tree-Based:
- **Decision Tree Regressor** (`DecisionTreeRegressor`)

#### 2. Ensemble Methods

##### Bagging (Parallel Ensembles):
- **Random Forest Regressor** (`RandomForestRegressor`)
  - Base version
  - Tuned version (hyperparameter-optimized)

##### Boosting (Sequential Ensembles):
- **Gradient Boosting Regressor** (`GradientBoostingRegressor`)
  - Base version
  - Tuned version
- **XGBoost Regressor** (`XGBRegressor`)
  - Base version
  - Tuned version


#### 3. Neural Networks

##### Artificial Neural Network (ANN):
- **Sequential model** with:
  - Dense layers (128, 64, 32 neurons)
  - Batch normalization and dropout
  - Early stopping and checkpointing
- **Hyperparameter-tuned version** (using `RandomizedSearchCV`)

---

#### Dependencies
- Python 3.10+
- Libraries: 
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn`, `xgboost`, `tensorflow`
  - `yellowbrick` (for visualization)
