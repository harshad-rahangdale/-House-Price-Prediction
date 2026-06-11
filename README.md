# House Price Prediction

A machine learning project that predicts residential house prices using regression models built with scikit-learn pipelines.

---

# Overview

This project explores a house price dataset through exploratory data analysis (EDA) and trains multiple regression models to predict house prices based on property features like area, number of bedrooms, location, and condition.

---

# Dataset

**File:** `House Price Prediction Dataset.csv`

**Features:**

| Feature | Type | Description |
|---|---|---|
| `Area` | Numerical | Size of the house |
| `Bedrooms` | Numerical | Number of bedrooms |
| `Bathrooms` | Numerical | Number of bathrooms |
| `Floors` | Numerical | Number of floors |
| `YearBuilt` | Numerical | Year the house was constructed |
| `Location` | Categorical | Location of the house |
| `Condition` | Categorical | Condition of the house |
| `Garage` | Categorical | Garage availability |
| `Price` | Numerical | **Target variable** — house price |

---

##  Exploratory Data Analysis

The notebook performs the following EDA steps:

- **Data inspection** — shape, column types, value counts, and basic statistics
- **Missing value & duplicate checks**
- **Univariate analysis** — histograms and boxen plots for all numerical features
- **Categorical analysis** — count plots and pie charts for `Location`, `Condition`, and `Garage`
- **Bivariate analysis** — price breakdown by location, condition, and garage using bar plots

---

##  Preprocessing Pipeline

A `ColumnTransformer` is used to handle different feature types:

- **Numerical features** (`Area`, `Bedrooms`, `Bathrooms`, `Floors`, `YearBuilt`) → `RobustScaler`
- **Categorical features** (`Location`, `Condition`, `Garage`) → `OneHotEncoder` (with `handle_unknown='ignore'`)

---

##  Models Trained

| Model | Notes |
|---|---|
| **Decision Tree Regressor** | `max_depth=5`, `min_samples_split=10`, `random_state=42` |
| **Linear Regression** | Baseline linear model |

Both models are wrapped in a full `sklearn.Pipeline` (preprocessor + model).

---

##  Evaluation Metrics

Each model is evaluated on the test set (20% split) using:

- **MAE** — Mean Absolute Error
- **MSE** — Mean Squared Error
- **R²** — Coefficient of Determination

---

##  Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Run the Notebook

```bash
jupyter notebook house_price_prediction.ipynb
```

> **Note:** Update the dataset path in the notebook from the hardcoded local path to your own:
> ```python
> df = pd.read_csv("House Price Prediction Dataset.csv")
> ```

---

##  Project Structure

```
house-price-prediction/
│
├── house_price_prediction.ipynb   # Main notebook
├── House Price Prediction Dataset.csv  # Dataset (not included)
└── README.md                      # Project documentation
```

---

##  Tech Stack

- **Python 3.x**
- **pandas** — data manipulation
- **numpy** — numerical operations
- **matplotlib / seaborn** — data visualization
- **scikit-learn** — preprocessing, modeling, and evaluation

---

##  Future Improvements

- Add `RandomForestRegressor` and `XGBoost` for better accuracy
- Perform hyperparameter tuning with `GridSearchCV`
- Add cross-validation for more robust evaluation
- Engineer new features (e.g., house age from `YearBuilt`)
- Export the best model using `joblib` for deployment
