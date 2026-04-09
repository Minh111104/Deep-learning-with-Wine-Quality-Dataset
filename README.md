# Deep Learning with "Wine Quality" Dataset

A deep learning regression model built with TensorFlow/Keras to predict wine quality scores from physicochemical properties.

## Dataset

- **Source:** [UCI Machine Learning Repository - Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality)
- **Files used:** `data/winequality-red.csv` (1,599 samples) + `data/winequality-white.csv` (4,898 samples)
- **Total samples:** 6,497 (red + white combined)
- **Features:** 12 inputs — 11 physicochemical (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol) + `wine_type` (0=red, 1=white)
- **Target:** `quality` — expert rating from 0 to 10

## Project Structure

```text
├── data/
│   ├── winequality-red.csv
│   ├── winequality-white.csv
│   └── winequality.names
├── Q1_wine_regression.ipynb
└── README.md
```

## Workflow

### 1. Setup & Imports

- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `tensorflow`

### 2. Load & Explore Data

- Loaded CSV with semicolon separator
- Confirmed no missing values (1,599 rows × 12 columns)
- Plotted distribution of quality scores (most wines rated 5–6)

### 3. Preprocess

- Separated features (`X`) and target (`y`)
- Split: **70% train / 15% validation / 15% test**
- Applied `StandardScaler` — fitted on training data only to avoid data leakage

### 4. Build Neural Network

- Keras `Sequential` model
- Architecture: `Input(12)` → `Dense(64, ReLU)` → `Dense(32, ReLU)` → `Dense(16, ReLU)` → `Dense(1, linear)`
- Linear output activation for continuous regression output

### 5. Compile & Train

- Optimizer: `Adam`
- Loss: `MSE` (Mean Squared Error)
- Metric: `MAE` (Mean Absolute Error)
- `EarlyStopping` with `patience=10`, restores best weights
- Max 200 epochs, batch size 32

### 6. Evaluate

- Evaluated on held-out test set: MSE, MAE, RMSE
- Plotted predicted vs. actual quality scores

## Requirements

```text
tensorflow>=2.21.0
scikit-learn
pandas
numpy
matplotlib
```

Install with:

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

## Environment

- Python 3.12.3 (Anaconda)
- TensorFlow 2.21.0 / Keras 3.14.0
