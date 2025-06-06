# AI Based Data Leakage Detection

This project explores various machine learning models to detect data leakage based on user activity logs. The goal is to identify anomalous behavior that could indicate a potential data breach.

## Project Overview

Data leakage is a critical security concern for organizations. This project aims to build and evaluate different machine learning models for identifying patterns in user activity that deviate from normal behavior, potentially indicating data leakage. We implement and compare Isolation Forest, Autoencoder, and Random Forest models for this anomaly detection task.

## Dataset

The project uses the `Data_Leakage_Detection.csv` dataset. This dataset contains various attributes related to user activity, including date, user ID, authentication methods, and actions performed. The target variable is 'Abnormality', indicating whether an activity is considered normal (0) or anomalous (1).

## Methodology

This project employs a multi-faceted approach to data leakage detection, exploring both unsupervised and supervised machine learning techniques. The core steps involve data loading, preprocessing, model implementation, training, and evaluation.

### Data Loading and Initial Preview

The analysis begins by loading the `Data_Leakage_Detection.csv` dataset into a pandas DataFrame. Initial steps involve examining the first few rows (`df.head(10)`), checking the dataset's dimensions (`df.shape`), and inspecting the data types and non-null values (`df.info()`).

### Data Preprocessing

Data preprocessing is crucial for preparing the data for different models. The specific steps vary slightly depending on the model requirements:

- **Missing Value Handling:** For the Isolation Forest and Random Forest models, rows with missing values were initially removed (`df.dropna()`). For the Autoencoder model, missing values were imputed using the median for numerical features and the most frequent value for categorical features.
- **Data Type Transformation:** The 'date' column is converted to datetime objects (`pd.to_datetime`). Several boolean-like columns (`Through_pwd`, `Through_pin`, `Through_MFA`, `Data Modification`, `Confidential Data Access`, `Confidential File Transfer`, `Abnormality`) are converted to integer type (`astype('int')`). For the Autoencoder and Random Forest models, the 'date' column was dropped, and for the Autoencoder, the date was converted to ordinal for scaling.
- **Feature Engineering (for Autoencoder and Random Forest):** Irrelevant columns like 'id' and 'date' are dropped.
- **Encoding Categorical Features:**
    - **Isolation Forest:** OneHotEncoder is applied to categorical columns, handling unknown categories by ignoring them.
    - **Autoencoder and Random Forest:** LabelEncoder is used to convert categorical features into numerical representations.
- **Scaling Numerical Features:** StandardScaler is applied to numerical features to standardize their range, which is particularly important for models like Isolation Forest and Autoencoder that are sensitive to feature scaling.
- **Splitting Data:** The dataset is split into training, validation, and/or test sets using `train_test_split`. For the Random Forest model, stratification is used to ensure that the proportion of the target class ('Abnormality') is maintained in both training and test sets. The Autoencoder model is specifically trained on the normal data (where 'Abnormality' is 0).

### Model Implementations and Training

Three distinct machine learning models were implemented and trained:

#### Isolation Forest

- **Model:** `sklearn.ensemble.IsolationForest`
- **Training:** The model is trained on the preprocessed training data (`X_train`).
- **Parameters:** The model is initialized with specific parameters, including `n_estimators=1000`, `max_samples=0.8`, `contamination=0.2`, `max_features=0.75`, and `random_state=42`.
- **Prediction:** The trained model is used to predict anomalies on both the training and validation sets. Predictions (-1 for anomaly, 1 for normal) are converted to binary labels (1 for anomaly, 0 for normal).

#### Autoencoder

- **Model:** A simple feedforward neural network autoencoder built with `tensorflow.keras`.
- **Architecture:** The autoencoder consists of an input layer, an encoded layer with 6 units and 'relu' activation, and a decoded layer with a dimensionality equal to the input dimension and 'sigmoid' activation.
- **Compilation:** The model is compiled with the Adam optimizer (`learning_rate=0.001`) and the 'mse' loss function.
- **Training:** The autoencoder is trained on the scaled normal data (`X_train`) for 50 epochs with a batch size of 32, shuffling the data during training. Validation is performed on the reserved normal validation data (`X_val`).
- **Anomaly Detection:** After training, the reconstruction error is calculated for all data points. A threshold for anomaly detection is determined as the 95th percentile of the reconstruction error for the normal data. Data points with a reconstruction error above this threshold are classified as anomalies.

#### Random Forest

- **Model:** `sklearn.ensemble.RandomForestClassifier`
- **Training:** The model is trained on the preprocessed training data (`X_train_rf`).
- **Parameters:** The model is initialized with `class_weight='balanced'` to handle potential class imbalance and `random_state=42`.
- **Prediction:** The trained model is used to predict anomaly labels (`y_pred_rf`) and the probability of being anomalous (`y_pred_proba_rf`) on the test set.

### Evaluation

Each model's performance is evaluated using a comprehensive set of metrics relevant to classification and anomaly detection. The evaluation includes:

- **Classification Report:** Provides precision, recall, F1-score, and support for both the 'Normal' and 'Anomalous' classes on the validation/test sets.
- **Accuracy Score:** Measures the overall correctness of predictions (for Random Forest).
- **ROC AUC Score:** Evaluates the model's ability to distinguish between positive and negative classes (for Autoencoder and Random Forest).
- **Confusion Matrix:** Visualizes the counts of true positive, true negative, false positive, and false negative predictions (for Random Forest).

The generated plots visually compare the Precision, Recall, and F1-score of the Isolation Forest, Autoencoder, and Random Forest models for both normal and anomalous classes, providing a clear picture of their performance across different metrics. An additional plot compares the overall accuracy of the three models.

## How to Run the Code

1. Clone the repository: bash git clone


2. Navigate to the project directory.
3. Ensure you have the necessary dependencies installed (see [Dependencies](#dependencies)).
4. Run the Jupyter Notebook or Python script containing the code.

## Dependencies

The following libraries are required to run the code:

- `numpy`
- `pandas`
- `seaborn`
- `plotly.express`
- `matplotlib.pyplot`
- `sklearn` (with modules: `model_selection`, `preprocessing`, `impute`, `ensemble`, `metrics`)
- `tensorflow` (specifically `tensorflow.keras`)

You can install the dependencies using pip: bash pip install numpy pandas seaborn plotly scikit-learn tensorflow

