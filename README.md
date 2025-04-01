# Credit Card Fraud Detection Project

This project aims to detect credit card fraud using machine learning techniques. It involves data loading, exploratory data analysis (EDA), preprocessing, feature engineering, balancing, feature selection, and model training/tuning.

## Project Structure

The project is organized into separate Python scripts for each stage of the process:

* `load_and_initial_analysis.py`: Loads the dataset and performs initial exploratory analysis.
* `eda.py`: Performs Exploratory Data Analysis (EDA) and generates visualizations.
* `preprocessing.py`: Preprocesses the data by handling outliers and scaling features.
* `feature_engineering.py`: Engineers new features from the existing data.
* `balancing_and_feature_selection.py`: Balances the data using SMOTE and selects important features using Random Forest.
* `model_training_and_tuning.py`: Trains, tunes, and evaluates a Random Forest model.

## Data

The dataset is taken from Kaggle:
link: `https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud`

The dataset used in this project is `creditcard.csv`, which contains credit card transactions with features `V1` to `V28`, `Amount`, and `Time`. The target variable is `Class`, where 1 indicates fraud and 0 indicates a normal transaction.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    * **Windows:**
        ```bash
        venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Upload `creditcard.csv` to your Kaggle notebook's data section (if using Kaggle). Or place the file in the same directory as the scripts if using locally.**

5.  **Run the Python scripts in the following order:**
    ```bash
    python load_and_initial_analysis.py
    python eda.py
    python preprocessing.py
    python feature_engineering.py
    python balancing_and_feature_selection.py
    python model_training_and_tuning.py
    ```
    * **If you're using kaggle:**
        ```python
        !python load_and_initial_analysis.py
        !python eda.py
        !python preprocessing.py
        !python feature_engineering.py
        !python balancing_and_feature_selection.py
        !python model_training_and_tuning.py
        ```

## Requirements

The project dependencies are listed in `requirements.txt`.

## Virtual Environment

It is highly recommended to create a virtual environment to isolate the project dependencies from your system's global Python packages. This ensures that the project runs with the correct versions of the libraries.

## Project Workflow

1.  **Data Loading and Initial Analysis:**
    * The `load_and_initial_analysis.py` script loads the dataset and provides basic information about its structure and content.

2.  **Exploratory Data Analysis (EDA):**
    * The `eda.py` script generates visualizations to understand the data distribution, correlations, and patterns.
    * It includes histograms, box plots, correlation matrices, scatter plots, and time series analysis.

3.  **Preprocessing:**
    * The `preprocessing.py` script handles outliers in the "Amount" and "V" variables and scales the "Amount" feature.

4.  **Feature Engineering:**
    * The `feature_engineering.py` script creates new features, including interaction features, binary features, polynomial features, time-based features, and feature combinations.

5.  **Balancing and Feature Selection:**
    * The `balancing_and_feature_selection.py` script balances the data using SMOTE and selects the top 24 features using Random Forest feature importance.
    * A feature importance plot is generated.

6.  **Model Training and Tuning:**
    * The `model_training_and_tuning.py` script trains an initial Random Forest model, tunes its hyperparameters using RandomizedSearchCV on a subset of the training data, and trains the final model on the full training data with the best hyperparameters.
    * The final model is evaluated using classification reports and AUC.

## Outputs

* Visualizations: Saved as `.png` files in the Kaggle notebook's output directory.
* Preprocessed, engineered, balanced, and selected data: Saved as `.joblib` files.
* Trained Random Forest models (initial and final): Saved as `.joblib` files.
* Evaluation results: Printed to the console.

## Results

The final tuned Random Forest model achieves 

              precision    recall  f1-score   support

           0       1.00      1.00      1.00     85307
           1       0.91      0.85      0.88       136

    accuracy                           1.00     85443
   macro avg       0.96      0.92      0.94     85443
weighted avg       1.00      1.00      1.00     85443

AUC: 0.9772009865406537


## Future Improvements

* Experiment with other machine learning models (e.g., Logistic Regression, XGBoost).
* Perform more advanced feature engineering techniques.
* Explore different hyperparameter tuning strategies.
* Implement a more robust cross-validation strategy.
* Deploy the model for real-time fraud detection.
