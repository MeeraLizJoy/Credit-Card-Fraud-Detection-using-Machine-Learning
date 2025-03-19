# Credit Card Fraud Detection Project - Observations

## Initial Data Loading and Analysis (`load_and_initial_analysis.py`)

* **Shape:** The dataset contains a large number of transactions, but the number of fraudulent transactions is expected to be very small, indicating a class imbalance.
* **Info:** The dataset contains numerical features (V1-V28, Amount, Time) and a binary target variable (Class). No missing values were observed.
* **Descriptive Statistics:** The "Amount" feature shows a wide range and high standard deviation, suggesting potential outliers. The "Time" feature represents seconds elapsed from the first transaction, which might need transformation.

## Exploratory Data Analysis (EDA) (`eda.py`)

* **Amount Distribution:** The "Amount" feature is heavily right-skewed, with most transactions being small and a few very large ones. Log transformation might be helpful.
* **Correlation Matrix:** Some "V" variables show strong correlations, suggesting potential interaction features. "Amount" doesn't show strong linear correlations with most "V" variables.
* **Amount vs. V Variables:** Scatter plots reveal complex relationships between "Amount" and "V" variables, indicating potential non-linear patterns.
* **Class Distribution:** The dataset is highly imbalanced, with very few fraudulent transactions. This necessitates techniques like SMOTE to balance the data.
* **Time Series Analysis:**
    * Transactions occur throughout the time period, but there are fluctuations in volume.
    * Fraudulent transactions also occur throughout the time period, but their distribution might differ from normal transactions.
    * A low activity period was found around the 100,000 seconds mark.
* **Fraudulent Transaction Analysis:**
    * The distribution of "Amount" for fraudulent transactions differs from the overall distribution.
    * "V" variable distributions for fraudulent transactions show distinct patterns.
    * The correlation matrix and scatter plots for fraudulent transactions reveal specific relationships.
* **Box and Violin Plots:** These plots highlight differences in "Amount" and "V" variable distributions between normal and fraudulent transactions.
* **Numerical Analysis:** Percentiles, IQR, and Z-scores confirm the presence of outliers and inform preprocessing decisions.

## Preprocessing (`preprocessing.py`)

* **Amount Outlier Handling:** Capped "Amount" at the 1st and 99th percentiles to mitigate the impact of extreme outliers.
* **Amount Log Transformation:** Applied log transformation to "Amount" to reduce skewness and improve distribution.
* **Amount Scaling:** Scaled the log-transformed "Amount" using StandardScaler.
* **V Variable Outlier Handling:** Used IQR method to handle outliers in the "V" variables.

## Feature Engineering (`feature_engineering.py`)

* **Interaction Features:** Created interaction features (V12_V14_interaction, V10_V17_interaction) to capture non-linear relationships.
* **Low Amount Binary Feature:** Generated a binary feature (low_amount) for transactions below 50.
* **Polynomial Features:** Created polynomial features (V3^2, V4^2, V3 V4) to capture complex relationships between V3 and V4.
* **Time-Based Feature:** Extracted the hour of the day (hour) from the "Time" feature.
* **Feature Combination:** Created a combined feature (amount_V2_combined) by multiplying "Amount" and "V2".

## Balancing and Feature Selection (`balancing_and_feature_selection.py`)

* **SMOTE Balancing:** Used SMOTE to address the class imbalance, improving the model's ability to detect fraud.
* **Random Forest Feature Importance:** Used Random Forest to rank feature importance.
* **Feature Selection:** Selected the top 24 features based on importance scores, reducing dimensionality and improving model efficiency.
* **Feature Importance Visualization:** Generated a bar plot of feature importances, showing that V4, V12_V14_interaction, and polynomial features of V3 and V4 are the most important.

## Model Training and Tuning (`model_training_and_tuning.py`)

* **Initial Model:** Trained an initial Random Forest model with default hyperparameters.
* **Hyperparameter Tuning:** Performed hyperparameter tuning using RandomizedSearchCV on a subset of the training data.
* **Final Model:** Trained the final Random Forest model with the best hyperparameters on the full training data.
* **Evaluation:** Evaluated the final model using classification reports and AUC, demonstrating good performance in fraud detection.
* **Subset Tuning:** The tuning was done on a 20% subset of the data to reduce computational load.
* **Final model training:** The final model was trained on the full training data to maximize learning.