import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

def balance_and_select(df):
    """
    Balances the data using SMOTE and performs feature selection using Random Forest.

    Args:
        df (pandas.DataFrame): DataFrame to balance and select features from.
    """

    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Perform feature selection using Random Forest
    rf_fs = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_fs.fit(X_train_resampled, y_train_resampled)
    feature_importances = rf_fs.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_train_resampled.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Visualize feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20))
    plt.title('Feature Importances (Random Forest)')
    plt.savefig('feature_importance.png')
    plt.close()

    # Select the top 24 features
    selected_features = feature_importance_df['Feature'].head(24).tolist()
    X_train_selected = X_train_resampled[selected_features]
    X_test_selected = X_test[selected_features]

    # Save the balanced and selected data
    joblib.dump((X_train_selected, X_test_selected, y_train_resampled, y_test), 'balanced_selected_data.joblib')
    print("\nBalancing and feature selection complete. Data saved as 'balanced_selected_data.joblib'")

if __name__ == "__main__":
    df = joblib.load('engineered_data.joblib') # Load the engineered dataframe
    balance_and_select(df)
