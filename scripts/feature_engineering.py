import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import PolynomialFeatures

def feature_engineer(df):
    """
    Performs feature engineering on the DataFrame.

    Args:
        df (pandas.DataFrame): DataFrame to engineer features from.
    """

    # 1. Interaction Features (Strong Correlations)
    df['V12_V14_interaction'] = df['V12'] * df['V14']
    df['V10_V17_interaction'] = df['V10'] * df['V17']

    # 2. Features Based on "Amount" Clumping (Low Amount Binary Feature)
    df['low_amount'] = np.where(df['Amount'] < 50, 1, 0)

    # 3. Polynomial Features (Example: V3 and V4)
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    poly_features = poly.fit_transform(df[['V3', 'V4']])
    poly_feature_names = poly.get_feature_names_out(['V3', 'V4'])
    df[poly_feature_names] = poly_features

    # 4. Time-Based Features (Example: hour of day)
    df['hour'] = df['Time'] // 3600

    # 5. Feature Combinations (Example: Amount with V2)
    df['amount_V2_combined'] = df['Amount'] * df['V2']

    joblib.dump(df, 'engineered_data.joblib') # Save the engineered dataframe for later use
    print("\nFeature engineering complete. Dataframe saved as 'engineered_data.joblib'")

if __name__ == "__main__":
    df = joblib.load('preprocessed_data.joblib') # Load the preprocessed dataframe
    feature_engineer(df)
