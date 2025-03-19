import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def perform_eda(df):
    """
    Performs Exploratory Data Analysis (EDA) and saves visualizations.

    Args:
        df (pandas.DataFrame): DataFrame to analyze.
    """
    # Histograms: Distribution of Transaction Amounts
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Amount'], bins=50)
    plt.title('Distribution of Transaction Amounts')
    plt.savefig('amount_hist.png')
    plt.close()

    # Box Plot: Transaction Amounts
    plt.figure(figsize=(8, 6))
    sns.boxplot(df['Amount'])
    plt.title('Transaction Amounts')
    plt.savefig('amount_box.png')
    plt.close()

    # Correlation Matrix Analysis
    corr = df.corr()
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.savefig('corr_matrix.png')
    plt.close()

    # Scatter Plot Analysis: Amount vs. V1-V28
    for i in range(1, 29):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Amount', y=f'V{i}', data=df)
        plt.title(f'Amount vs. V{i}')
        plt.savefig(f'scatter_amount_v{i}.png')
        plt.close()

    # Class Distribution Analysis
    plt.figure(figsize=(6, 4))
    sns.countplot(df['Class'])
    plt.title('Class Distribution')
    plt.savefig('class_dist.png')
    plt.close()

    # Time Series Analysis: Transaction Volume
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time'], label='All Transactions')
    plt.title('Transaction Volume Over Time')
    plt.savefig('time_series_all.png')
    plt.close()

    # Histogram: Distribution of Transactions Over Time
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Time'], bins=100)
    plt.title('Distribution of Transactions Over Time')
    plt.savefig('time_hist.png')
    plt.close()

    # Line Plot: Fraudulent Transactions Over Time
    fraud_df = df[df['Class'] == 1] # Here is where we make the dataset with only fraud
    plt.figure(figsize=(12, 6))
    plt.plot(fraud_df['Time'], label='Fraudulent Transactions', color='red')
    plt.title('Fraudulent Transactions Over Time')
    plt.savefig('time_series_fraud.png')
    plt.close()

    # Histogram: Distribution of Fraudulent Transactions Over Time
    plt.figure(figsize=(12, 6))
    sns.histplot(fraud_df['Time'], bins=100, color='red') # Using fraud_df
    plt.title('Distribution of Fraudulent Transactions Over Time')
    plt.savefig('time_hist_fraud.png')
    plt.close()

    # Overlay: All vs. Fraudulent Transactions (Line Plot)
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time'], label='All Transactions')
    plt.plot(fraud_df['Time'], label='Fraudulent Transactions', color='red') # Using fraud_df
    plt.title('Overlay: All vs. Fraudulent Transactions (Line Plot)')
    plt.savefig('time_overlay_line.png')
    plt.close()

    # Overlay: All vs. Fraudulent Transactions (Histogram)
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Time'], bins=100, label='All Transactions')
    sns.histplot(fraud_df['Time'], bins=100, label='Fraudulent Transactions', color='red') # Using fraud_df
    plt.title('Overlay: All vs. Fraudulent Transactions (Histogram)')
    plt.savefig('time_overlay_hist.png')
    plt.close()

    # Low Activity Period Around 100,000 Seconds
    low_activity_df = df[(df['Time'] > 90000) & (df['Time'] < 110000)]
    fraud_low_activity_df = low_activity_df[low_activity_df['Class'] == 1]

    # Distribution of Amount for Fraudulent Transactions
    plt.figure(figsize=(10, 6))
    sns.histplot(fraud_df['Amount'], bins=50, color='red') # Using fraud_df
    plt.title('Distribution of Amount for Fraudulent Transactions')
    plt.savefig('fraud_amount_hist.png')
    plt.close()

    # Analysis of V1 to V28 Distributions for Fraudulent Transactions
    for i in range(1, 29):
        plt.figure(figsize=(10, 6))
        sns.histplot(fraud_df[f'V{i}'], bins=50, color='red') # Using fraud_df
        plt.title(f'Distribution of V{i} for Fraudulent Transactions')
        plt.savefig(f'fraud_v{i}_hist.png')
        plt.close()

    # Correlation Matrix for Fraudulent Transactions
    fraud_corr = fraud_df.corr() # Using fraud_df
    plt.figure(figsize=(20, 15))
    sns.heatmap(fraud_corr, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix for Fraudulent Transactions')
    plt.savefig('fraud_corr_matrix.png')
    plt.close()

    # Scatter Plots: Amount vs. V1 to V28 for Fraudulent Transactions
    for i in range(1, 29):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Amount', y=f'V{i}', data=fraud_df) # Using fraud_df
        plt.title(f'Amount vs. V{i} for Fraudulent Transactions')
        plt.savefig(f'fraud_scatter_amount_v{i}.png')
        plt.close()

    # Box Plot: Amount vs. Class
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Class', y='Amount', data=df)
    plt.title('Amount vs. Class')
    plt.savefig('amount_vs_class_box.png')
    plt.close()

    # Violin Plots: V1 to V28 vs. Class
    for i in range(1, 29):
        plt.figure(figsize=(8, 6))
        sns.violinplot(x='Class', y=f'V{i}', data=df)
        plt.title(f'V{i} vs. Class')
        plt.savefig(f'violin_v{i}_class.png')
        plt.close()

    # Further Numerical Analysis (Percentiles, IQR, Z-scores)
    print("\nNumerical Analysis (Percentiles, IQR, Z-scores):")
    print(df.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]))

    joblib.dump(df, 'analyzed_data.joblib') # Save the dataframe for later use
    print("\nEDA complete. Dataframe saved as 'analyzed_data.joblib'")

if __name__ == "__main__":
    df = joblib.load('initial_dataframe.pkl') # Load the dataframe that was saved from the initial analysis
    perform_eda(df)
