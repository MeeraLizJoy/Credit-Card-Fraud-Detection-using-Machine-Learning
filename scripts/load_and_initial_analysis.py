import pandas as pd
import numpy as np

def load_and_analyze(input_file):
    """
    Loads the dataset and performs initial exploratory analysis.

    Args:
        input_file (str): Path to the input CSV file.

    Returns:
        pandas.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(input_file)

    # Basic shape and information
    print("Shape:", df.shape)
    print("\nInfo:")
    print(df.info())

    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print(df.describe())

    return df

if __name__ == "__main__":
    df = load_and_analyze('creditcard.csv')  # Load the data
    df.to_pickle('initial_dataframe.pkl') # Save the dataframe for later use
    print("\nInitial Analysis complete. Dataframe saved as 'initial_dataframe.pkl'")
