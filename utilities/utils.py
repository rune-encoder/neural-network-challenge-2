import pandas as pd

# Feature selection
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Model evaluation metrics
import statsmodels.api as sm
import numpy as np


def highlight_vif(row: pd.Series, threshold: float) -> list:
    """
    Highlight VIF values below a given threshold.

    Parameters:
    row (pd.Series): A row of VIF values.
    threshold (float): The threshold for highlighting.

    Returns:
    list: A list of styles for each cell in the row.
    """
    return ["background-color: black" if value < threshold else "" for value in row]


def calc_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) for each feature in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with features.

    Returns:
    pd.DataFrame: A DataFrame containing VIF values for each feature.
    """
    vif_values = [variance_inflation_factor(
        df.values, i) for i in range(df.shape[1])]
    vif = pd.DataFrame(data={"VIF": vif_values}, index=df.columns).sort_values(
        by="VIF", ascending=True)
    return vif


def highlight_p_values(row: pd.Series) -> list:
    """
    Highlight p-values below a significance level of 0.05.

    Parameters:
    row (pd.Series): A row of p-values.

    Returns:
    list: A list of styles for each cell in the row.
    """
    return ["background-color: black" if value <= 0.05 else "" for value in row]


def calc_p_values(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Calculate p-values for each feature using OLS regression.

    Parameters:
    X (pd.DataFrame): The input DataFrame with features.
    y (pd.Series): The target variable.

    Returns:
    tuple: A DataFrame containing p-values for each feature and the OLS model.
    """
    ols_model = sm.OLS(y, X).fit()
    p_values_df = ols_model.pvalues.sort_values().to_frame(name="p_value")
    return p_values_df, ols_model


def calc_correlation(df: pd.DataFrame, threshold: float = None):
    """
    Calculate the correlation matrix, optionally filter rows and columns based on a positive correlation threshold,
    and apply a color gradient for visualization.

    Parameters:
    df (pd.DataFrame): The input DataFrame with features.
    threshold (float, optional): The positive correlation threshold for filtering. Default is None.

    Returns:
    pd.io.formats.style.Styler: A styled DataFrame with the correlation matrix.
    """
    # Compute the correlation matrix
    corr_matrix = df.corr()

    if threshold is not None:
        # Filter the correlation matrix to keep only rows and columns with positive correlations above the threshold
        high_corr = corr_matrix[(corr_matrix > threshold) & (corr_matrix != 1.0)]

        # Drop rows and columns with all NaN values
        high_corr = high_corr.dropna(axis=0, how="all").dropna(axis=1, how="all")

        # Keep only the columns and rows that have high correlations
        keep_columns = high_corr.columns.tolist()
        keep_index = high_corr.index.tolist()

        # Create a new correlation matrix with the filtered rows and columns
        filtered_corr_matrix = corr_matrix.loc[keep_index, keep_columns]

        # Apply a color gradient for visualization
        styled_corr_matrix = filtered_corr_matrix.style.background_gradient(cmap="coolwarm")
    else:
        # Apply a color gradient for visualization without filtering
        styled_corr_matrix = corr_matrix.style.background_gradient(cmap="coolwarm")

    return styled_corr_matrix
