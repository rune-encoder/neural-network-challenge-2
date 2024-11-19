import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

def encode_categorical_data(df: pd.DataFrame, column_name: str, drop_first: bool = False) -> pd.DataFrame:
    """
    Encode a categorical feature using OneHotEncoder.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to encode.

    Returns:
    pd.DataFrame: A DataFrame with the encoded feature.
    """
    encoder = OneHotEncoder(
        drop='first' if drop_first else None,
        sparse_output=False,
        dtype=np.int64
    )

    encoded = encoder.fit_transform(df[column_name].values.reshape(-1, 1))

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out([column_name]),
    )
    return encoded_df

def encode_categorical_target(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Encode a categorical target using LabelEncoder.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column_name (str): The name of the column to encode.

    Returns:
    pd.DataFrame: A DataFrame with the encoded target.
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(df[column_name])
    encoded_df = pd.DataFrame(encoded, columns=[column_name], dtype=np.int64)
    return encoded_df
