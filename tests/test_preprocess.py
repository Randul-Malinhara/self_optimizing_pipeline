import pytest
import pandas as pd
from src.preprocess import Preprocessor

def test_preprocessor_fit_transform():
    # Sample data
    df = pd.DataFrame({
        'numeric_feature': [1, 2, None],
        'categorical_feature': ['cat', 'dog', 'cat']
    })
    
    preprocessor = Preprocessor()
    df_transformed = preprocessor.fit_transform(df)
    
    # Basic checks
    assert df_transformed is not None, "Preprocessing returned None"
    assert not df_transformed.isnull().values.any(), "Missing values still exist after preprocessing"
    assert df_transformed.shape[0] == 3, "Row count changed unexpectedly"

    # Example check for new columns (e.g., encoded categorical features)
    # Adjust the number of columns as per your Preprocessor logic
    # e.g., assert df_transformed.shape[1] == expected_num_cols
