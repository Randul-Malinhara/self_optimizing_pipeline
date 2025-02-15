import pytest
import pandas as pd
from src.feature_selection import FeatureSelector

def test_feature_selector():
    # Sample data
    df = pd.DataFrame({
        'feat1': [1, 2, 3, 4],
        'feat2': [2, 3, 4, 5],
        'feat3': [3, 4, 5, 6]
    })
    y = pd.Series([0, 1, 0, 1], name='target')
    
    fs = FeatureSelector()
    df_selected = fs.select_features(df, y)
    
    assert df_selected is not None, "Feature selection returned None"
    # You can also assert that at least one feature remains
    assert df_selected.shape[1] > 0, "No features selected"
