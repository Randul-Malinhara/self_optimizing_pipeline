import pytest
import pandas as pd
from src.model_selection import ModelSelection

def test_model_selection():
    # Sample data
    df = pd.DataFrame({
        'feat1': [1, 2, 3, 4],
        'feat2': [2, 3, 4, 5],
        'feat3': [3, 4, 5, 6]
    })
    y = pd.Series([0, 1, 0, 1], name='target')
    
    # Instantiate the class (adjust as needed)
    model_selector = ModelSelection()
    best_model = model_selector.select_best_model(df, y)
    
    # Check that a model was returned
    assert best_model is not None, "No best model was selected"
    # Optional: Check the type of the returned model, e.g.:
    # from sklearn.ensemble import RandomForestClassifier
    # assert isinstance(best_model, RandomForestClassifier)
