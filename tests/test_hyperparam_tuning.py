import pytest
import pandas as pd
from src.hyperparam_tuning import HyperparamTuning
from sklearn.ensemble import RandomForestClassifier

def test_hyperparam_tuning():
    # Sample data
    df = pd.DataFrame({
        'feat1': [1, 2, 3, 4],
        'feat2': [2, 3, 4, 5]
    })
    y = pd.Series([0, 1, 0, 1], name='target')
    
    # A sample model
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Instantiate your hyperparameter tuner (adjust as needed)
    tuner = HyperparamTuning()
    tuned_model = tuner.tune(model, df, y)
    
    assert tuned_model is not None, "Hyperparameter tuning returned None"
    # Check if the tuned model is still a RandomForestClassifier or a pipeline
    assert hasattr(tuned_model, 'predict'), "Tuned model has no predict method"
