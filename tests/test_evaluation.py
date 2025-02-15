import pytest
import pandas as pd
import numpy as np
from src.evaluation import Evaluation
from sklearn.ensemble import RandomForestClassifier

def test_evaluation():
    # Sample data
    df = pd.DataFrame({
        'feat1': [1, 2, 3, 4],
        'feat2': [2, 3, 4, 5]
    })
    y = pd.Series([0, 1, 0, 1], name='target')
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(df, y)
    
    evaluator = Evaluation()
    metrics = evaluator.evaluate_model(model, df, y)
    
    # Check if metrics is a dictionary or has the expected structure
    assert isinstance(metrics, dict), "Metrics should be returned in a dictionary"
    assert 'accuracy' in metrics, "Missing accuracy in evaluation results"
    assert 0.0 <= metrics['accuracy'] <= 1.0, "Accuracy should be between 0 and 1"
