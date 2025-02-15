import pytest
import pandas as pd
from src.preprocess import Preprocessor
from src.feature_selection import FeatureSelector
from src.model_selection import ModelSelection
from src.hyperparam_tuning import HyperparamTuning
from src.evaluation import Evaluation

def test_end_to_end_pipeline():
    # Sample dataset
    df = pd.DataFrame({
        'feat1': [1, 2, 3, 4],
        'feat2': [2, 3, 4, 5],
        'feat3': [3, 4, 5, 6]
    })
    y = pd.Series([0, 1, 0, 1], name='target')
    
    # 1. Preprocess
    preprocessor = Preprocessor()
    df_preprocessed = preprocessor.fit_transform(df)
    
    # 2. Feature Selection
    fs = FeatureSelector()
    df_selected = fs.select_features(df_preprocessed, y)
    
    # 3. Model Selection
    ms = ModelSelection()
    model = ms.select_best_model(df_selected, y)
    
    # 4. Hyperparameter Tuning
    tuner = HyperparamTuning()
    tuned_model = tuner.tune(model, df_selected, y)
    
    # 5. Evaluation
    evaluator = Evaluation()
    metrics = evaluator.evaluate_model(tuned_model, df_selected, y)
    
    # Basic checks
    assert metrics is not None, "End-to-end pipeline returned no metrics"
    assert 'accuracy' in metrics, "No accuracy metric found in end-to-end pipeline"
    assert 0.0 <= metrics['accuracy'] <= 1.0, "Accuracy should be between 0 and 1"
