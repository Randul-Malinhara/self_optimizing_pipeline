import pytest
import pandas as pd
from src.pipeline_manager import PipelineManager

def test_pipeline_manager():
    # Create a small sample dataset
    df = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [4, 5, 6]
    })
    y = pd.Series([0, 1, 0], name='target')
    
    pipeline = PipelineManager()
    
    # Assume PipelineManager has methods like run_pipeline
    results = pipeline.run_pipeline(df, y)
    
    # Check the pipeline's output structure
    assert results is not None, "Pipeline manager returned None"
    assert 'best_model' in results, "No best_model found in pipeline results"
    assert 'metrics' in results, "No metrics found in pipeline results"
