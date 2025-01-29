from preprocess import Preprocessor
from feature_selection import FeatureSelector
from model_selection import ModelSelector
from hyperparam_tuning import HyperparameterOptimizer

class PipelineManager:
    def __init__(self, df, numeric_features, categorical_features):
        self.df = df
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features

    def run_pipeline(self):
        # Step 1: Preprocessing
        preprocessor = Preprocessor(self.numeric_features, self.categorical_features)
        processed_data = preprocessor.preprocess(self.df)

        # Step 2: Feature Selection
        selector = FeatureSelector(k=5)
        X_selected = selector.fit_transform(processed_data, self.df["target"])

        # Step 3: Model Selection
        model_selector = ModelSelector()
        best_model = model_selector.select_best_model(X_selected, self.df["target"])

        # Step 4: Hyperparameter Tuning
        optimizer = HyperparameterOptimizer(X_selected, self.df["target"])
        best_params = optimizer.optimize()

        print("Best Model:", best_model)
        print("Best Hyperparameters:", best_params)

# Example usage
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/sample_data.csv")
    pipeline = PipelineManager(df, numeric_features=["age", "income"], categorical_features=["gender"])
    pipeline.run_pipeline()

import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(filename="results/logs.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

def log_results(model, metrics, params):
    log_data = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Best Model": str(model),
        "Metrics": metrics,
        "Best Hyperparameters": params
    }
    logging.info(json.dumps(log_data, indent=4))
