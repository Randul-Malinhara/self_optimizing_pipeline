import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.preprocess import Preprocessor
from src.feature_selection import FeatureSelector
from src.model_selection import ModelSelector
from src.hyperparam_tuning import HyperparameterOptimizer
from src.evaluation import Evaluator

class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Load sample data and prepare it for testing."""
        cls.df = pd.DataFrame({
            'age': [25, 30, 35, 40, np.nan, 50],
            'income': [50000, 60000, 70000, 80000, 90000, 100000],
            'gender': ['M', 'F', 'M', 'F', 'M', 'F'],
            'target': [0, 1, 0, 1, 0, 1]
        })
        
        cls.numeric_features = ['age', 'income']
        cls.categorical_features = ['gender']
        
        cls.X = cls.df.drop(columns=['target'])
        cls.y = cls.df['target']

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(cls.X, cls.y, test_size=0.2, random_state=42)

    def test_preprocessing(self):
        """Test data preprocessing"""
        preprocessor = Preprocessor(self.numeric_features, self.categorical_features)
        transformed_data = preprocessor.preprocess(self.X_train)
        self.assertIsNotNone(transformed_data)

    def test_feature_selection(self):
        """Test feature selection"""
        selector = FeatureSelector(k=2)
        X_selected = selector.fit_transform(self.X_train, self.y_train)
        self.assertEqual(X_selected.shape[1], 2)

    def test_model_selection(self):
        """Test model selection"""
        selector = ModelSelector()
        best_model = selector.select_best_model(self.X_train, self.y_train)
        self.assertIsInstance(best_model, RandomForestClassifier)

    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning"""
        optimizer = HyperparameterOptimizer(self.X_train, self.y_train)
        best_params = optimizer.optimize(n_trials=2)  # Reduced trials for speed
        self.assertIn("n_estimators", best_params)
        self.assertIn("max_depth", best_params)

    def test_evaluation(self):
        """Test model evaluation"""
        model = RandomForestClassifier(n_estimators=10)
        model.fit(self.X_train, self.y_train)
        evaluator = Evaluator(model, self.X_test, self.y_test)
        metrics = evaluator.compute_metrics()
        self.assertGreater(metrics["Accuracy"], 0)

if __name__ == "__main__":
    unittest.main()
