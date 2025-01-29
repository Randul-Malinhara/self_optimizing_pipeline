import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

class HyperparameterOptimizer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def objective(self, trial):
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 2, 20)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        return cross_val_score(model, self.X, self.y, cv=5).mean()

    def optimize(self, n_trials=10):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        return study.best_params

# Example usage
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/sample_data.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    optimizer = HyperparameterOptimizer(X, y)
    best_params = optimizer.optimize()
    print("Best Parameters:", best_params)
