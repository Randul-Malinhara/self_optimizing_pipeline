from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class ModelSelector:
    def __init__(self):
        self.models = {
            'LogisticRegression': LogisticRegression(),
            'RandomForest': RandomForestClassifier(),
            'SVM': SVC()
        }
        self.best_model = None

    def select_best_model(self, X, y):
        best_score = 0
        for name, model in self.models.items():
            score = cross_val_score(model, X, y, cv=5).mean()
            if score > best_score:
                best_score = score
                self.best_model = model
        return self.best_model

# Example usage
if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/sample_data.csv")
    X = df.drop(columns=["target"])
