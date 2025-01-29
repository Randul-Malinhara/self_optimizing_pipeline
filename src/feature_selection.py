import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureSelector:
    def __init__(self, k=10):
        self.k = k
        self.selector = None

    def fit_transform(self, X, y):
        self.selector = SelectKBest(score_func=f_classif, k=self.k)
        return self.selector.fit_transform(X, y)

    def get_selected_features(self, feature_names):
        mask = self.selector.get_support()
        return [feature for feature, selected in zip(feature_names, mask) if selected]

# Example usage
if __name__ == "__main__":
    df = pd.read_csv("data/sample_data.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    selector = FeatureSelector(k=5)
    X_new = selector.fit_transform(X, y)
    print("Selected Features:", selector.get_selected_features(X.columns))
