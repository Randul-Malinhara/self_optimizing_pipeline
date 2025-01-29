import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

class Evaluator:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = self.model.predict(self.X_test)

    def compute_metrics(self):
        """Calculates and returns key performance metrics."""
        metrics = {
            "Accuracy": accuracy_score(self.y_test, self.y_pred),
            "Precision": precision_score(self.y_test, self.y_pred, average='weighted'),
            "Recall": recall_score(self.y_test, self.y_pred, average='weighted'),
            "F1 Score": f1_score(self.y_test, self.y_pred, average='weighted')
        }
        return metrics

    def plot_confusion_matrix(self):
        """Plots the confusion matrix."""
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(self.y_test), yticklabels=np.unique(self.y_test))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def classification_report(self):
        """Prints a detailed classification report."""
        print(classification_report(self.y_test, self.y_pred))

# Example Usage
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    df = pd.read_csv("data/sample_data.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    evaluator = Evaluator(model, X_test, y_test)
    print("Metrics:", evaluator.compute_metrics())
    evaluator.plot_confusion_matrix()
    evaluator.classification_report()
