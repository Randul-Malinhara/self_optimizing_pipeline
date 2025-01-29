import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self):
        self.pipeline = None

    def build_pipeline(self, numeric_features, categorical_features):
        # Imputers
        num_imputer = SimpleImputer(strategy='mean')
        cat_imputer = SimpleImputer(strategy='most_frequent')
        
        # Transformers
        scaler = StandardScaler()
        encoder = OneHotEncoder(handle_unknown='ignore')
        
        # Pipelines
        num_pipeline = Pipeline([('imputer', num_imputer), ('scaler', scaler)])
        cat_pipeline = Pipeline([('imputer', cat_imputer), ('encoder', encoder)])
        
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, numeric_features),
                ('cat', cat_pipeline, categorical_features),
            ]
        )

    def preprocess(self, data):
        return self.pipeline.fit_transform(data)

# Usage
if __name__ == "__main__":
    data = pd.read_csv("data.csv")
    numeric_features = ['age', 'income']
    categorical_features = ['gender', 'occupation']
    
    preprocessor = Preprocessor()
    preprocessor.build_pipeline(numeric_features, categorical_features)
    processed_data = preprocessor.preprocess(data)
    print("Preprocessed Data:", processed_data)
