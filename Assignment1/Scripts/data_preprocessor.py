import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

class DataPreprocessor:
    def __init__(self, file_path):
        
        self.data = pd.read_csv(file_path)

    def handle_missing_values(self, strategy="mean"):
        
        for column in self.data.columns:
            if self.data[column].isnull().sum() > 0:
                if strategy == "mean" and self.data[column].dtype in ['int64', 'float64']:
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
                elif strategy == "median" and self.data[column].dtype in ['int64', 'float64']:
                    self.data[column].fillna(self.data[column].median(), inplace=True)
                elif strategy == "mode":
                    self.data[column].fillna(self.data[column].mode()[0], inplace=True)

    def remove_duplicates(self):
        
        self.data.drop_duplicates(inplace=True)

    def normalize_numeric_columns(self):
        
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        scaler = MinMaxScaler()
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])

    def encode_categorical_columns(self):
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        self.data = pd.get_dummies(self.data, columns=categorical_cols, drop_first=True)

    def remove_outliers(self, z_threshold=3):
        
        numeric_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        for col in numeric_cols:
            z_scores = (self.data[col] - self.data[col].mean()) / self.data[col].std()
            self.data = self.data[(z_scores.abs() <= z_threshold)]

    def save_cleaned_data(self, output_path):
        
        self.data.to_csv(output_path, index=False)

