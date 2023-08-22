# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 13:53:52 2023

@author: m8abb
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from joblib import dump

def load_data(wine_type):
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    if wine_type == "red":
        return pd.read_csv(base_url + "winequality-red.csv", sep=";")
    elif wine_type == "white":
        return pd.read_csv(base_url + "winequality-white.csv", sep=";")
    else:
        raise ValueError("Invalid wine type. Choose either 'red' or 'white'.")

def preprocess_data(df):
    X = df.drop("quality", axis=1)
    y = df["quality"]
    
    # Determine the smallest class size to set k_neighbors
    min_class_size = y.value_counts().min() - 1
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(k_neighbors=min_class_size, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

def train_and_evaluate(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}\n")
    print("Classification Report:\n")
    print(classification_report(y_test, predictions, zero_division=1))
    
    return clf

def main():
    for wine_type in ["red", "white"]:
        print(f"Training model for {wine_type} wine...\n")
        df = load_data(wine_type)
        X_train, X_test, y_train, y_test = preprocess_data(df)
        model = train_and_evaluate(X_train, X_test, y_train, y_test)
        
        # Save the trained model
        dump(model, f'{wine_type}_wine_quality_model.joblib')
        print(f"Model for {wine_type} wine saved as {wine_type}_wine_quality_model.joblib\n")

if __name__ == "__main__":
    main()