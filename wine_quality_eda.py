# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 01:09:00 2023

@author: m8abb
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(wine_type):
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
    if wine_type == "red":
        return pd.read_csv(base_url + "winequality-red.csv", sep=";")
    elif wine_type == "white":
        return pd.read_csv(base_url + "winequality-white.csv", sep=";")
    else:
        raise ValueError("Invalid wine type. Choose either 'red' or 'white'.")

def initial_exploration(df, wine_type):
    print(f"Initial Exploration for {wine_type.capitalize()} Wine:\n")
    print(df.head())
    print("\nBasic Statistics:\n")
    print(df.describe())
    print("\nMissing values in each column:\n")
    print(df.isnull().sum())
    print("\n")

def visualize_data(df, wine_type):
    # Visualize the distribution of wine qualities
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='quality')
    plt.title(f'Distribution of {wine_type.capitalize()} Wine Qualities')
    plt.show()

    # Box plots to visualize the relationship between features and wine quality
    features = df.columns[:-1]  # Exclude 'quality' column
    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='quality', y=feature)
        plt.title(f'Relationship between {feature} and {wine_type.capitalize()} Wine Quality')
        plt.show()

    # Correlation matrix
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f'Correlation Matrix for {wine_type.capitalize()} Wine')
    plt.show()

def main():
    for wine_type in ["red", "white"]:
        print(f"Analyzing {wine_type} wine data...\n")
        df = load_data(wine_type)
        initial_exploration(df, wine_type)
        visualize_data(df, wine_type)

if __name__ == "__main__":
    main()
