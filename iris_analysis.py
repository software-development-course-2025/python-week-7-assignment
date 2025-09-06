# iris_analysis.py

import pandas as pd
from sklearn.datasets import load_iris

def load_and_inspect_iris():
    print("🔍 Loading the Iris dataset...")

    try:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

        # Display first 5 rows
        print("\n✅ First 5 rows:")
        print(df.head())

        # DataFrame info
        print("\nℹ️ Dataset Info:")
        print(df.info())

        # Check for missing values
        print("\n🔎 Missing Values Check:")
        print(df.isnull().sum())

        # Descriptive statistics
        print("\n📊 Descriptive Statistics:")
        print(df.describe())

        # Group by species and compute means
        print("\n🌸 Mean Measurements per Species:")
        print(df.groupby("species").mean())

        # Simple observation
        print("\n📝 Observation:")
        print("Setosa has significantly smaller petal length and width compared to the other species.")

        return df

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    df = load_and_inspect_iris()
