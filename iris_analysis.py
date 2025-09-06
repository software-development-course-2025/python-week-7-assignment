# iris_analysis.py

import pandas as pd
from sklearn.datasets import load_iris

def load_and_inspect_iris():
    print("🔍 Loading the Iris dataset...")

    try:
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

        print("\n✅ First 5 rows:")
        print(df.head())

        print("\n🧠 Dataset Info:")
        print(df.info())

        print("\n🔎 Missing values check:")
        print(df.isnull().sum())

        return df

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    df = load_and_inspect_iris()
