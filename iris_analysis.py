# iris_analysis.py

import pandas as pd
from sklearn.datasets import load_iris
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Suppress FutureWarnings (cleaner output for reviewers)
warnings.filterwarnings("ignore", category=FutureWarning)

# Use Agg backend for headless environments (safe for servers and CI)
matplotlib.use('Agg')


# ---------------------------
# Data loading and inspection
# ---------------------------
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
        print("\nℹ️ Dataset Information:")
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


# ---------------------------
# Visualization functions
# ---------------------------
def save_visualizations(df, outdir='images'):
    """Save example visualizations to `outdir` (PNG files)."""
    os.makedirs(outdir, exist_ok=True)
    sns.set(style="whitegrid")

    # 1) Line chart (simulated trend)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
    ax.plot(df.index, df['sepal width (cm)'], label='Sepal Width')
    ax.set_title('Sepal Length and Width Trends')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Centimeters')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'line_sepal_trends.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 2) Bar chart (average petal length by species)
    mean_petal_length = df.groupby('species')['petal length (cm)'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(data=mean_petal_length, x='species', y='petal length (cm)', ax=ax)
    ax.set_title('Average Petal Length by Species')
    ax.set_xlabel('Species')
    ax.set_ylabel('Average Petal Length (cm)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'bar_mean_petal_length.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 3) Histogram (petal width distribution)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(df['petal width (cm)'], bins=15, edgecolor='black')
    ax.set_title('Distribution of Petal Width')
    ax.set_xlabel('Petal Width (cm)')
    ax.set_ylabel('Frequency')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'hist_petal_width.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    # 4) Scatter plot (sepal length vs petal length)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', ax=ax)
    ax.set_title('Sepal Length vs Petal Length by Species')
    ax.set_xlabel('Sepal Length (cm)')
    ax.set_ylabel('Petal Length (cm)')
    ax.legend(title='Species')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'scatter_sepal_vs_petal.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ Saved visualizations to '{outdir}/'")


# ---------------------------
# Main entry point
# ---------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Iris dataset analysis and optional figure export")
    parser.add_argument("--save-figs", action="store_true", help="Generate and save example plots to ./images/")
    args = parser.parse_args()

    df = load_and_inspect_iris()
    if df is None:
        raise SystemExit("Failed to load dataset. Exiting.")

    if args.save_figs:
        save_visualizations(df)
