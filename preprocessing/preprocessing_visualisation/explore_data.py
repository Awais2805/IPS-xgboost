import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from pathlib import Path 
import numpy as np

SPLITS_DIR = Path("dataset/data-split/splits")
OUTPUT_DIR = Path("dataset/viz/plots")

def run_viz():

    print("Loading data samples for visualisation")

    y_train = pd.read_parquet(SPLITS_DIR / "y_train.parquet")
    y_test = pd.read_parquet(SPLITS_DIR / "y_test.parquet")

    print("Sampled 100 000 rows for complex plots")
    X_train_sample = pd.read_parquet(SPLITS_DIR / "X_train.parquet").sample(n=100000, random_state=42)
    y_train_sample = y_train.loc[X_train_sample.index]

    df_plot = X_train_sample.copy()
    df_plot['Target'] = y_train_sample['Target']

    sns.set_theme(style="whitegrid")

    # Plot 1: Stratification Verification 

    print("Generating Stratification Plot")
    fig, axes = plt.subplots(1,2, figsize=(12,5))

    sns.countplot(x='Target', data=y_train, ax=axes[0], palette="viridis")
    axes[0].set_title(f"Training Set Distribution\nTotal: {len(y_train):,}")
    axes[0].set_xticklabels(['Benign (0)', 'Attack (1)'])

    sns.countplot(x='Target', data=y_test, ax=axes[1], palette="viridis")
    axes[1].set_title(f"Testing Set Distribution\nTotal: {len(y_test):,}")
    axes[1].set_xticklabels(['Benign (0)', 'Attack (1)'])

    plt.tight_layout()
    plt.savefig("dataset/viz/plots/stratification_comparison.png")
    plt.close()


    # Plot 2: Correlation Heatmap 

    print("Generating Correlation Heatmap")

    correlations = df_plot.corr()['Target'].abs().sort_values(ascending=False)
    top_features = correlations.index[1:16]

    plt.figure(figsize=(12,10))
    corr_matrix = df_plot[list(top_features)+['Target']].corr()

    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title("Top 15 Features Correlated with Target (Attack vs Benign)")
    plt.tight_layout()
    plt.savefig("dataset/viz/plots/correlation_heatmap.png")
    plt.close()


    # Plot 3: Feature Seperability 
    print("Generating Feature Seperability Plots")

    features_to_plot = top_features[:3] 
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, feature in enumerate(features_to_plot):
        df_plot[f'log_{feature}'] = np.log1p(df_plot[feature])
        
        sns.violinplot(x='Target', y=f'log_{feature}', data=df_plot, ax=axes[i], palette="muted")
        axes[i].set_title(f"Log Distribution of {feature}")
        axes[i].set_xticklabels(['Benign (0)', 'Attack (1)'])


    plt.tight_layout()
    plt.savefig("dataset/viz/plots/feature_separability.png")
    plt.close()

    print(f"\nVisualisations complete - check {OUTPUT_DIR}")

if __name__ == "__main__":
    run_viz()