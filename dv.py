import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = "cleaned_dataset.csv" 

df = pd.read_csv(DATA_PATH)

true_num_cols = [col for col in ["Age", "Fare", "SibSp", "Parch"] if col in df.columns]
target_col = "Survived" if "Survived" in df.columns else None

fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1.2, 1.2])

stats_df = df.describe().round(2).reset_index()

ax_table = fig.add_subplot(gs[0, :])
ax_table.axis("off")
ax_table.set_title("Dataset Summary Statistics", fontsize=14, weight="bold", pad=10)

tbl = ax_table.table(
    cellText=stats_df.values,
    colLabels=stats_df.columns,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.3)

for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#2c3e50")

plot_positions = [(1, 0), (1, 1), (2, 0), (2, 1)]
for idx, col in enumerate(true_num_cols):
    if idx < len(plot_positions):
        r, c = plot_positions[idx]
        ax_hist = fig.add_subplot(gs[r, c])
        sns.histplot(df[col], kde=True, bins=20, color="teal", ax=ax_hist)
        ax_hist.set_title(f"Distribution of {col}")

plt.tight_layout()
plt.show()

numeric_df = df.select_dtypes(include=[np.number])

if "Parch" in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=["Parch"])

if not numeric_df.empty:
    plt.figure(figsize=(10, 8))
    correlation_matrix = numeric_df.corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
    )
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()

if target_col and "Pclass" in df.columns:
    plt.figure(figsize=(6, 5))
    sns.barplot(
        x=df["Pclass"].astype(str),
        y=df[target_col],
        legend=False,
        errorbar=None,
        palette="muted",
    )
    plt.title("Survival Rate by Passenger Class")
    plt.ylabel("Survival Probability")
    plt.tight_layout()
    plt.show()
