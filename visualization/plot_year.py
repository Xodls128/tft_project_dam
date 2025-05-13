# visualization/plot_year.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

def plot_next_year_prediction(csv_path="./results/next_year_only.csv", output_path="./results/next_year_lineplot.png"):
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="date", y="predicted_quantity", hue="menu", marker="o")
    plt.xticks(rotation=45)
    plt.title("\U0001F4C6 다음해 예측 판매량 (메뉴별)")
    plt.ylabel("판매량")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
