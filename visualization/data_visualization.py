# models/data_visualization.py


import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns



def plot_next_week_prediction():
    # 한글 폰트 설정
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    df = pd.read_csv("./results/next_week_only.csv")
    df["date"] = pd.to_datetime(df["date"])

    # 메뉴별 lineplot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="date", y="predicted_quantity", hue="menu", marker="o")
    plt.xticks(rotation=45)
    plt.title("📈 다음주 예측 판매량 (메뉴별)")
    plt.ylabel("판매량")
    plt.tight_layout()
    plt.savefig("./results/next_week_lineplot.png")
    plt.close()
