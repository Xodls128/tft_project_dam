# models/predictor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from torch.utils.data import DataLoader

class Predictor:
    def __init__(self, model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path = data_path
        self.model = None
        self.dataset = None
        self.dataloader = None
        self.raw_data = None
        self.predictions = None
        self.result_df = None

    def load_data(self):
        df = pd.read_excel(self.data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        daily = df.groupby(["date", "menu"]).size().reset_index(name="sold_quantity")
        daily["date"] = pd.to_datetime(daily["date"])
        daily["menu_id"] = daily["menu"].astype("category").cat.codes
        daily = daily.sort_values("date")

        start_date = daily["date"].min()
        daily["time_idx"] = (daily["date"] - start_date).dt.days

        # 미래 예측 구간 추가
        last_date = daily["date"].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
        menus = daily["menu"].unique()
        future = pd.DataFrame([(d, m) for d in future_dates for m in menus], columns=["date", "menu"])
        future["timestamp"] = future["date"]
        future["menu_id"] = future["menu"].astype("category").cat.codes
        future["sold_quantity"] = 0
        future["time_idx"] = (future["date"] - start_date).dt.days

        self.raw_data = pd.concat([daily, future], ignore_index=True).sort_values(["menu_id", "date"])

    def setup_dataset(self):
        self.dataset = TimeSeriesDataSet(
            self.raw_data,
            time_idx="time_idx",
            target="sold_quantity",
            group_ids=["menu_id"],
            max_encoder_length=30,
            max_prediction_length=7,
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=["sold_quantity"],
            allow_missing_timesteps=True
        )
        self.dataloader = self.dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    def load_model(self):
        self.model = TemporalFusionTransformer.load_from_checkpoint(self.model_path)


    def predict(self):
        raw_predictions, x = self.model.predict(self.dataloader, mode="raw", return_x=True)
        decoder_target = x["decoder_target"]
        time_idxs = x["decoder_time_idx"].detach().cpu().numpy()
        menu_ids = x["groups"][0].detach().cpu().numpy()
        print("x keys:", x.keys())

        # 평균 + 예측분산으로 오차 추정 (QuantileLoss 기반)
        means = raw_predictions["prediction"].mean(1).detach().cpu().numpy()
        p10 = raw_predictions["prediction"][:, :, 0].detach().cpu().numpy()
        p90 = raw_predictions["prediction"][:, :, -1].detach().cpu().numpy()

        print("menu_ids (after repeat):", np.repeat(menu_ids, means.shape[1]).shape)
        print("time_idxs:", time_idxs.flatten().shape)
        print("means:", means.flatten().shape)
        print("p10:", p10.flatten().shape)
        print("p90:", p90.flatten().shape)

        self.result_df = pd.DataFrame({
            "menu_id": np.repeat(menu_ids, means.shape[1]),
            "time_idx": time_idxs.flatten(),
            "predicted_quantity": means.flatten(),
            "p10": p10.flatten(),
            "p90": p90.flatten()
        })

        # menu name + date 복원
        menu_map = self.raw_data[["menu_id", "menu"]].drop_duplicates().set_index("menu_id")["menu"].to_dict()
        time_map = self.raw_data[["time_idx", "date"]].drop_duplicates().set_index("time_idx")["date"].to_dict()
        self.result_df["menu"] = self.result_df["menu_id"].map(menu_map)
        self.result_df["date"] = self.result_df["time_idx"].map(time_map)
        self.result_df = self.result_df.sort_values(["date", "menu"])

    def plot_forecast(self):
        for menu in self.result_df["menu"].unique():
            df_m = self.result_df[self.result_df["menu"] == menu]
            plt.figure(figsize=(10, 4))
            plt.plot(df_m["date"], df_m["predicted_quantity"], label="예측", marker="o")
            plt.fill_between(df_m["date"], df_m["p10"], df_m["p90"], alpha=0.3, label="오차범위")
            plt.title(f"{menu}의 예측 판매량")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"../results/{menu}_forecast.png")
            plt.close()

    def save_predictions(self):
        self.result_df.to_csv("../results/next_week_predictions.csv", index=False)
