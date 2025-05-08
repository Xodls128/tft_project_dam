# models/predictor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from torch.utils.data import DataLoader
from pytorch_forecasting.data import NaNLabelEncoder

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
        # (1) 원본 엑셀 불러오기
        df = pd.read_excel(self.data_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        df['menu'] = df['menu'].astype(str)

         # (2) 메뉴 → menu_id 명시적 매핑 (일관성 확보)
        menu_list = sorted(df["menu"].unique())
        menu_id_map =  {name: str(i) for i, name in enumerate(menu_list)}
        df["menu_id"] = df["menu"].map(menu_id_map)

        # (3) 일자-메뉴 단위 판매량 집계
        daily = df.groupby(["date", "menu", "menu_id"]).size().reset_index(name="sold_quantity")
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")

        # (4) time_idx 계산
        start_date = daily["date"].min()
        daily["time_idx"] = (daily["date"] - start_date).dt.days

        # (5) 미래 예측용 7일 생성
        last_date = daily["date"].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
        future = pd.DataFrame([(d, m) for d in future_dates for m in menu_list], columns=["date", "menu"])
        future["menu_id"] = future["menu"].map(menu_id_map)
        future["timestamp"] = future["date"]
        future["sold_quantity"] = 0
        future["time_idx"] = (future["date"] - start_date).dt.days
    
         # (6) 합치기 + menu 복원 컬럼 추가
        self.raw_data = pd.concat([daily, future], ignore_index=True)
        self.raw_data = self.raw_data.sort_values(["menu_id", "date"])
        self.raw_data["menu"] = self.raw_data["menu_id"].map({v: k for k, v in menu_id_map.items()})

        # (7) menu_id를 명시적으로 category로 변환
        self.raw_data["menu_id"] = self.raw_data["menu_id"].astype("category")

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
            allow_missing_timesteps=True,
            static_categoricals=["menu_id"],
            categorical_encoders={'menu_id': NaNLabelEncoder().fit(self.raw_data.menu_id)}
        )
        self.dataloader = self.dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    def load_model(self):
        self.model = TemporalFusionTransformer.load_from_checkpoint(self.model_path)


    def predict(self):
        raw_predictions, x = self.model.predict(self.dataloader, mode="raw", return_x=True)
         
         # 평균 + 예측분산으로 오차 추정 (QuantileLoss 기반)
        means = raw_predictions["prediction"].mean(1).detach().cpu().numpy()
        p10 = raw_predictions["prediction"][:, :, 0].detach().cpu().numpy()
        p90 = raw_predictions["prediction"][:, :, -1].detach().cpu().numpy()
        time_idxs = x["decoder_time_idx"].detach().cpu().numpy()

        print("decoder_cat shape:", x["decoder_cat"].shape)  

        menu_ids = x["decoder_cat"][:, 0, 0].detach().cpu().numpy()
        flat_menu_ids = np.repeat(menu_ids, means.shape[1])

        print("flat_menu_ids:", flat_menu_ids.shape)
        print("time_idxs:", time_idxs.flatten().shape)
        print("means:", means.flatten().shape)

        self.result_df = pd.DataFrame({
            "menu_id": flat_menu_ids,
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
        self.result_df = self.result_df.dropna(subset=["date", "menu"])

    def plot_forecast(self):
        import os
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # predictor.py가 있는 디렉토리
        RESULT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
        os.makedirs(RESULT_DIR, exist_ok=True)

        for menu in self.result_df["menu"].unique():
            df_m = self.result_df[self.result_df["menu"] == menu]
            plt.figure(figsize=(10, 4))
            plt.plot(df_m["date"], df_m["predicted_quantity"], label="예측", marker="o")
            plt.fill_between(df_m["date"], df_m["p10"], df_m["p90"], alpha=0.3, label="오차범위")
            plt.title(f"{menu}의 예측 판매량")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(RESULT_DIR, f"{menu}_forecast.png"))
            plt.close()

    def save_predictions(self):
        import os

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        RESULT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
        os.makedirs(RESULT_DIR, exist_ok=True)

        self.result_df.to_csv(os.path.join(RESULT_DIR, "next_week_predictions.csv"), index=False)