# models/predictor.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from torch.utils.data import DataLoader
from pytorch_forecasting.data import NaNLabelEncoder
import os

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

        # 1. dataloader 진단
        print("📦 Checking dataloader...")
        print("len(dataloader):", len(self.dataloader))
            
        for i, batch in enumerate(self.dataloader):
            x = batch[0]  # ← 예측할 입력 데이터
            print(f"Batch {i} keys:", x.keys())
            break

        # 2. raw_data 내 예측용 row 존재 여부
        print("📅 Checking future data...")
        print("Max time_idx in dataset:", self.raw_data["time_idx"].max())
        print("Example future rows:")
        print(self.raw_data[self.raw_data["sold_quantity"] == 0].head())

        # 3. menu_id NaN 여부
        print("🧪 Checking menu_id in future:")
        future_rows = self.raw_data[self.raw_data["sold_quantity"] == 0]
        print("Any NaN in menu_id?", future_rows["menu_id"].isna().any())
        assert not future_rows["menu_id"].isna().any(), "❗ menu_id 매핑 실패: future 데이터에 메뉴 이름 누락 가능성"


        raw_predictions, x = self.model.predict(self.dataloader, mode="raw", return_x=True)
         
         # 평균 + 예측분산으로 오차 추정 (QuantileLoss 기반)
        means = raw_predictions["prediction"].mean(1).detach().cpu().numpy()
        p10 = raw_predictions["prediction"][:, :, 0].detach().cpu().numpy()
        p90 = raw_predictions["prediction"][:, :, -1].detach().cpu().numpy()
        time_idxs = x["decoder_time_idx"].detach().cpu().numpy()

        print("decoder_cat shape:", x["decoder_cat"].shape)  

        menu_ids = x["decoder_cat"][:, 0, 0].detach().cpu().numpy()
        flat_menu_ids = np.repeat(menu_ids, means.shape[1]).astype(int)

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

        self.raw_data["menu_id"] = self.raw_data["menu_id"].astype(int)
        menu_map = dict(zip(self.raw_data["menu_id"].astype(int), self.raw_data["menu"]))
        time_map = self.raw_data[["time_idx", "date"]].drop_duplicates().set_index("time_idx")["date"].to_dict()
        
        self.result_df["menu"] = self.result_df["menu_id"].map(menu_map)
        self.result_df["date"] = self.result_df["time_idx"].map(time_map)
        self.result_df["menu_id"] = self.result_df["menu_id"].astype(int)
        self.result_df = self.result_df.dropna(subset=["date", "menu"])


        print("menu isna count:", self.result_df["menu"].isna().sum())
        print("date isna count:", self.result_df["date"].isna().sum())
        
        print("📦 result_df preview:")
        print(self.result_df.head())

        print("메뉴 값 분포:")
        print(self.result_df["menu"].value_counts(dropna=False))

        print("▶ raw_predictions['prediction'].shape:", raw_predictions["prediction"].shape)

        # 5. 미래 기간별로 예측값 분기 저장
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        RESULT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
        os.makedirs(RESULT_DIR, exist_ok=True)

        # 🔍 미래 예측 대상 날짜만 추출
        # future_dates = self.raw_data[self.raw_data["sold_quantity"] == 0]["date"].unique()
        # df_future = self.result_df[self.result_df["date"].isin(future_dates)]
        future_dates = pd.to_datetime(self.raw_data[self.raw_data["sold_quantity"] == 0]["date"]).dt.date
        df_future = self.result_df[self.result_df["date"].dt.date.isin(future_dates)]
        
        df_future = df_future.sort_values("date")

        # 기준 날짜: 예측 시작일
        start_date = df_future["date"].min()

        # 다음주
        df_week = df_future[df_future["date"] <= start_date + pd.Timedelta(days=6)]
        df_week.to_csv(os.path.join(RESULT_DIR, "next_week_only.csv"), index=False)
        df_week.pivot_table(index="date", columns="menu", values="predicted_quantity", aggfunc="sum").to_csv(
            os.path.join(RESULT_DIR, "next_week_summary.csv")
        )
        output_path = os.path.abspath("./results/test_save.csv")
        print("✅ Writing to:", output_path)

        df_week.to_csv(output_path, index=False)

        # 바로 읽어서 내용 확인
        if os.path.exists(output_path):
            print("✅ File created. Preview:")
            preview = pd.read_csv(output_path)
            print(preview.head())
        else:
            print("❌ File not found.")

        # 다음달
        df_month = df_future[df_future["date"] <= start_date + pd.Timedelta(days=29)]
        df_month.to_csv(os.path.join(RESULT_DIR, "next_month_only.csv"), index=False)
        df_month.pivot_table(index="date", columns="menu", values="predicted_quantity", aggfunc="sum").to_csv(
            os.path.join(RESULT_DIR, "next_month_summary.csv")
        )

        # 다음해
        df_year = df_future[df_future["date"] <= start_date + pd.Timedelta(days=364)]
        df_year.to_csv(os.path.join(RESULT_DIR, "next_year_only.csv"), index=False)
        df_year.pivot_table(index="date", columns="menu", values="predicted_quantity", aggfunc="sum").to_csv(
            os.path.join(RESULT_DIR, "next_year_summary.csv")
        )

        print("🔍 df_future preview:")
        print(df_future.head())
        print("df_future shape:", df_future.shape)
        print("df_future date range:", df_future["date"].min(), "~", df_future["date"].max())

        print("📊 예측 결과 요약:")
        print(df_week.pivot_table(index="date", columns="menu", values="predicted_quantity", aggfunc="sum"))

        # future_dates = self.raw_data[self.raw_data["sold_quantity"] == 0]["date"].unique()
        # df_future = self.result_df[self.result_df["date"].isin(future_dates)]

        # df_future.to_csv("./results/next_week_only.csv", index=False)

        
        # pivot = df_future.pivot_table(
        #     index="date",
        #     columns="menu",
        #     values="predicted_quantity",
        #     aggfunc="sum"
        # )
        # pivot.to_csv("./results/next_week_summary.csv")
        # print("📊 예측 결과 요약:")
        # print(pivot)

    def plot_forecast(self):
        
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

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        RESULT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "results"))
        os.makedirs(RESULT_DIR, exist_ok=True)

        self.result_df.to_csv(os.path.join(RESULT_DIR, "next_week_predictions.csv"), index=False)

    def summarize_by_weekday(self):
        df = self.result_df.copy()

        print("🔍 Summarizing... result_df shape:", df.shape)
        print("menu unique:", df["menu"].unique())

        df = df.dropna(subset=["menu", "date", "predicted_quantity"])
        df["weekday"] = pd.to_datetime(df["date"]).dt.day_name()
        df["weekday"] = pd.Categorical(
            df["weekday"],
            categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            ordered=True
        )

        pivot = df.pivot_table(
            index="weekday",
            columns="menu",
            values="predicted_quantity",
            aggfunc="sum",
            fill_value=0
        )

        pivot["Total"] = pivot.sum(axis=1)
        pivot = pivot.sort_index()

        print("📊 요약표:")
        print(pivot)

        pivot.to_csv("../results/weekday_menu_summary.csv")

