# models/data_module.py

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder

class TFTDataModule:
    def __init__(self, path: str, max_encoder_length: int = 30, max_prediction_length: int = 30):
        self.path = path
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.data = None
        self.dataset = None
        self.dataloader = None

    def load_and_preprocess(self, and_future_days: int = 30):
        df = pd.read_excel(self.path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['data'] = df['timestamp'].dt.date

        daily = (
            df.groupby(['data', 'menu'])
            .size()
            .reset_index(name='sold_quantity')
        )
        daily['data'] = pd.to_datetime(daily['data'])
        daily = daily.sort_values('data')

        daily['menu_id'] = daily['menu'].astype(('category')).cat.codes
        menu_list = daily['menu'].unique()
        menu_id_map = dict(zip(menu_list, daily['menu_id'].unique()))


        last_date = daily['data'].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, add_future_days + 1)]
        

        # 4. 미래 데이터 생성 (모든 메뉴에 대해)
        future_rows = pd.DataFrame(
            [(d, m) for d in future_dates for m in menu_list],
            columns=['data', 'menu']
        )
        
        future_rows['sold_quantity'] = 0
        future_rows['data'] = pd.to_datetime(future_rows['data'])
        future_rows['menu_id'] = future_rows['menu'].map(menu_id_map)

        # 5. 과거+미래 데이터 병합
        full_data = pd.concat([daily, future_rows], ignore_index=True)
        full_data = full_data.sort_values(['menu_id', 'data'])
        full_data['time_idx'] = (full_data['data'] - full_data['data'].min()).dt.days



        self.data = daily
        
    def setup(self):
        self.dataset = TimeSeriesDataSet(
            self.data,
            time_idx='time_idx',
            target='sold_quantity',
            group_ids=['menu_id'],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=['time_idx'],
            time_varying_unknown_reals=['sold_quantity'],
            categorical_encoders={'menu_id': NaNLabelEncoder().fit(self.data.menu_id)},
            allow_missing_timesteps=True # 비어날 수 있는 결측치 허용 (비연속시계열 허용)
        )
    def get_dataloader(self, batch_size: int = 64, num_workers: int = 0):
        self.dataloader = self.dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
        return self.dataloader
    