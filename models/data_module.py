# models/data_module.py

import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder

class TFTDataModule:
    def __init__(self, path: str, max_encoder_length: int = 30, max_prediction_length: int = 7):
        self.path = path
        self.max_encoder_length = max_encoder_length
        self.max_prediction_length = max_prediction_length
        self.data = None
        self.dataset = None
        self.dataloader = None

    def load_and_preprocess(self):
        df = pd.read_csv(self.path)
        df['time_idx'] = pd.to_datetime(df['timestamp'])
        df['data'] = df['timestamp'].dt.date

        daily = (
            df.groupby(['data', 'menu'])
            .size()
            .reset_index(name='sold_quantity')
        )
        daily['data'] = pd.to_datetime(daily['data'])
        daily = daily.sort_values('data')
        daily['time_idx'] = (daily['data'] - daily['data'].min()).dt.days
        daily['menu_id'] = daily['menu'].astype(('category')).cat.codes

        slef.data = daily
        
    def setup(self):
        self.dataset = TimeSeriesDataSet(
            self.data,
            time_indx='time_idx',
            target='sold_quantity',
            group_ids=['menu_id'],
            max_encdoer_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            time_varying_known_reals=['time_idx'],
            time_varying_unkown_reals=['sold_quantity'],
            categorical_encoders={'menu_id': NaNLabelEncoder().fit(self.data.menu_id),}
        )
    def get_dataloader(self, batch_size: int = 64, num_workers: int = 0):
        self.dataloader = self.dataset.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
        return self.dataloader
    