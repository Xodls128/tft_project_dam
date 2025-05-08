# models/tft_trainer.py

import torch
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_lightning import Trainer
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

class TFTTrainer:
    def __init__(self, dataset: TimeSeriesDataSet):
        self.dataset = dataset
        self.model = None
        self.trainer = None

    def build_model(self):
        self.model = TemporalFusionTransformer.from_dataset(
            self.dataset,
            learning_rate=0.03,
            hidden_size=16,
            attention_head_size=1,
            dropout=0.1,
            loss=torch.nn.L1Loss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

    def train(self, dataloader: DataLoader, max_epochs: int = 5):
        self.trainer = Trainer(max_epochs=max_epochs, gradient_clip_val=0.1)
        self.trainer.fit(self.model, train_dataloaders=dataloader)

    def save(self, path: str):
        self.trainer.save_checkpoint(path)
        print(f"Model saved to {path}")
