# models/train.py

from pytorch_lightning import seed_everything
from data_module import TFTDataModule
from tft_trainer import TFTTrainer

if __name__ == "__main__":
    seed_everything(42)

    data_module = TFTDataModule(path="../data/sample_sales_202408_202504.xlsx")
    data_module.load_and_preprocess()
    data_module.setup()
    dataloader = data_module.get_dataloader()

    trainer = TFTTrainer(dataset=data_module.dataset)
    trainer.build_model()
    trainer.train(dataloader)
    trainer.save("../models/tft_model.ckpt")
    print("Training complete. Model saved.")
    