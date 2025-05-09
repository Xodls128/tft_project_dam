# predict.py

from models.predictor import Predictor
from visualization.data_visualization import plot_next_week_prediction

if __name__ == "__main__":
    p = Predictor(
        model_path="./models/tft_model.ckpt",
        data_path="./data/sample_sales_202408_202504.xlsx"
    )
    p.load_data()
    p.setup_dataset()
    p.load_model()
    p.predict()
    p.plot_forecast()
    p.save_predictions()
    print("Prediction complete. Results saved.")
    p.summarize_by_weekday()

    plot_next_week_prediction()