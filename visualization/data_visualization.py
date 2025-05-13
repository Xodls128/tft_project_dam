# visualization/data_visualization.py

from visualization.plot_week import plot_next_week_prediction
from visualization.plot_month import plot_next_month_prediction
from visualization.plot_year import plot_next_year_prediction

def run_all_plots():
    plot_next_week_prediction()
    plot_next_month_prediction()
    plot_next_year_prediction()
