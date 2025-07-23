import csv
from datetime import datetime
import os

LOG_FILE = "news_predictions_log.csv"

def log_prediction(entry: dict):
    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=entry.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(entry)
