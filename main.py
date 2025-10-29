# main.py
import yaml
from src.data.data_loader import download_ohlcv
from src.features.feature_engineering import add_technical_indicators
import logging
import os

# Ensure logs/ exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='logs/run.log',   # log file
    filemode='a',              # append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO         # capture INFO and above
)

# Example usage
logging.info("Pipeline started")
logging.info("Data download complete")
logging.warning("Missing values found in the dataset")
logging.error("Model training failed")  # if an exception occurs

def run_pipeline(cfg_path='config.yaml'):
    with open(cfg_path,'r') as f:
        cfg = yaml.safe_load(f)
    t = cfg['data']['tickers'][0]
    df = download_ohlcv(t, cfg['data']['start_date'], cfg['data']['end_date'])
    df = add_technical_indicators(df)
    print(df.tail())

if __name__ == '__main__':
    run_pipeline()
