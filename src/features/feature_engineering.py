# src/features/feature_engineering.py
import pandas as pd

def add_technical_indicators(df):
    df = df.copy()
    df['return_1'] = df['Close'].pct_change()
    df['ma_5'] = df['Close'].rolling(5).mean()
    df['ma_10'] = df['Close'].rolling(10).mean()
    df['vol_10'] = df['Close'].rolling(10).std()
    return df

def compute_sentiment_surprise(sentiment_series, window=7):
    roll = sentiment_series.rolling(window).mean()
    surprise = sentiment_series - roll
    return surprise
