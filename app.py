import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import os
import logging
import tweepy
from textblob import TextBlob
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import mlflow
import joblib

# -------------------------------
# 1️⃣ Logging setup
# -------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename='logs/run.log', level=logging.INFO)
logging.info("Stock Prediction App started with Twitter sentiment.")

# -------------------------------
# 2️⃣ Streamlit UI
# -------------------------------
st.title("📈 AI-Based Stock Prediction App with Twitter Sentiment 🐦")
st.write("Fetch stock data, analyze Twitter sentiment, train a model, and predict trends.")

ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, AAPL):", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2018-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))
run_button = st.button("🚀 Fetch Data & Train Model")

# -------------------------------
# 3️⃣ Twitter API setup
# -------------------------------
# Replace this with your actual bearer token from Twitter Developer Portal
BEARER_TOKEN = "YOUR_TWITTER_BEARER_TOKEN"

def fetch_twitter_sentiment(query, max_results=50):
    """Fetch tweets and compute average sentiment."""
    try:
        client = tweepy.Client(bearer_token=BEARER_TOKEN)
        tweets = client.search_recent_tweets(query=query, max_results=max_results)
        sentiments = []
        if tweets.data:
            for tweet in tweets.data:
                sentiment = TextBlob(tweet.text).sentiment.polarity
                sentiments.append(sentiment)
            avg_sentiment = np.mean(sentiments)
        else:
            avg_sentiment = 0.0
        return avg_sentiment
    except Exception as e:
        logging.error(f"Twitter fetch failed: {e}")
        return 0.0

# -------------------------------
# 4️⃣ Main Logic
# -------------------------------
if run_button:
    st.info(f"Fetching stock data for {ticker} ...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

    if df.empty:
        st.error("No stock data found for this ticker/date range.")
        st.stop()

    df.reset_index(inplace=True)
    st.success("✅ Stock data fetched successfully!")
    st.dataframe(df.head())

    # Detect close column safely
    close_col = None
    for col in df.columns:
        if isinstance(col, str) and "close" in col.lower():
            close_col = col
            break
    if close_col is None:
        st.warning("No 'Close' column found, using synthetic close prices.")
        df["Close"] = np.random.rand(len(df)) * 100
        close_col = "Close"

    # -------------------------------
    # 5️⃣ Feature Engineering
    # -------------------------------
    st.info("Computing features and sentiment...")

    df["ma_5"] = df[close_col].rolling(window=5, min_periods=1).mean()
    df["ma_10"] = df[close_col].rolling(window=10, min_periods=1).mean()
    df["return_1"] = df[close_col].pct_change().fillna(0)

    # -------------------------------
    # 6️⃣ Fetch Twitter Sentiment
    # -------------------------------
    st.info("Fetching Twitter sentiment (this may take a few seconds)...")
    sentiment_query = f"{ticker} stock -is:retweet lang:en"
    avg_sentiment = fetch_twitter_sentiment(sentiment_query)
    df["sentiment"] = avg_sentiment  # add sentiment as a constant feature
    st.success(f"🐦 Avg Sentiment Score: {avg_sentiment:.3f}")

    # -------------------------------
    # 7️⃣ Handle Missing Data Safely
    # -------------------------------
    features = ["ma_5", "ma_10", "return_1", "sentiment"]
    df = df.dropna(subset=["ma_5", "ma_10", "return_1"], how="all")

    if df.empty:
        st.error("Not enough data after feature creation.")
        st.stop()

    # -------------------------------
    # 8️⃣ Model Preparation
    # -------------------------------
    df["target"] = (df[close_col].shift(-1) > df[close_col]).astype(int)
    X = df[features].iloc[:-1]
    y = df["target"].iloc[:-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # -------------------------------
    # 9️⃣ Train Model
    # -------------------------------
    st.info("Training model...")
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.success(f"✅ Training complete! Model accuracy: {accuracy:.2f}")

    # -------------------------------
    # 🔟 MLflow Logging
    # -------------------------------
    try:
        mlflow.start_run()
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("model", "HistGradientBoostingClassifier")
        mlflow.log_param("sentiment_avg", avg_sentiment)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.end_run()
        st.info("✅ MLflow metrics logged.")
    except Exception as e:
        st.warning(f"MLflow logging skipped: {e}")

    # -------------------------------
    # 11️⃣ Save Model & Data
    # -------------------------------
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{ticker.replace('.', '_')}_model.pkl"
    joblib.dump(model, model_path)
    st.success(f"💾 Model saved at `{model_path}`")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(f"data/processed/{ticker.replace('.', '_')}_processed.csv", index=False)
    st.info("📁 Processed data saved.")

    # -------------------------------
    # 12️⃣ Predict Next Day Trend
    # -------------------------------
    latest = df[features].iloc[-1].values.reshape(1, -1)
    pred = model.predict(latest)[0]
    result = "📈 UP" if pred == 1 else "📉 DOWN"

    st.subheader("📊 Prediction for Next Day")
    st.write(f"**The stock is predicted to go:** {result}")
    