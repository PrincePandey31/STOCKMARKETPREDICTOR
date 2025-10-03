import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def preprocess_data(df):
   
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

    if "Date" not in df.columns and df.index.name is not None:
        df = df.reset_index()

    df.columns = [str(c).strip().lower() for c in df.columns]

    df.columns = [c.split("_")[0] if "_" in c else c for c in df.columns]

    rename_map = {
        "date": "Date",
        "timestamp": "Date",
        "close/last": "Close",
        "adj close": "Close",
        "close": "Close",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "volume": "Volume"
    }
    df.rename(columns=rename_map, inplace=True)

    required = ["Date", "Close", "Open", "High", "Low", "Volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

    df = df[required]

    for col in ["Close", "Open", "High", "Low", "Volume"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)   
            .str.replace(" ", "", regex=True)       
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)
    df.sort_values("Date", inplace=True)

    df["Return"] = df["Close"].pct_change()
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["Volatility"] = df["Return"].rolling(5).std()
    df.dropna(inplace=True)

    return df

def train_model(df):
    features = ["Open", "High", "Low", "Volume", "MA5", "MA10", "Volatility"]
    X = df[features]
    y = df["Close"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_scaled, y)

    preds = model.predict(X_scaled)

    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    df["Predicted"] = preds

    return model, scaler, features, rmse, r2, df

st.set_page_config(page_title="📈 Stock Price Prediction", layout="wide")

st.title("📈 Stock Price Prediction App")
st.write("Upload a CSV or fetch stock data online to train and predict closing prices.")

mode = st.radio("Choose Data Source:", ["Upload CSV", "Fetch Online (Yahoo Finance)"])

if mode == "Upload CSV":
    file = st.file_uploader("Upload your CSV file", type=["csv"])
    if file:
        try:
            raw_df = pd.read_csv(file)
            df = preprocess_data(raw_df)
            model, scaler, features, rmse, r2, df = train_model(df)

            st.success("✅ Model trained successfully on uploaded CSV")
            st.write(f"**RMSE:** {rmse:.2f} | **R²:** {r2:.2f}")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["Date"], df["Close"], label="Actual")
            ax.plot(df["Date"], df["Predicted"], label="Predicted")
            ax.legend()
            st.pyplot(fig)

            last_actual = float(df["Close"].iloc[-1])
            last_pred = float(df["Predicted"].iloc[-1])
            st.write(f"📊 Last Actual Price: **{last_actual:.2f}**")
            st.write(f"🤖 Last Predicted Price: **{last_pred:.2f}**")

        except Exception as e:
            st.error(f"❌ Error processing CSV: {e}")

elif mode == "Fetch Online (Yahoo Finance)":
    ticker = st.text_input("Enter Stock Ticker (e.g. AAPL, MSFT, TSLA):", "AAPL")

    if st.button("Fetch & Train"):
        try:
            raw_df = yf.download(ticker, period="5y")
            df = preprocess_data(raw_df)
            model, scaler, features, rmse, r2, df = train_model(df)

            st.success(f"✅ Model trained successfully on {ticker} (Yahoo Finance)")
            st.write(f"**RMSE:** {rmse:.2f} | **R²:** {r2:.2f}")

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df["Date"], df["Close"], label="Actual")
            ax.plot(df["Date"], df["Predicted"], label="Predicted")
            ax.legend()
            st.pyplot(fig)

            last_actual = float(df["Close"].iloc[-1])
            last_pred = float(df["Predicted"].iloc[-1])
            st.write(f"📊 Last Actual Price: **{last_actual:.2f}**")
            st.write(f"🤖 Last Predicted Price: **{last_pred:.2f}**")

        except Exception as e:
            st.error(f"❌ Error processing data: {e}")





