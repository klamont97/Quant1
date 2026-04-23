import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

st.title("📈 Next-Day Direction Model")

ticker = st.text_input("Enter Ticker", "AAPL")

if st.button("Run Model"):

    df = yf.download(ticker, start="2015-01-01").dropna()

    # Features
    df['ret_1'] = np.log(df['Close'] / df['Close'].shift(1))
    df['ret_5'] = np.log(df['Close'] / df['Close'].shift(5))
    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_z'] = (df['vol_20'] - df['vol_20'].rolling(252).mean()) / df['vol_20'].rolling(252).std()
    df['range_asym'] = ((df['High'] - df['Close']) - (df['Close'] - df['Low'])) / (df['High'] - df['Low'])
    df['vol_surge'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['accel'] = df['ret_1'].rolling(5).mean() - df['ret_1'].rolling(10).mean()

    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()

    features = ['ret_5', 'vol_z', 'range_asym', 'vol_surge', 'accel']

    split = int(len(df)*0.8)

    train = df.iloc[:split]
    test = df.iloc[split:]

    X_train = train[features]
    y_train = train['target']

    X_test = test[features]
    y_test = test['target']

    model = XGBClassifier(eval_metric='logloss')
    model.fit(X_train, y_train)

    latest = df[features].iloc[-1:].values
    prob = model.predict_proba(latest)[0][1]

    st.metric("Probability of Up Move", f"{prob:.2%}")

    if prob > 0.55:
        st.success("📈 Model Bias: UP")
    elif prob < 0.45:
        st.error("📉 Model Bias: DOWN")
    else:
        st.warning("⚖️ No strong signal")