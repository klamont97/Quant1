import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

st.title("📈 Next-Day Direction Model")

ticker = st.text_input("Enter Ticker", "AAPL")

@st.cache_data
def load_data(ticker):
    return yf.download(ticker, start="2015-01-01").dropna()

if st.button("Run Model"):

    df = load_data(ticker)

    # --- Feature Engineering ---
    df['ret_1'] = np.log(df['Close'] / df['Close'].shift(1))
    df['ret_5'] = np.log(df['Close'] / df['Close'].shift(5))
    df['vol_20'] = df['ret_1'].rolling(20).std()
    df['vol_z'] = (df['vol_20'] - df['vol_20'].rolling(252).mean()) / df['vol_20'].rolling(252).std()
    df['range_asym'] = ((df['High'] - df['Close']) - (df['Close'] - df['Low'])) / (df['High'] - df['Low'])
    df['vol_surge'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['accel'] = df['ret_1'].rolling(5).mean() - df['ret_1'].rolling(10).mean()

    # Target
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

    df = df.dropna()

    features = ['ret_5', 'vol_z', 'range_asym', 'vol_surge', 'accel']

    # --- Train/Test Split ---
    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    X_train = train[features]
    y_train = train['target']

    X_test = test[features]
    y_test = test['target']

    # --- Model ---
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # --- Live Prediction ---
    latest = df[features].iloc[-1:].values
    prob = model.predict_proba(latest)[0][1]

    st.metric("Probability of Up Move", f"{prob:.2%}")

    if prob > 0.55:
        st.success("📈 Model Bias: UP")
    elif prob < 0.45:
        st.error("📉 Model Bias: DOWN")
    else:
        st.warning("⚖️ No strong signal")

    # =========================
    # 📊 VALIDATION SECTION
    # =========================

    st.write("---")
    st.header("📊 Model Validation")

    # --- Accuracy ---
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    st.write(f"**Backtest Accuracy:** {accuracy:.2%}")

    # --- Strategy Simulation ---
    test['pred'] = preds
    test['position'] = test['pred'] * 2 - 1  # +1 or -1

    test['strategy_ret'] = test['position'] * test['ret_1']
    test['cum_strategy'] = (1 + test['strategy_ret']).cumprod()
    test['cum_market'] = (1 + test['ret_1']).cumprod()

    st.subheader("📈 Strategy vs Market")
    st.line_chart(test[['cum_strategy', 'cum_market']])

    # --- Transaction Costs ---
    cost = 0.001  # 0.1%
    test['trade'] = test['position'].diff().abs()
    test['cost'] = test['trade'] * cost

    test['net_ret'] = test['strategy_ret'] - test['cost']
    test['cum_net'] = (1 + test['net_ret']).cumprod()

    st.subheader("💰 After Costs")
    st.line_chart(test[['cum_net', 'cum_market']])

    # --- Calibration ---
    probs = model.predict_proba(X_test)[:, 1]
    test['prob'] = probs

    test['bucket'] = pd.cut(test['prob'], bins=5)
    calibration = test.groupby('bucket')['target'].mean()

    st.subheader("🔍 Probability Calibration")
    st.write(calibration)