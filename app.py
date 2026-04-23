import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(layout="wide")

st.title("📈 Next-Day Direction Model (Validated)")

ticker = st.text_input("Enter Ticker", "AAPL")
confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.7, 0.55)

@st.cache_data
def load_data(ticker):
    return yf.download(ticker, start="2015-01-01").dropna()

if st.button("Run Model"):

    df = load_data(ticker)

    # =========================
    # FEATURE ENGINEERING
    # =========================
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

    # =========================
    # TRAIN / TEST SPLIT
    # =========================
    split = int(len(df) * 0.8)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()

    X_train = train[features]
    y_train = train['target']

    X_test = test[features]
    y_test = test['target']

    # =========================
    # MODEL
    # =========================
    model = XGBClassifier(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # =========================
    # LIVE PREDICTION
    # =========================
    latest = df[features].iloc[-1:].values
    prob = model.predict_proba(latest)[0][1]

    st.metric("📊 Probability of Up Move", f"{prob:.2%}")

    if prob > confidence_threshold:
        st.success("📈 Model Bias: UP")
    elif prob < (1 - confidence_threshold):
        st.error("📉 Model Bias: DOWN")
    else:
        st.warning("⚖️ No Trade Zone")

    # =========================
    # VALIDATION
    # =========================
    st.write("---")
    st.header("📊 Model Validation")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, preds)
    st.write(f"**Backtest Accuracy:** {accuracy:.2%}")

    # =========================
    # STRATEGY SIMULATION
    # =========================
    test = test.copy()

    test['prob'] = probs
    test['pred'] = preds

    # Apply confidence filter
    test['position'] = 0
    test.loc[test['prob'] > confidence_threshold, 'position'] = 1
    test.loc[test['prob'] < (1 - confidence_threshold), 'position'] = -1

    # Returns
    test['strategy_ret'] = test['position'] * test['ret_1']

    test['strategy_ret'] = test['strategy_ret'].fillna(0)
    test['ret_1'] = test['ret_1'].fillna(0)

    # Correct cumulative returns (log → exp)
    test['cum_strategy'] = np.exp(test['strategy_ret'].cumsum())
    test['cum_market'] = np.exp(test['ret_1'].cumsum())

    # Plot
    st.subheader("📈 Strategy vs Market")
    st.line_chart(test[['cum_strategy', 'cum_market']].dropna())

    # =========================
    # COSTS
    # =========================
    cost = 0.001  # 10bps

    test['trade'] = test['position'].diff().abs()
    test['cost'] = test['trade'] * cost

    test['net_ret'] = test['strategy_ret'] - test['cost']
    test['cum_net'] = np.exp(test['net_ret'].cumsum())

    st.subheader("💰 After Costs")
    st.line_chart(test[['cum_net', 'cum_market']].dropna())

    # =========================
    # CALIBRATION
    # =========================
    test['bucket'] = pd.cut(test['prob'], bins=5)
    calibration = test.groupby('bucket')['target'].mean()

    st.subheader("🔍 Probability Calibration")
    st.write(calibration)

    # =========================
    # TRADE STATS
    # =========================
    trades = (test['position'] != 0).sum()
    st.write(f"📊 Number of Trades: {trades}")

    hit_rate = test[test['position'] != 0]['target'].mean()
    st.write(f"🎯 Hit Rate (Filtered Trades): {hit_rate:.2%}")