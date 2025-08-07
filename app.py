import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import json
import re
from collections import deque
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# ----------- 配置 ----------- #
transaction_fee_rate = 0.001
short_term_penalty = -0.001
DAILY_DECISION_LOG_DIR = "logs"
PORTFOLIO_DIR = "portfolios"
MODEL_DIR = "models"
os.makedirs(DAILY_DECISION_LOG_DIR, exist_ok=True)
os.makedirs(PORTFOLIO_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ----------- Streamlit UI ----------- #
st.set_page_config(page_title="RL 策略分析", layout="centered")
st.title("🧠 强化学习股票策略助手")
with st.form("setup"):
    symbol = st.text_input("输入股票代码", "QQQ").upper()
    initial_cash = st.number_input("初始投资金额（美元）", value=10000)
    retrain = st.checkbox("是否重新训练模型", value=False)
    submitted = st.form_submit_button("开始分析")

# ----------- 模型类 ----------- #
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class RLAgent:
    def __init__(self, state_size, model_path):
        self.model_path = model_path
        self.state_size = state_size
        self.action_size = 2
        self.memory = deque(maxlen=1000)
        self.model = DQN(state_size, self.action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        if os.path.exists(model_path) and not retrain:
            self.model.load_state_dict(torch.load(model_path))

    def act(self, state):
        with torch.no_grad():
            q = self.model(torch.FloatTensor(state))
        return torch.argmax(q).item()

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        for s, a, r, s_next in batch:
            q_update = r + 0.95 * torch.max(self.model(torch.FloatTensor(s_next))).item()
            q_values = self.model(torch.FloatTensor(s))
            q_values[a] = q_update
            loss = self.criterion(self.model(torch.FloatTensor(s)), q_values.detach())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def save(self):
        torch.save(self.model.state_dict(), self.model_path)

# ----------- 工具函数 ----------- #
def load_data(symbol):
    df = yf.download(symbol, start="2000-01-01")
    vix = yf.download("^VIX", start="2000-01-01")[["Close"]]
    vix.columns = ["VIX"]
    df = df.join(vix, how="inner")
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_features(df):
    close = df['Close']
    df['return_5'] = close.pct_change(5)
    df['rsi'] = compute_rsi(close)
    df['ma10'] = close.rolling(10).mean()
    df['ma30'] = close.rolling(30).mean()
    df['ma_ratio'] = df['ma10'] / df['ma30'] - 1
    df['volatility'] = close.rolling(10).std()
    df['volume_change'] = df['Volume'].pct_change()
    df['ema20'] = close.ewm(span=20).mean()
    df['ema50'] = close.ewm(span=50).mean()
    df['ema_diff'] = df['ema20'] - df['ema50']
    df['price_ema_ratio'] = close / df['ema20'] - 1
    df['bb_upper'] = df['ma10'] + 2 * df['volatility']
    df['bb_lower'] = df['ma10'] - 2 * df['volatility']
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['ma10']
    df['vix_change'] = df['VIX'].pct_change()
    df['future_return'] = close.shift(-5) / close - 1
    return df.dropna()

def portfolio_path(symbol):
    return os.path.join(PORTFOLIO_DIR, f"{symbol}_portfolio.json")

def decision_log_path(symbol):
    return os.path.join(DAILY_DECISION_LOG_DIR, f"{symbol}_log.csv")

def load_portfolio(symbol):
    path = portfolio_path(symbol)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {"cash": initial_cash, "shares": 0, "last_action": None}

def save_portfolio(symbol, state):
    with open(portfolio_path(symbol), 'w') as f:
        json.dump(state, f)

# ----------- 执行逻辑 ----------- #
if submitted:
    df = load_data(symbol)
    df = compute_features(df)
    features = ['rsi', 'ma_ratio', 'volatility', 'volume_change', 'ema_diff',
                'price_ema_ratio', 'bb_width', 'vix_change']
    df = df.dropna()
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    agent = RLAgent(len(features), os.path.join(MODEL_DIR, f"{symbol}_model.pth"))

    today, tomorrow = df.index[-2], df.index[-1]
    state = df.loc[today, features].values
    next_state = df.loc[tomorrow, features].values
    price = df.loc[today, 'Close']
    next_price = df.loc[tomorrow, 'Close']

    portfolio = load_portfolio(symbol)
    cash, shares, last_action = portfolio['cash'], portfolio['shares'], portfolio['last_action']
    action = agent.act(state)
    action_str = "HOLD"
    reward = 0

    if action == 1 and shares == 0:
        shares = int(cash // price)
        if shares > 0:
            cash -= shares * price * (1 + transaction_fee_rate)
            action_str = f"BUY @{price:.2f}"
            if last_action == "SELL":
                reward += short_term_penalty
            last_action = "BUY"
    elif action == 0 and shares > 0:
        cash += shares * price * (1 - transaction_fee_rate)
        action_str = f"SELL @{price:.2f}"
        if last_action == "BUY":
            reward += short_term_penalty
        shares = 0
        last_action = "SELL"

    total_value = cash + shares * next_price
    reward += (next_price - price) / price if shares > 0 else 0
    agent.remember(state, action, reward, next_state)
    agent.replay()
    agent.save()
    save_portfolio(symbol, {"cash": cash, "shares": shares, "last_action": last_action})

    log_path = decision_log_path(symbol)
    new_log = pd.DataFrame([[today.strftime("%Y-%m-%d"), action_str, cash, shares, total_value]],
                           columns=["Date", "Action", "Cash", "Shares", "Portfolio"])
    if os.path.exists(log_path):
        new_log.to_csv(log_path, mode='a', header=False, index=False)
    else:
        new_log.to_csv(log_path, index=False)

    # ----------- 展示结果 ----------- #
    st.subheader(f"📈 {symbol} 当前策略建议")
    st.markdown(f"""
    - 今日操作建议: **{action_str}**
    - 当前股价: **${price:.2f}**
    - 当前持仓: **{shares} 股**
    - 剩余现金: **${cash:,.2f}**
    - 总资产: **${total_value:,.2f}**
    """)
