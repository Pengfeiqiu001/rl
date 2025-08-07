# Streamlit 强化学习策略分析器（支持多股票、持仓追踪、每日更新）

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
from email.message import EmailMessage
import smtplib

transaction_fee_rate = 0.001
short_term_penalty = -0.001
MODEL_PATH = "model.pth"

st.set_page_config(page_title="RL 策略分析", layout="centered")
st.title("🧠 强化学习策略分析器")
st.caption("请输入你想分析的股票代码和初始投资金额")

with st.form(key="ticker_form"):
    symbol = st.text_input("输入股票代码（如 AAPL、MSFT、TSLA）:", "QQQ").upper()
    initial_cash = st.number_input("初始投资金额（美元）", value=10000, min_value=100)
    retrain = st.checkbox("重新训练强化学习模型", value=False)
    reset_portfolio = st.checkbox("重置投资组合", value=False)
    EMAIL_ALERT = st.checkbox("开启邮件提醒（需配置）", value=False)
    submitted = st.form_submit_button("开始分析")

if submitted:

    # 使用 symbol 创建独立状态路径
    PORTFOLIO_STATE = f"portfolio_state_{symbol}.json"
    DAILY_DECISION_LOG = f"daily_decision_log_{symbol}.csv"

    if reset_portfolio and os.path.exists(PORTFOLIO_STATE):
        os.remove(PORTFOLIO_STATE)

    def load_data(symbol):
        df = yf.download(symbol, start="2000-01-01", group_by='ticker')
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
        vix = yf.download("^VIX", start="2000-01-01")[["Close"]]
        vix.columns = ["VIX"]
        combined = df.join(vix, how="inner")
        combined.dropna(inplace=True)
        return combined

    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def compute_features(data):
        close_candidates = [col for col in data.columns if 'Close' in col]
        if not close_candidates:
            raise KeyError("No valid 'Close' column found.")
        close_col = close_candidates[0]
        close_price = data[close_col]

        data['return_5'] = close_price.pct_change(5)
        data['rsi'] = compute_rsi(close_price)
        data['ma10'] = close_price.rolling(10).mean()
        data['ma30'] = close_price.rolling(30).mean()
        data['ma_ratio'] = data['ma10'] / data['ma30'] - 1
        data['volatility'] = close_price.rolling(10).std()
        volume_col = next((col for col in data.columns if 'Volume' in col), 'Volume')
        data['volume_change'] = data[volume_col].pct_change()
        data['ema20'] = close_price.ewm(span=20).mean()
        data['ema50'] = close_price.ewm(span=50).mean()
        data['ema_diff'] = data['ema20'] - data['ema50']
        data['price_ema_ratio'] = close_price / data['ema20'] - 1
        data['bb_upper'] = data['ma10'] + 2 * data['volatility']
        data['bb_lower'] = data['ma10'] - 2 * data['volatility']
        data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['ma10']
        data['vix_change'] = data['VIX'].pct_change()
        data['future_return'] = close_price.shift(-5) / close_price - 1
        data['Close'] = close_price
        return data

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
        def __init__(self, state_size):
            self.state_size = state_size
            self.action_size = 2
            self.memory = deque(maxlen=1000)
            self.model = DQN(state_size, self.action_size)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.MSELoss()
            if os.path.exists(MODEL_PATH) and not retrain:
                self.model.load_state_dict(torch.load(MODEL_PATH))

        def act(self, state):
            with torch.no_grad():
                q_values = self.model(torch.FloatTensor(state))
            return torch.argmax(q_values).item()

        def remember(self, state, action, reward, next_state):
            self.memory.append((state, action, reward, next_state))

        def replay(self, batch_size=32):
            if len(self.memory) < batch_size:
                return
            batch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state in batch:
                q_update = reward + 0.95 * torch.max(self.model(torch.FloatTensor(next_state))).item()
                q_values = self.model(torch.FloatTensor(state))
                q_values[action] = q_update
                loss = self.criterion(self.model(torch.FloatTensor(state)), q_values.detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        def save_model(self):
            torch.save(self.model.state_dict(), MODEL_PATH)

    def save_portfolio_state(cash, shares, last_action):
        state = {"cash": cash, "shares": shares, "last_action": last_action}
        with open(PORTFOLIO_STATE, 'w') as f:
            json.dump(state, f)

    def load_portfolio_state():
        try:
            if os.path.exists(PORTFOLIO_STATE):
                with open(PORTFOLIO_STATE, 'r') as f:
                    state = json.load(f)
                if not all(k in state for k in ['cash', 'shares', 'last_action']):
                    raise ValueError("Invalid portfolio structure.")
                return state
        except:
            pass
        return {"cash": initial_cash, "shares": 0, "last_action": None}

    def evaluate_today(agent, all_data, features):
        today = all_data.index[-2]
        tomorrow = all_data.index[-1]
        state = all_data.loc[today, features].values
        next_state = all_data.loc[tomorrow, features].values
        price = all_data.loc[today, 'Close']
        next_price = all_data.loc[tomorrow, 'Close']

        portfolio = load_portfolio_state()
        cash, shares, last_action = portfolio['cash'], portfolio['shares'], portfolio['last_action']

        action = agent.act(state)
        action_str = ""
        reward = 0

        if action == 1 and shares == 0:
            shares = int(cash // price)
            cash -= shares * price * (1 + transaction_fee_rate)
            action_str = f"BUY @ ${price:.2f}"
            if last_action == 'SELL':
                reward += short_term_penalty
            last_action = 'BUY'
        elif action == 0 and shares > 0:
            cash += shares * price * (1 - transaction_fee_rate)
            action_str = f"SELL @ ${price:.2f}"
            if last_action == 'BUY':
                reward += short_term_penalty
            last_action = 'SELL'
            shares = 0

        total_value = cash + shares * next_price
        reward += (next_price - price) / price if shares > 0 else 0
        agent.remember(state, action, reward, next_state)
        agent.replay()

        save_portfolio_state(cash, shares, last_action)

        log = pd.DataFrame([[today.date(), action_str, cash, shares, total_value]],
                           columns=["Date", "Action", "Cash", "Shares", "Portfolio"])
        if os.path.exists(DAILY_DECISION_LOG):
            log.to_csv(DAILY_DECISION_LOG, mode='a', header=False, index=False)
        else:
            log.to_csv(DAILY_DECISION_LOG, index=False)

        return action_str, total_value

    # 主流程执行
    raw_data = load_data(symbol)
    all_data = compute_features(raw_data).dropna()
    features = ['rsi', 'ma_ratio', 'volatility', 'volume_change', 'ema_diff', 'price_ema_ratio', 'bb_width', 'vix_change']
    features = [re.sub(r'[^A-Za-z0-9_]', '_', str(f)) for f in features]
    scaler = StandardScaler()
    all_data[features] = scaler.fit_transform(all_data[features])

    agent = RLAgent(state_size=len(features))
    action_str, current_value = evaluate_today(agent, all_data, features)

    st.success(f"📌 今日建议：{action_str} 当前组合价值：${current_value:,.2f}")
