# RL 股票策略分析器

本项目使用强化学习策略每日自动评估买卖建议，支持部署到 Streamlit Cloud。

## 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 部署

1. 打开 https://streamlit.io/cloud
2. 创建新应用，连接 GitHub 仓库（或上传 zip 解压后的内容）
3. 设置启动命令为：

```bash
streamlit run app.py
```

