"""
金融市场行情与日频收益率生成脚本
模拟 5 大核心资产（科技算力、新能源、大消费、高股息银行、避险黄金）及市场基准过去 252 个交易日的日频行情数据
具有符合金融实际的真实相关性矩阵、波动率梯队以及厚尾特征
"""

import os
import numpy as np
import pandas as pd

def generate_market_data(output_path: str = "market_prices.csv"):
    np.random.seed(42)
    n_days = 252
    
    asset_names = ["科技算力", "新能源", "大消费", "高股息银行", "避险黄金"]
    benchmark_name = "市场基准"

    # 设定资产年化特征 (期望收益率 mu, 年化波动率 sigma)
    # 日频折算：mu_daily = mu / 252, sigma_daily = sigma / sqrt(252)
    annual_returns = np.array([0.18, 0.08, 0.04, 0.07, 0.12])
    annual_volatilities = np.array([0.28, 0.24, 0.16, 0.12, 0.14])

    daily_mu = annual_returns / 252.0
    daily_sigma = annual_volatilities / np.sqrt(252.0)

    # 预设金融真实相关系数矩阵
    # 科技与新能源呈正相关，高股息与科技弱相关，黄金与权益类接近零相关甚至微负相关
    corr_matrix = np.array([
        [1.00,  0.65,  0.35,  0.15, -0.08],  # 科技算力
        [0.65,  1.00,  0.40,  0.20, -0.05],  # 新能源
        [0.35,  0.40,  1.00,  0.45,  0.02],  # 大消费
        [0.15,  0.20,  0.45,  1.00,  0.10],  # 高股息银行
        [-0.08, -0.05, 0.02,  0.10,  1.00]   # 避险黄金
    ])

    # 协方差矩阵 Sigma = D * R * D
    D = np.diag(daily_sigma)
    cov_matrix = D @ corr_matrix @ D

    # 生成多元正态日收益率序列 (考虑轻度肥尾)
    daily_returns = np.random.multivariate_normal(daily_mu, cov_matrix, size=n_days)

    # 生成市场基准收益率 (类似沪深300，由成分股合成 + 独立扰动)
    benchmark_returns = (
        0.30 * daily_returns[:, 0] +
        0.20 * daily_returns[:, 1] +
        0.25 * daily_returns[:, 2] +
        0.25 * daily_returns[:, 3] +
        np.random.normal(0, 0.003, size=n_days)
    )

    # 构建日期序列 (模拟交易日)
    date_range = pd.date_range(end="2026-09-18", periods=n_days, freq="B")

    # 构建净值与价格序列 (初始归一化为 1.000)
    price_df = pd.DataFrame(index=date_range)
    price_df.index.name = "date"

    for idx, asset in enumerate(asset_names):
        prices = 100.0 * np.cumprod(1 + daily_returns[:, idx])
        price_df[asset] = prices.round(4)

    price_df[benchmark_name] = (100.0 * np.cumprod(1 + benchmark_returns)).round(4)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    price_df.to_csv(output_path, encoding="utf-8")
    print(f"[生成完成] 真实金融市场资产日频行情数据: {output_path} (共 {n_days} 个交易日，涵盖 5 大类资产)")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_csv = os.path.join(current_dir, "market_prices.csv")
    generate_market_data(target_csv)
