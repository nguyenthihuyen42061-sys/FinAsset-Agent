"""
统计学量化配置核心引擎 (Quantitative & Black-Litterman Engine)
实现国际金融工程皇冠级资产配置模型：
1. 市场隐含均衡收益率反向推导 (CAPM Reverse Optimization)
2. Black-Litterman 贝叶斯研报观点融合推导
3. 马科维茨现代投资组合理论 (MPT) 凸优化求解器
4. 有效前沿 (Efficient Frontier) 曲线生成
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import RISK_FREE_RATE, DEFAULT_DELTA, TAU, MAX_SINGLE_ASSET_WEIGHT

def load_returns_and_covariance(csv_path: str, assets: List[str]) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """读取市场历史行情，计算日收益率、年化期望收益率与年化协方差矩阵"""
    df = pd.read_csv(csv_path, index_col=0)
    # 筛选资产列
    valid_assets = [a for a in assets if a in df.columns]
    if len(valid_assets) < 2:
        raise ValueError(f"资产池中有效资产少于 2 种: {valid_assets}")

    price_df = df[valid_assets]
    returns_df = price_df.pct_change().dropna()

    # 年化协方差矩阵 (252 个交易日)
    cov_annual = returns_df.cov().to_numpy() * 252.0
    mean_daily = returns_df.mean().to_numpy()
    mean_annual = mean_daily * 252.0

    return returns_df, mean_annual, cov_annual

def calculate_black_litterman_posterior(
    assets: List[str],
    cov_annual: np.ndarray,
    views_dict: Dict[str, float],
    confidence_dict: Dict[str, float],
    delta: float = DEFAULT_DELTA,
    tau: float = TAU
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    执行 Black-Litterman 贝叶斯后验期望收益率计算
    公式:
      Pi = delta * Sigma * w_mkt
      E(R) = [ (tau * Sigma)^(-1) + P^T * Omega^(-1) * P ]^(-1) * [ (tau * Sigma)^(-1) * Pi + P^T * Omega^(-1) * Q ]
    """
    n = len(assets)
    # 设定等权市场基准先验 w_mkt
    w_mkt = np.ones(n) / n

    # 1. 市场先验均衡超额收益率 Pi (Reverse Optimization)
    pi = delta * (cov_annual @ w_mkt)

    # 2. 构造研报观点矩阵 P 与观点向量 Q
    # 针对绝对观点：P 为单位对角阵 I, Q 为研报分析师提取的预期年化超额收益
    P = np.eye(n)
    Q = np.array([views_dict.get(a, pi[idx]) for idx, a in enumerate(assets)])

    # 3. 构造观点不确定性对角协方差阵 Omega (He & Litterman 标准公式)
    # Omega_ii = P_i * (tau * Sigma) * P_i^T * (1 - c_i) / c_i
    tau_sigma = tau * cov_annual
    omega = np.zeros((n, n))
    for i in range(n):
        c_i = max(0.01, min(0.99, confidence_dict.get(assets[i], 0.80)))
        variance_term = P[i] @ tau_sigma @ P[i].T
        omega[i, i] = variance_term * ((1.0 - c_i) / c_i)

    # 4. 贝叶斯后验更新
    inv_tau_sigma = np.linalg.inv(tau_sigma)
    inv_omega = np.linalg.inv(omega)

    # 后验精度矩阵 M
    M = np.linalg.inv(inv_tau_sigma + P.T @ inv_omega @ P)
    # 后验期望收益率 E(R)
    er_posterior = M @ (inv_tau_sigma @ pi + P.T @ inv_omega @ Q)

    # 后验协方差矩阵 (考虑参数估计不确定性膨胀)
    cov_posterior = cov_annual + M

    return pi, er_posterior, cov_posterior

def optimize_portfolio(
    er: np.ndarray,
    cov: np.ndarray,
    max_single_weight: float = MAX_SINGLE_ASSET_WEIGHT,
    risk_aversion: float = DEFAULT_DELTA
) -> Tuple[np.ndarray, float, float, float]:
    """
    马科维茨均值-方差凸优化求解器 (二次规划 / SLSQP)
    目标: max  w^T * er - 0.5 * delta * w^T * cov * w
    约束: sum(w) = 1,  0 <= w_i <= max_single_weight
    """
    n = len(er)

    def objective(w):
        port_return = np.dot(w, er)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        # 最大化夏普等价于最小化负夏普 (无风险利率 rf)
        sharpe = (port_return - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0
        # 融入风险厌恶惩罚
        return - (port_return - 0.5 * risk_aversion * (port_vol ** 2))

    # 约束条件: 权重之和为 1
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    # 边界约束: 不允许做空 (0 <= w_i <= max_single_weight)
    bounds = tuple((0.0, max_single_weight) for _ in range(n))

    # 初始均权投点
    init_guess = np.ones(n) / n

    res = minimize(
        objective, 
        init_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-9}
    )

    optimal_w = res.x
    # 归一化消除浮点微小误差
    optimal_w = optimal_w / np.sum(optimal_w)

    port_ret = float(np.dot(optimal_w, er))
    port_vol = float(np.sqrt(np.dot(optimal_w.T, np.dot(cov, optimal_w))))
    sharpe = float((port_ret - RISK_FREE_RATE) / port_vol) if port_vol > 0 else 0.0

    return optimal_w, port_ret, port_vol, sharpe

def generate_efficient_frontier(
    er: np.ndarray, 
    cov: np.ndarray, 
    n_points: int = 25,
    max_weight: float = MAX_SINGLE_ASSET_WEIGHT
) -> List[Dict[str, float]]:
    """生成马科维茨有效前沿坐标集 (供 Plotly 交互式绘制)"""
    n = len(er)
    min_ret = np.min(er)
    max_ret = np.max(er)
    target_returns = np.linspace(min_ret * 0.95, max_ret * 1.02, n_points)

    frontier_points = []
    init_guess = np.ones(n) / n
    bounds = tuple((0.0, max_weight) for _ in range(n))

    for r_target in target_returns:
        # 给定目标收益率，最小化组合方差
        def min_variance(w):
            return np.dot(w.T, np.dot(cov, w))

        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': lambda w: np.dot(w, er) - r_target}
        )

        res = minimize(min_variance, init_guess, method='SLSQP', bounds=bounds, constraints=cons)
        if res.success:
            vol = float(np.sqrt(res.fun))
            sharpe = (r_target - RISK_FREE_RATE) / vol if vol > 0 else 0
            frontier_points.append({
                "return": round(float(r_target * 100), 2),
                "volatility": round(float(vol * 100), 2),
                "sharpe": round(float(sharpe), 3)
            })

    return frontier_points
