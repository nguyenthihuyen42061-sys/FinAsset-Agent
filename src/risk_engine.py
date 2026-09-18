"""
机构级风控与压力测试引擎 (Institutional Risk & Stress Testing Engine)
实现资管与财富管理核心风控指标测算与极端市场情景压力测试：
1. 参数法与历史模拟法在险价值 (VaR at 95%)
2. 期望尾部损失 (CVaR / Expected Shortfall)
3. 组合历史最大回撤 (Max Drawdown)
4. 单一资产集中度合规审计 (Concentration Audit)
5. 1000 条几何布朗运动 (GBM) 蒙特卡洛压力测试
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from src.config import MAX_ALLOWED_VAR_95, MAX_SINGLE_ASSET_WEIGHT, MONTE_CARLO_SIMS, MONTE_CARLO_DAYS

def calculate_portfolio_risk_metrics(
    returns_df: pd.DataFrame,
    weights: np.ndarray,
    annual_return: float,
    annual_volatility: float
) -> Dict[str, Any]:
    """
    全面测算资产组合风控指标
    """
    # 历史日收益率序列合成 R_p(t) = sum(w_i * R_i(t))
    daily_returns_port = (returns_df.to_numpy() @ weights)

    # 1. 参数法日度 VaR (95% 置信度，单尾分位数 1.645)
    daily_mu = annual_return / 252.0
    daily_sigma = annual_volatility / np.sqrt(252.0)
    parametric_var_95 = max(0.0, 1.6449 * daily_sigma - daily_mu)

    # 2. 历史模拟法日度 VaR (5% 经验下分位数)
    hist_var_95 = max(0.0, -float(np.percentile(daily_returns_port, 5.0)))

    # 3. 条件在险价值 CVaR (期望缺口：低于 VaR 损失的平均值)
    tail_losses = daily_returns_port[daily_returns_port <= -hist_var_95]
    hist_cvar_95 = -float(np.mean(tail_losses)) if len(tail_losses) > 0 else hist_var_95 * 1.25

    # 4. 历史最大回撤 (Max Drawdown)
    cum_wealth = np.cumprod(1.0 + daily_returns_port)
    cum_peaks = np.maximum.accumulate(cum_wealth)
    drawdowns = (cum_peaks - cum_wealth) / cum_peaks
    max_drawdown = float(np.max(drawdowns))

    # 5. 单一资产权重集中度
    max_weight = float(np.max(weights))

    # 合规门禁判定
    is_var_passed = bool(parametric_var_95 <= MAX_ALLOWED_VAR_95)
    is_concentration_passed = bool(max_weight <= (MAX_SINGLE_ASSET_WEIGHT + 0.005)) # 容忍微小浮点误差
    overall_passed = is_var_passed and is_concentration_passed

    rejection_reasons = []
    if not is_var_passed:
        rejection_reasons.append(
            f"95% 日度在险价值 (VaR={parametric_var_95*100:.2f}%) 超出资管机构风控上限 ({MAX_ALLOWED_VAR_95*100:.2f}%)，极端下行波动过大！"
        )
    if not is_concentration_passed:
        rejection_reasons.append(
            f"单一资产最高持仓权重 ({max_weight*100:.1f}%) 突破公募资管集中度红线 ({MAX_SINGLE_ASSET_WEIGHT*100:.1f}%)，存在过度押注风险！"
        )

    return {
        "parametric_var_95": round(parametric_var_95, 4),
        "hist_var_95": round(hist_var_95, 4),
        "hist_cvar_95": round(hist_cvar_95, 4),
        "max_drawdown": round(max_drawdown, 4),
        "max_single_weight": round(max_weight, 4),
        "is_var_passed": is_var_passed,
        "is_concentration_passed": is_concentration_passed,
        "overall_risk_passed": overall_passed,
        "rejection_reason": " 且 ".join(rejection_reasons) if rejection_reasons else None
    }

def run_monte_carlo_stress_test(
    annual_return: float,
    annual_volatility: float,
    n_sims: int = MONTE_CARLO_SIMS,
    n_days: int = MONTE_CARLO_DAYS,
    init_nav: float = 1.000
) -> Dict[str, Any]:
    """
    运行 1000 条几何布朗运动 (GBM) 随机游走路径，开展前瞻性压力测试
    """
    np.random.seed(42)
    dt = 1.0 / 252.0
    mu = annual_return
    sigma = annual_volatility

    # 漂移项与随机项 (GBM: S_t = S_0 * exp((mu - 0.5*sigma^2)*t + sigma*sqrt(t)*Z))
    drift = (mu - 0.5 * sigma ** 2) * dt
    shock = sigma * np.sqrt(dt)

    daily_shocks = np.random.normal(0, 1, size=(n_days, n_sims))
    daily_returns = np.exp(drift + shock * daily_shocks)

    # 累积净值路径 (第 0 天为初始净值 1.0)
    paths = np.empty((n_days + 1, n_sims))
    paths[0] = init_nav
    paths[1:] = init_nav * np.cumprod(daily_returns, axis=0)

    # 计算时间序列分位数扇形带 (Fan Chart Percentiles)
    days_axis = list(range(n_days + 1))
    p5 = np.percentile(paths, 5, axis=1)    # 极端悲观情景 (5% 分位)
    p25 = np.percentile(paths, 25, axis=1)  # 弱市情景 (25% 分位)
    p50 = np.percentile(paths, 50, axis=1)  # 中性基准情景 (中位数 50% 分位)
    p75 = np.percentile(paths, 75, axis=1)  # 景气情景 (75% 分位)
    p95 = np.percentile(paths, 95, axis=1)  # 强乐观情景 (95% 分位)

    # 终止净值损失率与极值
    end_navs = paths[-1]
    loss_prob = float(np.mean(end_navs < init_nav)) # 60 天后浮亏概率

    return {
        "days": days_axis,
        "p5": [round(float(v), 4) for v in p5],
        "p25": [round(float(v), 4) for v in p25],
        "p50": [round(float(v), 4) for v in p50],
        "p75": [round(float(v), 4) for v in p75],
        "p95": [round(float(v), 4) for v in p95],
        "sample_paths": [[round(float(paths[d, s]), 4) for d in range(n_days + 1)] for s in range(5)], # 取前 5 条示例轨迹
        "stress_min_nav": round(float(np.min(p5)), 4),
        "expected_median_nav": round(float(p50[-1]), 4),
        "loss_probability": round(loss_prob, 3)
    }
