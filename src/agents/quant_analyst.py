"""
量化配置工程师 Agent (Quant Analyst)
负责利用统计学 Black-Litterman 贝叶斯模型融合研报观点与市场先验，并利用凸优化求解马科维茨有效前沿最优权重
"""

import os
from typing import Dict, Any
from src.state import InvestmentState
from src.config import DEFAULT_DELTA, MAX_SINGLE_ASSET_WEIGHT
from src.quant_engine import (
    load_returns_and_covariance,
    calculate_black_litterman_posterior,
    optimize_portfolio,
    generate_efficient_frontier
)

def quant_analyst_node(state: InvestmentState) -> Dict[str, Any]:
    logs = list(state.get("step_logs", []))
    transcript = list(state.get("debate_transcript", []))

    # 检查是否有风控驳回施加的动态风险惩罚 (Feedback Loop)
    base_delta = DEFAULT_DELTA
    penalty = state.get("risk_aversion_penalty", 0.0)
    effective_delta = base_delta + penalty
    rejection_count = state.get("rejection_count", 0)

    if rejection_count > 0:
        logs.append(f"[量化工程师] 响应风控指令，启动二次重拟合！将风险厌恶系数 delta 从 {base_delta} 上调至 {effective_delta:.1f}，收紧优化边界...")
    else:
        logs.append("[量化工程师] 启动 Black-Litterman 贝叶斯量化推导与有效前沿凸优化...")

    # 读取行情数据
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "market_prices.csv")
    assets = state.get("assets", ["科技算力", "新能源", "大消费", "高股息银行", "避险黄金"])

    returns_df, mean_annual, cov_annual = load_returns_and_covariance(data_path, assets)

    views = state.get("analyst_views", {})
    confidences = state.get("analyst_confidence", {})

    # 1. 贝叶斯后验推导
    pi, er_post, cov_post = calculate_black_litterman_posterior(
        assets=assets,
        cov_annual=cov_annual,
        views_dict=views,
        confidence_dict=confidences,
        delta=effective_delta
    )

    # 2. 凸优化求解最优权重
    # 若被风控驳回过，单资产上限适当收紧 (例如从 35% 压低至 30%)
    max_cap = MAX_SINGLE_ASSET_WEIGHT if rejection_count == 0 else max(0.25, MAX_SINGLE_ASSET_WEIGHT - 0.05 * rejection_count)
    opt_w, p_ret, p_vol, p_sharpe = optimize_portfolio(
        er=er_post,
        cov=cov_post,
        max_single_weight=max_cap,
        risk_aversion=effective_delta
    )

    # 3. 生成有效前沿点集
    frontier_points = generate_efficient_frontier(
        er=er_post,
        cov=cov_post,
        n_points=25,
        max_weight=max_cap
    )

    # 格式化字典
    prior_dict = {a: round(float(pi[i]), 4) for i, a in enumerate(assets)}
    post_dict = {a: round(float(er_post[i]), 4) for i, a in enumerate(assets)}
    weights_dict = {a: round(float(opt_w[i]), 4) for i, a in enumerate(assets)}

    logs.append(
        f"[量化工程师] 优化完成！预期年化收益率: {p_ret*100:.2f}%, "
        f"年化波动率: {p_vol*100:.2f}%, 夏普比率: {p_sharpe:.3f}"
    )

    weight_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in weights_dict.items() if v > 0.01])
    transcript.append({
        "speaker": "量化配置工程师 (Quant Analyst)",
        "message": (
            f"已完成 Black-Litterman 后验收益分布计算与马科维茨二次规划求解：\n"
            f"• 配置权重草案：{weight_str}\n"
            f"• 绩效指标：预期年化收益率 {p_ret*100:.2f}% | 波动率 {p_vol*100:.2f}% | 夏普比率 {p_sharpe:.2f}\n"
            f"• 提交独立风控官进行机构级在险价值与集中度合规审查。"
        )
    })

    return {
        "step_logs": logs,
        "debate_transcript": transcript,
        "implied_prior_returns": prior_dict,
        "posterior_returns": post_dict,
        "optimal_weights": weights_dict,
        "portfolio_annual_return": round(p_ret, 4),
        "portfolio_annual_volatility": round(p_vol, 4),
        "portfolio_sharpe_ratio": round(p_sharpe, 4),
        "efficient_frontier_points": frontier_points,
        "current_agent": "quant_analyst"
    }
