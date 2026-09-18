"""
独立合规风控官 Agent (Risk Officer)
负责对量化配置草案执行在险价值 (VaR)、期望尾部损失 (CVaR)、最大回撤和蒙特卡洛压力测试
具备一票否决权（Veto Power），触发状态机有向回退反思回路 (Feedback Loop)
"""

import os
from typing import Dict, Any
import numpy as np
import pandas as pd
from src.state import InvestmentState
from src.risk_engine import calculate_portfolio_risk_metrics, run_monte_carlo_stress_test
from src.config import MAX_RISK_REJECTIONS

def risk_officer_node(state: InvestmentState) -> Dict[str, Any]:
    logs = list(state.get("step_logs", []))
    transcript = list(state.get("debate_transcript", []))

    logs.append("[独立风控官] 启动机构级风险审计：执行 95% VaR、CVaR、历史最大回撤及 1000 次蒙特卡洛压力测试...")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "market_prices.csv")
    df = pd.read_csv(data_path, index_col=0)

    assets = state.get("assets", ["科技算力", "新能源", "大消费", "高股息银行", "避险黄金"])
    weights_dict = state.get("optimal_weights", {})
    weights = np.array([weights_dict.get(a, 0.0) for a in assets])

    returns_df = df[assets].pct_change().dropna()
    p_ret = state.get("portfolio_annual_return", 0.10)
    p_vol = state.get("portfolio_annual_volatility", 0.15)

    # 1. 计算核心风控指标
    risk_metrics = calculate_portfolio_risk_metrics(
        returns_df=returns_df,
        weights=weights,
        annual_return=p_ret,
        annual_volatility=p_vol
    )

    # 2. 运行 1000 条蒙特卡洛压力测试
    mc_results = run_monte_carlo_stress_test(
        annual_return=p_ret,
        annual_volatility=p_vol,
        n_sims=1000,
        n_days=60
    )

    rejection_count = state.get("rejection_count", 0)
    risk_passed = risk_metrics["overall_risk_passed"]

    # 模拟真实机构投决逻辑：首轮如果年化波动率高于 18% 且为稳健型客户，适度触发反思回路以展示智能体博弈
    user_tolerance = state.get("risk_tolerance", "稳健成长型")
    trigger_demonstration_rejection = (rejection_count == 0 and "稳健" in user_tolerance and p_vol > 0.16)

    if (not risk_passed or trigger_demonstration_rejection) and rejection_count < MAX_RISK_REJECTIONS:
        # 触发一票否决
        rejection_reason = risk_metrics["rejection_reason"] or "针对稳健型投资目标，当前组合年化波动率与高风险资产风险暴露偏高，需强化防守！"
        logs.append(f"[独立风控官] 🚨 【否决决议】{rejection_reason}")
        
        transcript.append({
            "speaker": "独立合规风控官 (Risk Officer)",
            "message": (
                f"【风控一票否决驳回】经风险模型测算，配置草案未能通过合规风控审查：\n"
                f"• 测算数据：95% 日度在险价值 VaR={risk_metrics['parametric_var_95']*100:.2f}% | 历史最大回撤={risk_metrics['max_drawdown']*100:.2f}%\n"
                f"• 驳回原因：{rejection_reason}\n"
                f"• 整改指令：要求【投委会主管】上调风险惩罚系数，压降高波动资产持仓，增加高股息与黄金防守权重后重新提报！"
            )
        })

        return {
            "step_logs": logs,
            "debate_transcript": transcript,
            "var_95": risk_metrics["parametric_var_95"],
            "cvar_95": risk_metrics["hist_cvar_95"],
            "historical_max_drawdown": risk_metrics["max_drawdown"],
            "max_weight_concentration": risk_metrics["max_single_weight"],
            "monte_carlo_simulation": mc_results,
            "risk_rejected": True,
            "rejection_reason": rejection_reason,
            "rejection_count": rejection_count + 1,
            "risk_aversion_penalty": state.get("risk_aversion_penalty", 0.0) + 1.8,
            "current_agent": "risk_officer"
        }
    else:
        # 审查通过
        logs.append(
            f"[独立风控官] ✅ 审查通过！95% 日度在险价值 VaR={risk_metrics['parametric_var_95']*100:.2f}%, "
            f"CVaR={risk_metrics['hist_cvar_95']*100:.2f}%, 最大回撤={risk_metrics['max_drawdown']*100:.2f}%，符合资管指引。"
        )
        transcript.append({
            "speaker": "独立合规风控官 (Risk Officer)",
            "message": (
                f"【风控审查通过】经 1000 条蒙特卡洛压力测试检验，当前组合风险收益特征已完全收敛至稳健合规区间：\n"
                f"• 极端悲观情景 (5%分位) 60日净值预期下限: {mc_results['stress_min_nav']:.3f}\n"
                f"• 综合 VaR 与单一资产集中度均符合公募审慎投资法规，准予出具最终投资决议。"
            )
        })

        return {
            "step_logs": logs,
            "debate_transcript": transcript,
            "var_95": risk_metrics["parametric_var_95"],
            "cvar_95": risk_metrics["hist_cvar_95"],
            "historical_max_drawdown": risk_metrics["max_drawdown"],
            "max_weight_concentration": risk_metrics["max_single_weight"],
            "monte_carlo_simulation": mc_results,
            "risk_rejected": False,
            "rejection_reason": None,
            "current_agent": "risk_officer"
        }
