"""
投委会总指挥调度 Agent (Supervisor Agent)
负责资管投决会的全流程状态调度、多方辩论协调以及有环反思回退控制 (Cyclic Controller)
"""

from typing import Dict, Any
from src.state import InvestmentState
from src.config import MAX_RISK_REJECTIONS

def supervisor_node(state: InvestmentState) -> Dict[str, Any]:
    """
    Supervisor 节点：更新全局进度并记录投委会调度轨迹
    """
    logs = list(state.get("step_logs", []))
    transcript = list(state.get("debate_transcript", []))
    rejection_count = state.get("rejection_count", 0)
    risk_rejected = state.get("risk_rejected", False)

    if risk_rejected:
        logs.append(f"[投委会调度] 🚨 收到独立风控官的驳回令（当前已驳回 {rejection_count} 次）！触发有环反思回退机制。")
        transcript.append({
            "speaker": "投委会主管 (Supervisor)",
            "message": f"风控意见成立！现将资产配置方案打回【量化配置工程师】。要求启动自适应参数反思，上调风险惩罚系数，强制平抑波动率。"
        })
    else:
        if not state.get("analyst_views"):
            logs.append("[投委会调度] 启动新一轮资管资产配置评审会，下发研报分析任务给【研报投研员】。")
            transcript.append({
                "speaker": "投委会主管 (Supervisor)",
                "message": "各位委员好，本次投委会审议既定资产池的年度配置方案。请【研报投研员】先基于金融知识库汇报行业景气度与预期收益观点。"
            })
        else:
            logs.append("[投委会调度] 审核已通过风控审查，正式交由【资管投资经理】签署最终投资备忘录。")
            transcript.append({
                "speaker": "投委会主管 (Supervisor)",
                "message": "风控门禁与量化测算均已达标，请【资管投资经理】整理多方意见，形成正式《资产管理投资备忘录》。"
            })

    return {
        "step_logs": logs,
        "debate_transcript": transcript,
        "current_agent": "supervisor"
    }

def route_next_step(state: InvestmentState) -> str:
    """
    Supervisor 路由决策逻辑（支持有向循环带环回退）
    """
    # 1. 若尚未获取研报观点，前往研报投研员
    if not state.get("analyst_views"):
        return "alpha_analyst"

    # 2. 若刚刚被风控驳回，且未超过最大驳回次数，带环回退至量化工程师重新优化
    if state.get("risk_rejected", False):
        if state.get("rejection_count", 0) <= MAX_RISK_REJECTIONS:
            return "quant_analyst"
        else:
            # 达到最大驳回次数，强制收敛流转至投资经理进行风险折中签署
            return "portfolio_manager"

    # 3. 若尚未进行量化优化，前往量化工程师
    if not state.get("optimal_weights"):
        return "quant_analyst"

    # 4. 若尚未进行风控审计，前往独立风控官
    if not state.get("var_95"):
        return "risk_officer"

    # 5. 一切正常且风控通过，前往投资经理出具最终决议
    return "portfolio_manager"
