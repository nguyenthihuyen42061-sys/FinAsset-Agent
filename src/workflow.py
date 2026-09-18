"""
LangGraph 有环多智能体状态机编译模块 (Cyclic Multi-Agent Workflow)
搭建包含 Supervisor、Alpha Analyst、Quant Analyst、Risk Officer、Portfolio Manager 的有向有环图
支持风控驳回重审的动态反思闭环机制 (Cyclic Feedback Loop)
"""

from langgraph.graph import StateGraph, END
from src.state import InvestmentState
from src.agents.supervisor import supervisor_node, route_next_step
from src.agents.alpha_analyst import alpha_analyst_node
from src.agents.quant_analyst import quant_analyst_node
from src.agents.risk_officer import risk_officer_node
from src.agents.portfolio_manager import portfolio_manager_node
from src.config import MAX_RISK_REJECTIONS

def route_after_risk(state: InvestmentState) -> str:
    """
    风控节点后的条件路由：
    - 若被驳回且未超最大重试次数，则回退流转给 Supervisor 触发反思闭环
    - 若风控通过，则流转给 Portfolio Manager
    """
    if state.get("risk_rejected", False):
        if state.get("rejection_count", 0) <= MAX_RISK_REJECTIONS:
            return "supervisor"
        else:
            return "portfolio_manager"
    return "portfolio_manager"

def build_finasset_graph():
    """
    构建并编译资管投委会多智能体有环拓扑图
    """
    workflow = StateGraph(InvestmentState)

    # 1. 注册 5 大智能体节点
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("alpha_analyst", alpha_analyst_node)
    workflow.add_node("quant_analyst", quant_analyst_node)
    workflow.add_node("risk_officer", risk_officer_node)
    workflow.add_node("portfolio_manager", portfolio_manager_node)

    # 2. 设置起始入口为投委会主管
    workflow.set_entry_point("supervisor")

    # 3. Supervisor 条件分发边
    workflow.add_conditional_edges(
        "supervisor",
        route_next_step,
        {
            "alpha_analyst": "alpha_analyst",
            "quant_analyst": "quant_analyst",
            "risk_officer": "risk_officer",
            "portfolio_manager": "portfolio_manager"
        }
    )

    # 4. 投研员汇报后，进入量化配置
    workflow.add_edge("alpha_analyst", "quant_analyst")

    # 5. 量化配置求解后，提交独立风控官审查
    workflow.add_edge("quant_analyst", "risk_officer")

    # 6. 风控官审查后的条件边（核心有环反馈回路）
    workflow.add_conditional_edges(
        "risk_officer",
        route_after_risk,
        {
            "supervisor": "supervisor",             # 触发有环反思回退！
            "portfolio_manager": "portfolio_manager" # 审查通过，正常流转
        }
    )

    # 7. 投资经理签署后，流程终止
    workflow.add_edge("portfolio_manager", END)

    return workflow.compile()
