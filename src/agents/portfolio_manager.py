"""
资管投资经理 Agent (Portfolio Manager)
负责汇总统委会多方博弈论点、Black-Litterman 配置权重与风控审计数据，签署并出具正式《资产管理投资决策备忘录》
"""

from typing import Dict, Any
from src.state import InvestmentState
from src.report_template import generate_investment_memo

def portfolio_manager_node(state: InvestmentState) -> Dict[str, Any]:
    logs = list(state.get("step_logs", []))
    transcript = list(state.get("debate_transcript", []))

    logs.append("[资管投资经理] 正在综合各方研报、量化求解与合规风控数据，起草正式《资产管理投资决策备忘录》...")

    memo = generate_investment_memo(state)

    transcript.append({
        "speaker": "资管投资经理 (Portfolio Manager)",
        "message": (
            "全套投资方案已正式签署归档！已将组合权重、Black-Litterman 贝叶斯后验参数、"
            "有效前沿坐标与蒙特卡洛压力测试结论封装至正式投资建议书中，交付执行团队建仓。"
        )
    })
    logs.append("[资管投资经理] 投资备忘录生成完毕，全流程审议闭环完成！")

    return {
        "step_logs": logs,
        "debate_transcript": transcript,
        "final_memo": memo,
        "current_agent": "portfolio_manager"
    }
