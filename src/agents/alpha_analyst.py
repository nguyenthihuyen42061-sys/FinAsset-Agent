"""
研报投研分析 Agent (Alpha Analyst)
负责利用金融垂直校正性 RAG (CRAG) 检索本地券商研报，挖掘行业景气度并提炼 Black-Litterman 观点矩阵 Q 与置信度 Omega
"""

import os
from typing import Dict, Any
from src.state import InvestmentState
from src.rag_engine import FinancialCRAGEngine

def alpha_analyst_node(state: InvestmentState) -> Dict[str, Any]:
    logs = list(state.get("step_logs", []))
    transcript = list(state.get("debate_transcript", []))

    logs.append("[研报投研员] 启动金融垂直 CRAG 引擎，正在检索研报切片并评估信息置信度...")

    # 获取研报目录
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    reports_dir = os.path.join(base_dir, "data", "reports")

    rag = FinancialCRAGEngine(reports_dir)
    assets = state.get("assets", ["科技算力", "新能源", "大消费", "高股息银行", "避险黄金"])

    retrieved, views, confidences, avg_conf, corrections = rag.retrieve(assets)

    logs.append(
        f"[研报投研员] 检索完成！召回 {len(retrieved)} 条核心研报切片，"
        f"综合检索置信度: {avg_conf}，触发校正性扩展次数: {corrections}"
    )

    # 构造投委会辩论发言
    view_details = ", ".join([f"{k}: 预期超额收益 {v*100:+.1f}% (置信度 {confidences.get(k, 0.8)*100:.0f}%)" for k, v in views.items()])
    transcript.append({
        "speaker": "研报投研员 (Alpha Analyst)",
        "message": (
            f"根据最新券商研报 CRAG 检索与归纳，宏观产业景气度呈现分化结构：\n"
            f"• 观点提炼：{view_details}\n"
            f"• 核心洞察：科技算力具备强催化，建议作为进攻矛头；高股息银行与黄金具备极高避险对冲价值，建议纳入底仓。"
        )
    })

    return {
        "step_logs": logs,
        "debate_transcript": transcript,
        "retrieved_reports": retrieved,
        "analyst_views": views,
        "analyst_confidence": confidences,
        "crag_confidence_score": avg_conf,
        "crag_correction_count": corrections,
        "current_agent": "alpha_analyst"
    }
