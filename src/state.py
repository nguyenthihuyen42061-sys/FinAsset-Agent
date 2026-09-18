"""
多智能体投研状态契约 (Investment State Contract)
在资管投委会多智能体协同网络（Supervisor -> Alpha -> Quant -> Risk -> PM）中传递的强类型状态
"""

from typing import TypedDict, Optional, Dict, Any, List

class InvestmentState(TypedDict, total=False):
    """
    FinAsset-Agent 全局状态契约
    """
    # 1. 投资意图与资产池
    assets: List[str]                            # 资产池列表
    risk_tolerance: str                          # 风险承受度 (如 '稳健成长型', '积极进取型', '稳健保守型')
    investment_horizon: str                      # 投资周期 (如 '12个月', '中长期')
    benchmark_name: str                          # 基准指数名称

    # 2. 状态机路由与执行轨迹
    current_agent: str                           # 当前轮转执行的智能体角色
    step_logs: List[str]                         # 步骤执行流水日志
    debate_transcript: List[Dict[str, str]]      # 投委会多方辩论对抗记录清单

    # 3. 垂直 CRAG 研报检索结果
    retrieved_reports: List[Dict[str, Any]]      # 召回的研报核心切片
    crag_confidence_score: float                 # 研报检索质量评估置信度打分
    crag_correction_count: int                   # 校正检索触发计数
    analyst_views: Dict[str, float]              # 研报分析师提取的超额预期收益观点 Q
    analyst_confidence: Dict[str, float]         # 研报观点置信度 (决定 BL 矩阵 Omega)

    # 4. Black-Litterman 与马科维茨量化配置结果
    implied_prior_returns: Dict[str, float]      # 市场先验均衡收益率 Pi
    posterior_returns: Dict[str, float]          # Black-Litterman 贝叶斯后验预期收益率 E(R)
    optimal_weights: Dict[str, float]            # 最优资产配置权重 (和为 100%)
    portfolio_annual_return: float               # 组合预期年化收益率
    portfolio_annual_volatility: float           # 组合预期年化波动率
    portfolio_sharpe_ratio: float                # 组合预期夏普比率
    efficient_frontier_points: List[Dict[str, float]] # 马科维茨有效前沿坐标集 (用于 Plotly 画图)

    # 5. 风控审计、压力测试与反思回退控制
    var_95: float                                # 95% 置信度日度在险价值 (VaR)
    cvar_95: float                               # 95% 条件在险价值 (CVaR)
    historical_max_drawdown: float               # 组合历史最大回撤
    max_weight_concentration: float              # 单一最大资产权重集中度
    monte_carlo_simulation: Dict[str, Any]       # 1000 次蒙特卡洛未来路径与分位数坐标
    
    # 核心有环回退标识 (Feedback Loop)
    risk_rejected: bool                          # 是否被独立风控官一票否决驳回
    rejection_reason: Optional[str]              # 驳回整改意见
    rejection_count: int                         # 已驳回回退循环次数 (防止无限死循环)
    risk_aversion_penalty: float                 # 动态风险惩罚乘数 (驳回时调大，强制压降集中度与风险)

    # 6. 最终交付物
    final_memo: Optional[str]                    # 资管投委会最终签署的投资决策备忘录 (Markdown/LaTeX)
