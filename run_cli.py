"""
FinAsset-Agent 命令行测试与演示套件 (CLI Runner)
支持全自动多智能体协同评审与有环反思回退流转验证
"""

import os
import sys

# 兼容 Windows 终端 UTF-8 编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 将项目根目录加入 sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.workflow import build_finasset_graph

def run_investment_committee_simulation():
    print("=" * 80)
    print(">> 启动 FinAsset-Agent 资管投委会多智能体协同仿真系统")
    print("   涵盖角色: 投委会主管(Supervisor) | 研报投研员 | 量化工程师 | 独立风控官 | 投资经理")
    print("=" * 80)

    app = build_finasset_graph()

    initial_state = {
        "assets": ["科技算力", "新能源", "大消费", "高股息银行", "避险黄金"],
        "risk_tolerance": "稳健成长型",
        "investment_horizon": "12个月",
        "benchmark_name": "市场基准",
        "step_logs": [],
        "debate_transcript": [],
        "rejection_count": 0,
        "risk_aversion_penalty": 0.0
    }

    final_state = app.invoke(initial_state)

    print("\n【智能体状态流转执行流水】:")
    for log in final_state.get("step_logs", []):
        print(f"  {log}")

    print("\n" + "=" * 80)
    print("【投委会多智能体对抗辩论纪要 (Debate Transcript)】:")
    print("=" * 80)
    for idx, item in enumerate(final_state.get("debate_transcript", []), 1):
        print(f"\n[{idx}] 【{item.get('speaker')}】:")
        print(f"    {item.get('message')}")

    print("\n" + "=" * 80)
    print("【最终资产配置权重方案 (Optimal Weights)】:")
    print("=" * 80)
    weights = final_state.get("optimal_weights", {})
    prior = final_state.get("implied_prior_returns", {})
    post = final_state.get("posterior_returns", {})
    for asset, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        print(f"  • {asset:<8} | 最终权重: {w*100:5.1f}% | CAPM先验收益: {prior.get(asset, 0)*100:+5.2f}% | BL后验收益: {post.get(asset, 0)*100:+5.2f}%")

    print(f"\n  >> 预期年化收益率: {final_state.get('portfolio_annual_return', 0)*100:.2f}%")
    print(f"  >> 预期年化波动率: {final_state.get('portfolio_annual_volatility', 0)*100:.2f}%")
    print(f"  >> 预期夏普比率:   {final_state.get('portfolio_sharpe_ratio', 0):.2f}")
    print(f"  >> 95% 日度 VaR:   {final_state.get('var_95', 0)*100:.2f}%")
    print(f"  >> 历史最大回撤:   {final_state.get('historical_max_drawdown', 0)*100:.2f}%")

    print("\n" + "=" * 80)
    print("【最终投资决策备忘录预览 (前 25 行)】:")
    print("=" * 80)
    memo = final_state.get("final_memo", "")
    lines = memo.split("\n")
    for line in lines[:25]:
        print(f"  {line}")
    if len(lines) > 25:
        print("  ... (更多详细内容已收拢至最终文件)")

if __name__ == "__main__":
    run_investment_committee_simulation()
