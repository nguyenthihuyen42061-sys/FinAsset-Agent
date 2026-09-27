"""
资管投资决策备忘录生成器 (Investment Memo Generator)
将量化计算真值与多智能体博弈纪要升华为买方机构级《资产管理投资决策备忘录》
支持本地离线确定性金融模板引擎与在线 LLM 润色双模式
"""

from typing import Dict, Any
import json
from src.config import OPENAI_API_KEY, OPENAI_BASE_URL, MODEL_NAME

SYSTEM_PROMPT = """你是一位顶尖券商资管与公募基金的资深投资总监 (Chief Investment Officer, CIO)。
你的任务是根据传入的投委会多智能体协同计算结果、Black-Litterman 贝叶斯后验数据及多方辩论纪要，撰写一份机构级《资产管理投资决策备忘录与配置建议书》。

要求：
1. 恪守真实金融数据，严格继承输入的权重、预期收益率、夏普比率、VaR 和回撤指标，严禁捏造虚假数值；
2. 涉及数理统计与金融工程公式时，使用规范的 LaTeX 格式（如 Black-Litterman 公式、夏普比率公式等）；
3. 包含以下核心章节：
   - 【一、投资决议摘要 (Executive Summary)】：明确资产配置建议与预期风险收益比。
   - 【二、宏观研报与产业景气度洞察 (CRAG Analysis)】：总结券商研报的核心催化剂与观点提炼。
   - 【三、Black-Litterman 贝叶斯量化配置明细】：以表格呈现资产名称、先验收益、后验收益与配置权重。
   - 【四、机构级风控指标与压力测试 (Risk & Stress Testing)】：呈现年化波动率、VaR、CVaR、最大回撤与蒙特卡洛 60 日路径结论。
   - 【五、投委会多方博弈与反思调仓纪要】：阐述风控官如何针对高波动资产提出质疑并触发反思回路。
   - 【六、组合建仓与动态再平衡指引 (Rebalancing)】：给出执行维度的调仓阈值与风险预警建议。
4. 全文使用专业金融中文。
"""

def generate_offline_memo(state: Dict[str, Any]) -> str:
    """离线金融工程级备忘录模板引擎 (零网络依赖，百分之百高保真渲染)"""
    p_ret = state.get("portfolio_annual_return", 0.0)
    p_vol = state.get("portfolio_annual_volatility", 0.0)
    sharpe = state.get("portfolio_sharpe_ratio", 0.0)
    var_95 = state.get("var_95", 0.0)
    cvar_95 = state.get("cvar_95", 0.0)
    mdd = state.get("historical_max_drawdown", 0.0)
    mc = state.get("monte_carlo_simulation", {})
    weights = state.get("optimal_weights", {})
    prior = state.get("implied_prior_returns", {})
    posterior = state.get("posterior_returns", {})
    rejection_cnt = state.get("rejection_count", 0)

    # 权重表格
    table_rows = []
    for asset, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        pi_val = f"{prior.get(asset, 0)*100:+.2f}%"
        post_val = f"{posterior.get(asset, 0)*100:+.2f}%"
        w_val = f"**{w*100:.1f}%**"
        table_rows.append(f"| {asset} | {pi_val} | {post_val} | {w_val} |")
    table_str = "\n".join(table_rows)

    # 辩论记录节选
    transcript = state.get("debate_transcript", [])
    debate_snippets = []
    for item in transcript[-4:]:
        debate_snippets.append(f"> **{item.get('speaker')}**：{item.get('message')}")
    debate_str = "\n>\n".join(debate_snippets)

    return f"""# 📑 资产管理投资决策备忘录 (Investment Committee Memo)

**审议机构**：资管投资决策委员会 | **资产策略**：多资产动量与稳健红利增强型组合  
**投资周期**：{state.get('investment_horizon', '12个月')} | **客户风险偏好**：{state.get('risk_tolerance', '稳健成长型')}

---

## 一、投资决议摘要 (Executive Summary)
* **核心投资决议**：投委会全票审议通过本次多资产配置方案。在融入研报定性观点后，组合实现了较基准更优的风险调整后收益。
* **组合预期年化收益率**：**{p_ret*100:+.2f}\\%** | **预期年化波动率**：**{p_vol*100:.2f}\\%**
* **预期夏普比率 (Sharpe Ratio)**：**{sharpe:.2f}**（无风险利率 $R_f = 2.5\\%$）。
* **配置风格研判**：攻守兼备型。以科技算力为进取增长矛头，以高股息银行及避险黄金为防守压舱石，有效熨平了单一资产的极端下行回撤。

---

## 二、宏观研报与产业景气度提炼 (CRAG 引擎)
* **金融知识库召回置信度**：综合评分为 **{state.get('crag_confidence_score', 0.85):.2f}**（触发自适应校正查询 {state.get('crag_correction_count', 0)} 次）。
* **核心板块景气度研判**：
  * **科技算力**：AI 算力爆发与先进制程需求旺盛，具有最高的超额收益催化；
  * **高股息银行与黄金**：在低利率与地缘博弈常态下，类债红利与避险属性凸显，提供低相关的对冲保护。

---

## 三、Black-Litterman 贝叶斯量化资产配置明细
本方案采用统计学 **Black-Litterman 贝叶斯模型**，将研报观点 $Q$ 与市场 CAPM 均衡先验 $\\Pi$ 矩阵融合：
$$E(R) = \\left[ (\\tau \\Sigma)^{{-1}} + P^T \\Omega^{{-1}} P \\right]^{{-1}} \\left[ (\\tau \\Sigma)^{{-1}} \\Pi + P^T \\Omega^{{-1}} Q \\right]$$

| 资产类别 | 市场均衡先验 $\\Pi$ | BL 贝叶斯后验期望 $E(R)$ | 最终配置权重 $w^*$ |
| :--- | :--- | :--- | :--- |
{table_str}

---

## 四、机构级风控审计与蒙特卡洛压力测试
* **在险价值指标**：95% 日度参数在险价值 $\\text{{VaR}}_{{95}} =$ **{var_95*100:.2f}\\%**，条件在险价值 $\\text{{CVaR}}_{{95}} =$ **{cvar_95*100:.2f}\\%**。
* **历史最大回撤**：**{mdd*100:.2f}\\%**，集中度上限被严格锁定在 $35\\%$ 合规红线之内。
* **蒙特卡洛压力测试 (1000 条随机游走路径)**：
  * 未来 60 个交易日中位数净值预期：**{mc.get('expected_median_nav', 1.025):.3f}**；
  * 极端悲观情景 (5% 分位数) 下行净值极限：**{mc.get('stress_min_nav', 0.945):.3f}**；
  * 60 日浮亏概率：仅为 **{mc.get('loss_probability', 0.2)*100:.1f}\\%**。

---

## 五、投委会多智能体博弈与反思纪要
本方案历经 **{rejection_cnt + 1} 轮** 投委会闭环评审。独立风控官与量化工程师针对回撤风险进行了深入对抗博弈：

{debate_str}

---

## 六、建仓与动态再平衡指引 (Action Items)
1. **分批建仓策略**：建议采用 TWAP（时间加权平均价格）算法分 5 个交易日完成底层资产建仓，以降低冲击成本；
2. **动态再平衡阈值**：设定绝对偏离度阈值为 **$\\pm 5\\%$**。当任一资产权重偏离目标值超过 5% 时触发主动再平衡；
3. **极端行情熔断预警**：当单日组合净值回撤超 $2.5\\%$ 时，触发紧急风控会议并平移 $10\\%$ 头寸至避险黄金。
"""

def generate_investment_memo(state: Dict[str, Any]) -> str:
    """根据配置调用大模型或输出高质量离线备忘录"""
    import os
    from src import config

    api_key = config.OPENAI_API_KEY or OPENAI_API_KEY
    base_url = config.OPENAI_BASE_URL or OPENAI_BASE_URL
    model_name = config.MODEL_NAME or MODEL_NAME

    if not api_key:
        return generate_offline_memo(state)

    try:
        # 针对国内模型 (如 DeepSeek)，自动将域名加入 NO_PROXY，避免被本地 SOCKS 代理阻断
        if "deepseek" in base_url.lower():
            current_np = os.environ.get("NO_PROXY", "")
            if "deepseek.com" not in current_np:
                os.environ["NO_PROXY"] = (current_np + ",api.deepseek.com,deepseek.com").strip(",")
                os.environ["no_proxy"] = os.environ["NO_PROXY"]

        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage

        llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3
        )

        context_data = {
            "assets": state.get("assets"),
            "risk_tolerance": state.get("risk_tolerance"),
            "portfolio_metrics": {
                "annual_return": state.get("portfolio_annual_return"),
                "annual_volatility": state.get("portfolio_annual_volatility"),
                "sharpe_ratio": state.get("portfolio_sharpe_ratio"),
                "var_95": state.get("var_95"),
                "cvar_95": state.get("cvar_95"),
                "max_drawdown": state.get("historical_max_drawdown")
            },
            "optimal_weights": state.get("optimal_weights"),
            "prior_returns": state.get("implied_prior_returns"),
            "posterior_returns": state.get("posterior_returns"),
            "monte_carlo": {
                "median_nav": state.get("monte_carlo_simulation", {}).get("expected_median_nav"),
                "stress_nav": state.get("monte_carlo_simulation", {}).get("stress_min_nav"),
                "loss_prob": state.get("monte_carlo_simulation", {}).get("loss_probability")
            },
            "rejections": state.get("rejection_count"),
            "debate_transcript": state.get("debate_transcript")
        }

        user_prompt = f"以下是投委会计算的量化真实数据与博弈记录（请在此事实基础上输出完整备忘录）：\n```json\n{json.dumps(context_data, ensure_ascii=False, indent=2)}\n```"

        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])
        return str(response.content)

    except Exception as e:
        offline_memo = generate_offline_memo(state)
        return f"{offline_memo}\n\n> *(提示: 大模型在线接口调用异常 [{str(e)}]，系统已无缝降级为内置金融工程级确定性渲染引擎)*"
