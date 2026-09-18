"""
FinAsset-Agent: 资管投委会多智能体协同投研与资产配置系统 (Web 终端)
基于 Streamlit + Plotly 构建，专为恒生电子金融场景智能体、资产管理与投研系统设计
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 确保加载 src 模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.workflow import build_finasset_graph
from src import config

st.set_page_config(
    page_title="FinAsset-Agent - 资管投委会多智能体系统",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义金融终端样式
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1E3A8A;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #64748B;
    }
    .agent-card-supervisor { border-left: 4px solid #EC4899; padding: 10px 14px; background: #FDF2F8; border-radius: 4px; margin-bottom: 10px; }
    .agent-card-alpha { border-left: 4px solid #3B82F6; padding: 10px 14px; background: #EFF6FF; border-radius: 4px; margin-bottom: 10px; }
    .agent-card-quant { border-left: 4px solid #8B5CF6; padding: 10px 14px; background: #F5F3FF; border-radius: 4px; margin-bottom: 10px; }
    .agent-card-risk { border-left: 4px solid #EF4444; padding: 10px 14px; background: #FEF2F2; border-radius: 4px; margin-bottom: 10px; }
    .agent-card-pm { border-left: 4px solid #10B981; padding: 10px 14px; background: #ECFDF5; border-radius: 4px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 FinAsset-Agent: 资管投委会多智能体协同投研系统</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">面向证券、财富管理与基金资管场景：多智能体博弈（Supervisor / 研报CRAG / Black-Litterman贝叶斯量化 / 独立风控官反思回退）</div>', unsafe_allow_html=True)

# 侧边栏：参数配置
st.sidebar.header("⚙️ 资管投委会参数配置")

default_assets = ["科技算力", "新能源", "大消费", "高股息银行", "避险黄金"]
selected_assets = st.sidebar.multiselect(
    "选择备选资产池 (Asset Pool)",
    options=default_assets,
    default=default_assets
)

if len(selected_assets) < 2:
    st.sidebar.error("资产池中至少需保留 2 种资产以进行组合优化！")
    st.stop()

risk_tolerance = st.sidebar.selectbox(
    "委托人风险偏好画像",
    ["稳健成长型 (Balanced Growth)", "保守防御型 (Conservative Defense)", "积极进取型 (Aggressive Alpha)"]
)

horizon = st.sidebar.selectbox(
    "考核与投资周期",
    ["12个月 (Annual Review)", "6个月 (Half-year)", "中长期 (3年周期)"]
)

max_weight_cap = st.sidebar.slider(
    "单一资产集中度上限红线 (%)",
    min_value=20, max_value=50, value=35, step=5
)
config.MAX_SINGLE_ASSET_WEIGHT = max_weight_cap / 100.0

st.sidebar.subheader("🤖 大模型服务 (可选)")
api_key_input = st.sidebar.text_input("OpenAI / DeepSeek API Key", type="password", help="留空则自动运行本地金融工程级离线渲染引擎")
if api_key_input:
    config.OPENAI_API_KEY = api_key_input
    custom_base_url = st.sidebar.text_input("Base URL", value="https://api.deepseek.com/v1")
    config.OPENAI_BASE_URL = custom_base_url

run_button = st.sidebar.button("🚀 召开资管投委会评审会", type="primary", use_container_width=True)

# 历史行情读取与基准展示
base_dir = os.path.dirname(os.path.abspath(__file__))
market_csv = os.path.join(base_dir, "data", "market_prices.csv")
price_df = pd.read_csv(market_csv, index_col=0) if os.path.exists(market_csv) else pd.DataFrame()

# 状态缓存
if "final_state" not in st.session_state:
    st.session_state.final_state = None

if run_button:
    with st.status("正在召开资管投资决策委员会评审会...", expanded=True) as status:
        st.write("📋 投委会总指挥 (Supervisor) 正在下发分析任务...")
        app = build_finasset_graph()
        initial_state = {
            "assets": selected_assets,
            "risk_tolerance": risk_tolerance,
            "investment_horizon": horizon,
            "benchmark_name": "市场基准",
            "step_logs": [],
            "debate_transcript": [],
            "rejection_count": 0,
            "risk_aversion_penalty": 0.0
        }
        res_state = app.invoke(initial_state)
        for log in res_state.get("step_logs", []):
            st.write(log)
        status.update(label="✅ 投委会审议完毕，投资备忘录签署成功！", state="complete", expanded=False)
        st.session_state.final_state = res_state

# 若已有运行结果，渲染全维度看板
state = st.session_state.final_state
if state is not None:
    p_ret = state.get("portfolio_annual_return", 0.0)
    p_vol = state.get("portfolio_annual_volatility", 0.0)
    sharpe = state.get("portfolio_sharpe_ratio", 0.0)
    var_95 = state.get("var_95", 0.0)
    mdd = state.get("historical_max_drawdown", 0.0)

    # 顶层 KPI 核心看板
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f'<div class="metric-box"><div class="metric-val">+{p_ret*100:.2f}%</div><div class="metric-lbl">预期年化收益率</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{p_vol*100:.2f}%</div><div class="metric-lbl">预期年化波动率</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{sharpe:.2f}</div><div class="metric-lbl">夏普比率 (Rf=2.5%)</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{var_95*100:.2f}%</div><div class="metric-lbl">95% 日度在险价值 (VaR)</div></div>', unsafe_allow_html=True)
    with k5:
        st.markdown(f'<div class="metric-box"><div class="metric-val">{mdd*100:.2f}%</div><div class="metric-lbl">历史最大回撤</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 资产配置权重与有效前沿", 
        "📈 历史资产行情走势", 
        "🎲 蒙特卡洛压力测试 (60日)", 
        "🗣️ 投委会多智能体辩论纪要", 
        "📑 投资决策备忘录与报告"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            weights = state.get("optimal_weights", {})
            w_df = pd.DataFrame([{"Asset": k, "Weight": v * 100} for k, v in weights.items() if v > 0.001])
            fig_pie = px.pie(
                w_df, values="Weight", names="Asset", 
                title="Black-Litterman 贝叶斯后验最优权重配置",
                hole=0.45,
                color_discrete_sequence=px.colors.qualitative.Safe
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c2:
            frontier_data = state.get("efficient_frontier_points", [])
            if frontier_data:
                f_df = pd.DataFrame(frontier_data)
                fig_front = go.Figure()
                # 绘制有效前沿平滑曲线
                fig_front.add_trace(go.Scatter(
                    x=f_df["volatility"], y=f_df["return"],
                    mode="lines+markers",
                    name="马科维茨有效前沿 (Efficient Frontier)",
                    line=dict(color="#3B82F6", width=2.5),
                    marker=dict(size=4)
                ))
                # 标注当前最优切点组合 (Tangency Portfolio)
                fig_front.add_trace(go.Scatter(
                    x=[p_vol * 100], y=[p_ret * 100],
                    mode="markers+text",
                    name="当前最优配置点",
                    marker=dict(symbol="star", size=16, color="#EF4444"),
                    text=["★ 投委会最优配置点"],
                    textposition="top center"
                ))
                fig_front.update_layout(
                    title="马科维茨有效前沿与风险收益定位",
                    xaxis_title="年化波动率 (%)",
                    yaxis_title="预期年化收益率 (%)",
                    height=420,
                    margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_front, use_container_width=True)

    with tab2:
        if not price_df.empty:
            # 归一化初始价格为 1.000 绘制收益率走势
            norm_prices = (price_df[selected_assets] / price_df[selected_assets].iloc[0]).round(3)
            fig_line = px.line(
                norm_prices, 
                title="资产池过去 252 个交易日相对累积收益走势 (基准点 = 1.00)",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_line.update_layout(
                height=450, 
                xaxis_title="交易日", 
                yaxis_title="归一化净值",
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_line, use_container_width=True)

    with tab3:
        mc = state.get("monte_carlo_simulation", {})
        if mc:
            days = mc.get("days", [])
            fig_mc = go.Figure()
            # 添加分位数扇形带
            fig_mc.add_trace(go.Scatter(
                x=days + days[::-1],
                y=mc["p95"] + mc["p5"][::-1],
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name="90% 置信区间 (5% - 95%)"
            ))
            fig_mc.add_trace(go.Scatter(
                x=days + days[::-1],
                y=mc["p75"] + mc["p25"][::-1],
                fill='toself',
                fillcolor='rgba(59, 130, 246, 0.3)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                showlegend=True,
                name="50% 置信核心带 (25% - 75%)"
            ))
            fig_mc.add_trace(go.Scatter(
                x=days, y=mc["p50"],
                mode="lines",
                line=dict(color="#1E3A8A", width=2.5),
                name="中位数预测路径 (p50)"
            ))
            # 绘制 3 条随机路径
            for s_idx, sample_path in enumerate(mc.get("sample_paths", [])[:3]):
                fig_mc.add_trace(go.Scatter(
                    x=days, y=sample_path,
                    mode="lines",
                    line=dict(dash="dot", width=1, color="#94A3B8"),
                    showlegend=(s_idx == 0),
                    name="模拟随机抽样路径"
                ))

            fig_mc.update_layout(
                title="未来 60 个交易日蒙特卡洛压力测试 (1000 次随机游走情景扇形图)",
                xaxis_title="前瞻交易日数",
                yaxis_title="模拟预测净值 (初始 = 1.000)",
                height=450,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_mc, use_container_width=True)

            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("极端悲观下限净值 (5%分位)", f"{mc.get('stress_min_nav', 0):.3f}")
            with m_col2:
                st.metric("中位数预期净值 (p50)", f"{mc.get('expected_median_nav', 0):.3f}")
            with m_col3:
                st.metric("60日出现浮亏概率", f"{mc.get('loss_probability', 0)*100:.1f}%")

    with tab4:
        st.subheader("🗣️ 资管投资决策委员会博弈对抗纪要")
        transcript = state.get("debate_transcript", [])
        for item in transcript:
            speaker = item.get("speaker", "")
            msg = item.get("message", "").replace("\n", "<br>")
            
            card_class = "agent-card-supervisor"
            if "Alpha" in speaker or "投研" in speaker:
                card_class = "agent-card-alpha"
            elif "Quant" in speaker or "量化" in speaker:
                card_class = "agent-card-quant"
            elif "Risk" in speaker or "风控" in speaker:
                card_class = "agent-card-risk"
            elif "Portfolio" in speaker or "投资经理" in speaker:
                card_class = "agent-card-pm"

            st.markdown(f'<div class="{card_class}"><b>{speaker}</b><br>{msg}</div>', unsafe_allow_html=True)

    with tab5:
        memo_content = state.get("final_memo", "")
        st.markdown(memo_content)
        st.download_button(
            label="📥 导出完整《资产管理投资决策备忘录》 (Markdown)",
            data=memo_content,
            file_name=f"investment_memo_{risk_tolerance}.md",
            mime="text/markdown"
        )
else:
    st.info("👈 请在左侧侧边栏配置资产池与偏好，点击 **“🚀 召开资管投委会评审会”** 启动多智能体协同仿真。")
