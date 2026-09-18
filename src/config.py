"""
FinAsset-Agent 全局配置与量化金融超参数
包含宏观基准无风险收益率、Black-Litterman 参数、风控阈值及大模型配置
"""

import os
from dotenv import load_dotenv

load_dotenv()

# 大模型配置 (支持 OpenAI / DeepSeek / 通义千问等格式)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-chat")

# 金融工程与资产配置核心超参数
RISK_FREE_RATE = float(os.getenv("RISK_FREE_RATE", "0.025"))     # 无风险收益率 (2.5% 年化，参考国债基准)
DEFAULT_DELTA = float(os.getenv("RISK_AVERSION_DELTA", "2.5"))   # 市场风险厌恶系数 (Risk Aversion Parameter)
TAU = float(os.getenv("TAU", "0.05"))                            # Black-Litterman 标量不确定性系数 tau

# 机构级合规与风控硬门禁阈值
MAX_SINGLE_ASSET_WEIGHT = float(os.getenv("MAX_SINGLE_ASSET_WEIGHT", "0.35")) # 单一资产最大权重敞口 (35%)
MAX_ALLOWED_VAR_95 = float(os.getenv("MAX_ALLOWED_VAR_95", "0.025"))          # 95% 日度在险价值 VaR 上限 (2.5%)
MAX_RISK_REJECTIONS = int(os.getenv("MAX_RISK_REJECTIONS", "2"))               # 风控最大允许驳回重试次数 (避免死循环)

# 蒙特卡洛压力测试参数
MONTE_CARLO_SIMS = int(os.getenv("MONTE_CARLO_SIMS", "1000"))    # 随机路径模拟条数
MONTE_CARLO_DAYS = int(os.getenv("MONTE_CARLO_DAYS", "60"))      # 前瞻模拟预测交易日数
