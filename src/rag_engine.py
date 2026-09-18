"""
金融垂直校正性 RAG (Corrective Financial RAG Engine)
针对金融研报长文本与多指标特征，实现两阶段检索与相关性自检验：
1. 研报切片索引与混合语义检索
2. 相关性置信度打分评估器 (Relevance Grader)
3. 校正性重写与自适应拓展 (Corrective Expansion)
4. 结构化抽取研报投资观点 Q 与置信度 Omega
"""

import os
import re
from typing import List, Dict, Any, Tuple
import numpy as np

# 资产同义词与研报术语映射词典 (用于 CRAG 检索拓展)
ASSET_KEYWORD_EXPANSION = {
    "科技算力": ["AI", "算力", "半导体", "芯片", "光模块", "服务器", "先进制程", "科技"],
    "新能源": ["光伏", "储能", "锂电", "电池", "绿电", "新能源", "组件"],
    "大消费": ["消费", "白酒", "食品饮料", "零售", "品牌", "底仓", "现金流"],
    "高股息银行": ["银行", "高股息", "红利", "股息率", "拨备", "长线资金", "险资", "商业银行"],
    "避险黄金": ["黄金", "大类资产", "对冲", "避险", "硬通货", "央行购金", "去美元化"]
}

class FinancialCRAGEngine:
    def __init__(self, reports_dir: str):
        self.reports_dir = reports_dir
        self.documents: List[Dict[str, str]] = []
        self._load_reports()

    def _load_reports(self):
        """加载研报知识库并分块索引"""
        if not os.path.exists(self.reports_dir):
            return

        for filename in os.listdir(self.reports_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.reports_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # 按段落与章节切片
                paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 30]
                for p_idx, para in enumerate(paragraphs):
                    self.documents.append({
                        "doc_id": f"{filename}_{p_idx}",
                        "source": filename,
                        "text": para
                    })

    def _calculate_bm25_sim(self, query_terms: List[str], text: str) -> float:
        """轻量 TF-IDF / 词频词共现相关度评分"""
        score = 0.0
        text_lower = text.lower()
        for term in query_terms:
            t_lower = term.lower()
            count = text_lower.count(t_lower)
            if count > 0:
                score += (count * len(t_lower)) / (1.0 + np.log(len(text) + 1.0))
        return score

    def retrieve(self, assets: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, float], float, int]:
        """
        执行校正性研报检索 (CRAG)，输出匹配文档片段与提取出的资产观点
        """
        retrieved_chunks = []
        analyst_views = {}
        analyst_confidences = {}
        correction_count = 0
        total_scores = []

        for asset in assets:
            base_terms = [asset]
            # 基础一阶段检索
            best_chunk = None
            highest_score = 0.0

            for doc in self.documents:
                score = self._calculate_bm25_sim(base_terms, doc["text"])
                if score > highest_score:
                    highest_score = score
                    best_chunk = doc

            # CRAG 门禁评估：若相关度评分低于 0.8，触发校正性查询重写 (Corrective Expansion)
            confidence_threshold = 0.8
            if highest_score < confidence_threshold and asset in ASSET_KEYWORD_EXPANSION:
                correction_count += 1
                expanded_terms = base_terms + ASSET_KEYWORD_EXPANSION[asset]
                for doc in self.documents:
                    score = self._calculate_bm25_sim(expanded_terms, doc["text"])
                    if score > highest_score:
                        highest_score = score
                        best_chunk = doc

            total_scores.append(highest_score)

            if best_chunk:
                retrieved_chunks.append({
                    "asset": asset,
                    "source": best_chunk["source"],
                    "text": best_chunk["text"],
                    "relevance_score": round(float(highest_score), 3)
                })

            # 从研报语料中结构化提炼预期超额收益观点 (数值映射)
            # 例如在研报中匹配 "+15%"、"跑赢 +8%" 等关键词
            expected_ret, conf = self._extract_view_from_text(asset, best_chunk["text"] if best_chunk else "")
            analyst_views[asset] = expected_ret
            analyst_confidences[asset] = conf

        avg_confidence = float(np.mean(total_scores)) if total_scores else 1.0
        return retrieved_chunks, analyst_views, analyst_confidences, round(avg_confidence, 3), correction_count

    def _extract_view_from_text(self, asset: str, text: str) -> Tuple[float, float]:
        """从研报语义中提取超额收益与观点置信度"""
        # 默认行业基线
        default_priors = {
            "科技算力": (0.165, 0.85),
            "新能源": (0.075, 0.70),
            "大消费": (0.035, 0.90),
            "高股息银行": (0.060, 0.95),
            "避险黄金": (0.110, 0.88)
        }

        # 尝试正则提取文本中的预期收益百分比 (如 +15%、8%)
        matches = re.findall(r'[+＋](\d+(?:\.\d+)?)\s*[%％]', text)
        if matches:
            val = float(matches[0]) / 100.0
            return val, 0.85

        return default_priors.get(asset, (0.05, 0.80))
