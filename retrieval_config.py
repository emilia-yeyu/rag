#!/usr/bin/env python3
"""
检索配置模块
定义不同的检索模式和相关配置
"""
from enum import Enum
from typing import Dict, Any, List
from dataclasses import dataclass


class RetrievalMode(Enum):
    """检索模式枚举"""
    VECTOR_ONLY = "vector_only"           # 仅向量检索
    BM25_ONLY = "bm25_only"              # 仅BM25检索
    SQL_ONLY = "sql_only"                # 仅SQL检索
    VECTOR_BM25 = "vector_bm25"          # 向量+BM25混合
    VECTOR_SQL = "vector_sql"            # 向量+SQL混合
    BM25_SQL = "bm25_sql"                # BM25+SQL混合
    ALL_HYBRID = "all_hybrid"            # 三种检索混合


@dataclass
class RetrievalConfig:
    """检索配置类"""
    mode: RetrievalMode
    
    # 基础参数
    k: int = 5                           # 最终返回结果数
    
    # 向量检索参数
    enable_vector: bool = True
    vector_k: int = 10
    vector_weight: float = 0.6
    vector_search_type: str = "similarity"  # similarity, mmr, similarity_score_threshold
    
    # BM25检索参数
    enable_bm25: bool = True
    bm25_k: int = 10
    bm25_weight: float = 0.3
    bm25_min_match_ratio: float = 0.2
    bm25_score_threshold: float = 0.001
    
    # SQL检索参数
    enable_sql: bool = True
    sql_k: int = 8
    sql_weight: float = 0.2
    
    # RRF融合参数
    rrf_k: int = 60
    
    def __post_init__(self):
        """初始化后处理，根据模式设置启用状态"""
        if self.mode == RetrievalMode.VECTOR_ONLY:
            self.enable_vector = True
            self.enable_bm25 = False
            self.enable_sql = False
            self.vector_weight = 1.0
            
        elif self.mode == RetrievalMode.BM25_ONLY:
            self.enable_vector = False
            self.enable_bm25 = True
            self.enable_sql = False
            self.bm25_weight = 1.0
            
        elif self.mode == RetrievalMode.SQL_ONLY:
            self.enable_vector = False
            self.enable_bm25 = False
            self.enable_sql = True
            self.sql_weight = 1.0
            
        elif self.mode == RetrievalMode.VECTOR_BM25:
            self.enable_vector = True
            self.enable_bm25 = True
            self.enable_sql = False
            # 重新分配权重
            total_weight = self.vector_weight + self.bm25_weight
            self.vector_weight = self.vector_weight / total_weight
            self.bm25_weight = self.bm25_weight / total_weight
            
        elif self.mode == RetrievalMode.VECTOR_SQL:
            self.enable_vector = True
            self.enable_bm25 = False
            self.enable_sql = True
            # 重新分配权重
            total_weight = self.vector_weight + self.sql_weight
            self.vector_weight = self.vector_weight / total_weight
            self.sql_weight = self.sql_weight / total_weight
            
        elif self.mode == RetrievalMode.BM25_SQL:
            self.enable_vector = False
            self.enable_bm25 = True
            self.enable_sql = True
            # 重新分配权重
            total_weight = self.bm25_weight + self.sql_weight
            self.bm25_weight = self.bm25_weight / total_weight
            self.sql_weight = self.sql_weight / total_weight
            
        elif self.mode == RetrievalMode.ALL_HYBRID:
            self.enable_vector = True
            self.enable_bm25 = True
            self.enable_sql = True
            # 权重已在初始化时设置，无需调整
            
    def get_description(self) -> str:
        """获取配置描述"""
        descriptions = {
            RetrievalMode.VECTOR_ONLY: "仅向量语义检索",
            RetrievalMode.BM25_ONLY: "仅BM25关键词检索",
            RetrievalMode.SQL_ONLY: "仅SQL精确检索",
            RetrievalMode.VECTOR_BM25: "向量检索 + BM25检索",
            RetrievalMode.VECTOR_SQL: "向量检索 + SQL检索",
            RetrievalMode.BM25_SQL: "BM25检索 + SQL检索",
            RetrievalMode.ALL_HYBRID: "向量检索 + BM25检索 + SQL检索"
        }
        return descriptions.get(self.mode, "未知模式")
    
    def get_enabled_methods(self) -> List[str]:
        """获取启用的检索方法列表"""
        methods = []
        if self.enable_vector:
            methods.append("向量检索")
        if self.enable_bm25:
            methods.append("BM25检索")
        if self.enable_sql:
            methods.append("SQL检索")
        return methods


# 预定义配置
PRESET_CONFIGS = {
    "semantic": RetrievalConfig(
        mode=RetrievalMode.VECTOR_ONLY,
        k=5,
        vector_k=10,
        vector_search_type="similarity"
    ),
    
    "keyword": RetrievalConfig(
        mode=RetrievalMode.BM25_ONLY,
        k=5,
        bm25_k=10,
        bm25_min_match_ratio=0.1,
        bm25_score_threshold=0.001
    ),
    
    "structured": RetrievalConfig(
        mode=RetrievalMode.SQL_ONLY,
        k=10,
        sql_k=10
    ),
    
    "semantic_keyword": RetrievalConfig(
        mode=RetrievalMode.VECTOR_BM25,
        k=5,
        vector_k=8,
        bm25_k=8,
        vector_weight=0.7,
        bm25_weight=0.3
    ),
    
    "semantic_structured": RetrievalConfig(
        mode=RetrievalMode.VECTOR_SQL,
        k=5,
        vector_k=8,
        sql_k=6,
        vector_weight=0.6,
        sql_weight=0.4
    ),
    
    "keyword_structured": RetrievalConfig(
        mode=RetrievalMode.BM25_SQL,
        k=5,
        bm25_k=8,
        sql_k=6,
        bm25_weight=0.5,
        sql_weight=0.5
    ),
    
    "comprehensive": RetrievalConfig(
        mode=RetrievalMode.ALL_HYBRID,
        k=5,
        vector_k=8,
        bm25_k=8,
        sql_k=6,
        vector_weight=0.5,
        bm25_weight=0.3,
        sql_weight=0.2
    )
}


def get_config(config_name: str) -> RetrievalConfig:
    """
    获取预定义配置
    
    Args:
        config_name: 配置名称
        
    Returns:
        检索配置对象
    """
    if config_name not in PRESET_CONFIGS:
        raise ValueError(f"未知配置: {config_name}. 可用配置: {list(PRESET_CONFIGS.keys())}")
    
    return PRESET_CONFIGS[config_name]


def list_configs() -> Dict[str, str]:
    """
    列出所有可用配置及其描述
    
    Returns:
        配置名称到描述的映射
    """
    return {
        name: config.get_description() 
        for name, config in PRESET_CONFIGS.items()
    } 