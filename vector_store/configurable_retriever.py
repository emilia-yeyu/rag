#!/usr/bin/env python3
"""
可配置检索器模块
支持不同检索策略的灵活组合
"""
import time
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document
from collections import defaultdict

# 导入检索配置
from retrieval_config import RetrievalConfig, RetrievalMode

# 导入各种检索器
from vector_store.hybrid_retriever import RRFFusion, BM25Retriever

# 导入SQL检索器
try:
    from sql_retriever import SQLRetriever
    SQL_AVAILABLE = True
except ImportError:
    print("⚠️ SQL检索器模块未找到，SQL检索功能将不可用")
    SQL_AVAILABLE = False


class ConfigurableRetriever:
    """
    可配置检索器：支持多种检索策略的灵活组合
    """
    
    def __init__(self, vector_store_manager, config: RetrievalConfig):
        """
        初始化可配置检索器
        
        Args:
            vector_store_manager: 向量存储管理器
            config: 检索配置
        """
        self.vector_store = vector_store_manager
        self.config = config
        
        # 初始化各种检索器
        self.bm25_retriever = None
        self.sql_retriever = None
        self.rrf_fusion = RRFFusion(k=config.rrf_k)
        
        print(f"🔧 初始化可配置检索器...")
        print(f"📋 检索模式: {config.get_description()}")
        print(f"🎯 启用的方法: {', '.join(config.get_enabled_methods())}")
        
        # 根据配置初始化所需的检索器
        self._init_retrievers()
    
    def _init_retrievers(self):
        """根据配置初始化检索器"""
        # 初始化BM25检索器
        if self.config.enable_bm25:
            self._init_bm25()
        
        # 初始化SQL检索器
        if self.config.enable_sql:
            self._init_sql()
    
    def _init_bm25(self):
        """初始化BM25检索器"""
        try:
            # 获取所有文档
            if hasattr(self.vector_store, '_vector_store') and self.vector_store._vector_store:
                # 从chromadb获取所有文档
                all_docs = self.vector_store._vector_store.get()
                
                if all_docs and 'documents' in all_docs:
                    documents = []
                    metadatas = all_docs.get('metadatas', [])
                    
                    for i, content in enumerate(all_docs['documents']):
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        doc = Document(page_content=content, metadata=metadata)
                        documents.append(doc)
                    
                    self.bm25_retriever = BM25Retriever(
                        documents, 
                        min_match_ratio=self.config.bm25_min_match_ratio,
                        score_threshold=self.config.bm25_score_threshold
                    )
                    print(f"✅ BM25检索器初始化成功，共{len(documents)}个文档")
                else:
                    print("⚠️ 向量存储为空，无法初始化BM25检索器")
            else:
                print("⚠️ 向量存储未初始化，无法创建BM25检索器")
                
        except Exception as e:
            print(f"❌ BM25检索器初始化失败: {e}")
            self.bm25_retriever = None
    
    def _init_sql(self):
        """初始化SQL检索器"""
        if not SQL_AVAILABLE:
            print("⚠️ SQL检索器不可用，跳过初始化")
            return
            
        try:
            self.sql_retriever = SQLRetriever()
            # 获取数据库统计信息
            stats = self.sql_retriever.get_stats()
            print(f"✅ SQL检索器初始化成功，数据库记录: {stats.get('total_records', 0)}条")
            
        except Exception as e:
            print(f"❌ SQL检索器初始化失败: {e}")
            self.sql_retriever = None
    
    def search(self, query: str) -> List[Tuple[Document, float]]:
        """
        执行检索
        
        Args:
            query: 查询文本
            
        Returns:
            检索结果
        """
        print(f"🔍 开始检索 ({self.config.get_description()})...")
        start_time = time.time()
        
        rankings = []
        weights = []
        
        # 1. 向量检索
        if self.config.enable_vector:
            vector_results = self._vector_search(query)
            if vector_results:
                rankings.append(vector_results)
                weights.append(self.config.vector_weight)
        
        # 2. BM25检索
        if self.config.enable_bm25:
            bm25_results = self._bm25_search(query)
            if bm25_results:
                rankings.append(bm25_results)
                weights.append(self.config.bm25_weight)
        
        # 3. SQL检索
        if self.config.enable_sql:
            sql_results = self._sql_search(query)
            if sql_results:
                rankings.append(sql_results)
                weights.append(self.config.sql_weight)
        
        # 4. 融合结果
        final_results = self._fuse_results(rankings, weights)
        
        search_time = time.time() - start_time
        print(f"✅ 检索完成，找到{len(final_results)}个结果，耗时: {search_time:.2f}秒")
        
        return final_results
    
    def _vector_search(self, query: str) -> Optional[List[Tuple[Document, float]]]:
        """向量检索"""
        try:
            print(f"🔍 执行向量检索 (k={self.config.vector_k})...")
            
            if self.config.vector_search_type == "similarity":
                vector_results = self.vector_store.search_similarity(query, k=self.config.vector_k)
            elif self.config.vector_search_type == "mmr":
                vector_results = self.vector_store.search_mmr(query, k=self.config.vector_k)
            elif self.config.vector_search_type == "similarity_score_threshold":
                vector_results = self.vector_store.search_with_threshold(query, similarity_threshold=0.5)
            else:
                vector_results = self.vector_store.search_similarity(query, k=self.config.vector_k)
            
            if vector_results:
                # 转换为(Document, score)格式
                vector_ranking = [(doc, 1.0) for doc in vector_results]
                print(f"📊 向量检索找到 {len(vector_results)} 个文档")
                return vector_ranking
            else:
                print("⚠️ 向量检索无结果")
                return None
                
        except Exception as e:
            print(f"❌ 向量检索失败: {e}")
            return None
    
    def _bm25_search(self, query: str) -> Optional[List[Tuple[Document, float]]]:
        """BM25检索"""
        if not self.bm25_retriever:
            print("⚠️ BM25检索器未初始化，跳过BM25检索")
            return None
            
        try:
            print(f"🔍 执行BM25检索 (k={self.config.bm25_k})...")
            bm25_results = self.bm25_retriever.search(query, k=self.config.bm25_k)
            
            if bm25_results:
                # 过滤掉分数为0的结果
                bm25_ranking = [(doc, score) for doc, score in bm25_results if score > 0]
                if bm25_ranking:
                    print(f"📊 BM25检索找到 {len(bm25_ranking)} 个相关文档")
                    return bm25_ranking
                else:
                    print("⚠️ BM25检索无相关结果")
                    return None
            else:
                print("⚠️ BM25检索无结果")
                return None
                
        except Exception as e:
            print(f"❌ BM25检索失败: {e}")
            return None
    
    def _sql_search(self, query: str) -> Optional[List[Tuple[Document, float]]]:
        """SQL检索"""
        if not self.sql_retriever:
            print("⚠️ SQL检索器未初始化，跳过SQL检索")
            return None
            
        try:
            print(f"🔍 执行SQL检索 (k={self.config.sql_k})...")
            sql_results = self.sql_retriever.search(query, k=self.config.sql_k)
            
            if sql_results:
                # SQL结果已经是(Document, score)格式
                sql_ranking = [(doc, score) for doc, score in sql_results if score > 0]
                if sql_ranking:
                    print(f"📊 SQL检索找到 {len(sql_ranking)} 个精确匹配")
                    return sql_ranking
                else:
                    print("⚠️ SQL检索无相关结果")
                    return None
            else:
                print("⚠️ SQL检索无结果")
                return None
                
        except Exception as e:
            print(f"❌ SQL检索失败: {e}")
            return None
    
    def _fuse_results(self, rankings: List[List[Tuple[Document, float]]], weights: List[float]) -> List[Tuple[Document, float]]:
        """融合检索结果"""
        if not rankings:
            print("❌ 所有检索方法都失败了")
            return []
        
        if len(rankings) == 1:
            print("⚠️ 只有一种检索方法有结果，无需融合")
            final_results = rankings[0][:self.config.k]
        else:
            print(f"🔄 使用RRF融合 {len(rankings)} 个检索结果...")
            
            # 打印融合前的统计信息
            method_names = []
            if self.config.enable_vector:
                method_names.append("向量检索")
            if self.config.enable_bm25:
                method_names.append("BM25检索")
            if self.config.enable_sql:
                method_names.append("SQL检索")
            
            for i, (ranking, weight) in enumerate(zip(rankings, weights)):
                method_name = method_names[i] if i < len(method_names) else f"检索方法{i+1}"
                print(f"  📊 {method_name}: {len(ranking)}个结果，权重: {weight:.2f}")
            
            fused_results = self.rrf_fusion.fuse_rankings(rankings, weights)
            final_results = fused_results[:self.config.k]
            print(f"✅ RRF融合完成，返回 {len(final_results)} 个文档")
            
            # 打印融合后的来源统计
            self._print_source_stats(final_results)
        
        return final_results
    
    def _print_source_stats(self, results: List[Tuple[Document, float]]):
        """打印来源统计"""
        source_stats = defaultdict(int)
        for doc, score in results:
            source_type = doc.metadata.get('search_type', 'unknown')
            if source_type == 'unknown':
                # 根据metadata推断来源
                if 'sql_query' in doc.metadata:
                    source_type = 'sql'
                elif 'rrf_score' in doc.metadata:
                    source_type = 'hybrid'
                else:
                    source_type = 'vector'
            source_stats[source_type] += 1
        
        if source_stats:
            print(f"📊 最终结果来源分布: ", end="")
            for source_type, count in source_stats.items():
                source_name = {"sql": "SQL", "vector": "向量", "bm25": "BM25", "hybrid": "混合"}.get(source_type, source_type)
                print(f"{source_name}: {count}个", end="  ")
            print()
    
    def get_config_summary(self) -> Dict[str, any]:
        """获取配置摘要"""
        return {
            "mode": self.config.mode.value,
            "description": self.config.get_description(),
            "enabled_methods": self.config.get_enabled_methods(),
            "final_k": self.config.k,
            "weights": {
                "vector": self.config.vector_weight if self.config.enable_vector else 0,
                "bm25": self.config.bm25_weight if self.config.enable_bm25 else 0,
                "sql": self.config.sql_weight if self.config.enable_sql else 0
            }
        } 