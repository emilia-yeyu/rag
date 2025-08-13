# RAG/embedding/reranker.py
import os
import importlib
from typing import List, Tuple, Dict, Any
from langchain_core.documents import Document

class RerankAdapterError(Exception):
    """Rerank适配器相关的自定义错误。"""
    pass

class RerankAdapter:
    """
    重排序适配器，用于对检索结果进行重新排序以提升检索精度
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称，默认使用 BAAI/bge-reranker-v2-m3
        """
        self.model_name = model_name
        self.reranker = None
        self._init_reranker()
    
    def _init_reranker(self):
        """初始化重排序模型"""
        try:
            # 导入 FlagEmbedding
            from FlagEmbedding import FlagReranker
            
            # 获取项目根目录并设置模型缓存目录
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 回到 RAG 目录
            cache_dir = os.path.join(script_dir, "models", "rerankers")
            os.makedirs(cache_dir, exist_ok=True)
            
            print(f"🔄 正在加载重排序模型: {self.model_name}")
            print(f"📁 模型缓存目录: {cache_dir}")
            print(f"⚠️ 首次使用会下载模型到指定目录，请耐心等待...")
            
            # 初始化重排序器
            self.reranker = FlagReranker(
                self.model_name,
                use_fp16=True,  # 使用半精度，节省内存
                device='cpu',   # 可以改为 'cuda' 如果有GPU
                cache_dir=cache_dir,  # 指定模型缓存目录
            )
            
            print(f"✅ 重排序模型加载成功: {self.model_name}")
            
        except ImportError:
            raise RerankAdapterError(
                "缺少 FlagEmbedding 库。请安装: pip install FlagEmbedding"
            )
        except Exception as e:
            raise RerankAdapterError(f"初始化重排序模型失败: {e}")
    
    def rerank_documents(
        self, 
        query: str, 
        documents: List[Document], 
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        对文档进行重新排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前k个结果，None表示返回全部
        
        Returns:
            重新排序后的 (文档, 相关性分数) 列表
        """
        if not documents:
            return []
        
        if not self.reranker:
            raise RerankAdapterError("重排序模型未初始化")
        
        try:
            print(f"🔄 正在重排序 {len(documents)} 个文档...")
            
            # 准备query-document对
            pairs = []
            for doc in documents:
                pairs.append([query, doc.page_content])
            
            # 计算相关性分数
            scores = self.reranker.compute_score(pairs)
            
            # 处理不同格式的分数
            import numpy as np
            
            # 如果是单个值，转换为列表
            if not isinstance(scores, (list, np.ndarray)):
                scores = [scores]
            elif isinstance(scores, np.ndarray):
                scores = scores.tolist()  # 将 numpy 数组转换为 Python 列表
            
            # 组合文档和分数
            doc_scores = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                # 安全地处理分数转换
                try:
                    if hasattr(score, 'item'):  # numpy scalar
                        score_float = float(score.item())
                    elif isinstance(score, (int, float)):
                        score_float = float(score)
                    else:
                        score_float = float(score)
                except (ValueError, TypeError) as e:
                    print(f"⚠️ 分数转换失败 (doc {i}): {score}, 类型: {type(score)}")
                    score_float = 0.0
                
                doc_scores.append((doc, score_float))
            
            # 按分数降序排序
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 返回top_k个结果
            if top_k:
                doc_scores = doc_scores[:top_k]
            
            print(f"✅ 重排序完成，返回 {len(doc_scores)} 个结果")
            print(f"📊 分数范围: {doc_scores[-1][1]:.3f} ~ {doc_scores[0][1]:.3f}")
            
            return doc_scores
            
        except Exception as e:
            print(f"❌ 重排序失败: {e}")
            # 降级：返回原始文档，分数为0
            return [(doc, 0.0) for doc in documents]
    
    def rerank_and_filter(
        self, 
        query: str, 
        documents: List[Document], 
        threshold: float = 0.0,
        top_k: int = 5
    ) -> List[Document]:
        """
        重排序并过滤文档
        
        Args:
            query: 查询文本
            documents: 文档列表
            threshold: 相关性分数阈值
            top_k: 返回前k个结果
        
        Returns:
            过滤后的文档列表
        """
        reranked = self.rerank_documents(query, documents, top_k)
        
        # 过滤低分文档
        filtered = []
        for doc, score in reranked:
            if score >= threshold:
                # 将分数添加到文档元数据
                doc.metadata['rerank_score'] = score
                filtered.append(doc)
        
        print(f"🔍 阈值过滤: {len(reranked)} -> {len(filtered)} 个文档 (阈值: {threshold})")
        
        return filtered

class AdaptiveReranker:
    """
    自适应重排序器，根据查询类型选择是否使用rerank
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.reranker = RerankAdapter(model_name)
        
        # 需要精确匹配的查询关键词
        self.precision_keywords = [
            "处罚", "规定", "制度", "时间", "政策", "标准", 
            "要求", "条件", "流程", "程序", "多少", "几个",
            "什么时候", "如何", "怎么", "为什么"
        ]
    
    def should_rerank(self, query: str) -> bool:
        """
        判断是否需要重排序
        
        Args:
            query: 查询文本
            
        Returns:
            是否需要重排序
        """
        # 如果查询包含精确匹配关键词，使用rerank
        query_lower = query.lower()
        for keyword in self.precision_keywords:
            if keyword in query_lower:
                return True
        
        # 如果查询较短(<5字符)，通常不需要rerank
        if len(query) < 5:
            return False
            
        # 默认对中等长度查询使用rerank
        return len(query) >= 5
    
    def adaptive_rerank(
        self, 
        query: str, 
        documents: List[Document], 
        force_rerank: bool = False
    ) -> List[Document]:
        """
        自适应重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            force_rerank: 强制使用重排序
        
        Returns:
            处理后的文档列表
        """
        if not documents:
            return documents
        
        # 决定是否使用rerank
        use_rerank = force_rerank or self.should_rerank(query)
        
        if use_rerank:
            print(f"🎯 启用重排序 (查询: '{query[:20]}...')")
            return self.reranker.rerank_and_filter(
                query=query,
                documents=documents,
                threshold=0.1,  # 较低的阈值，保留更多结果
                top_k=5
            )
        else:
            print(f"⚡ 跳过重排序 (查询: '{query[:20]}...')")
            return documents[:5]  # 直接返回前5个 