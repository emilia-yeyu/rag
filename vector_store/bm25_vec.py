# RAG/vector_store/hybrid_retriever.py
import math
from typing import List, Dict, Tuple, Optional, Set
from langchain_core.documents import Document
from collections import defaultdict

class RRFFusion:
    """
    倒数排名融合（Reciprocal Rank Fusion）算法实现
    用于融合多个检索结果
    """
    
    def __init__(self, k: int = 60):
        """
        初始化RRF融合器
        
        Args:
            k: RRF参数，通常设为60，控制排名的平滑程度
        """
        self.k = k
    
    def fuse_rankings(
        self, 
        rankings: List[List[Tuple[Document, float]]], 
        weights: Optional[List[float]] = None
    ) -> List[Tuple[Document, float]]:
        """
        融合多个排名列表
        
        Args:
            rankings: 多个检索结果列表，每个包含(Document, score)对
            weights: 各个检索器的权重，默认均等权重
        
        Returns:
            融合后的排名列表
        """
        if not rankings:
            return []
        
        # 设置默认权重
        if weights is None:
            weights = [1.0] * len(rankings)
        
        # 用于存储每个文档的融合分数
        doc_scores = defaultdict(float)
        doc_objects = {}  # 存储文档对象
        
        # 对每个排名列表进行RRF计算
        for ranking_idx, ranking in enumerate(rankings):
            weight = weights[ranking_idx]
            
            for rank, (doc, original_score) in enumerate(ranking):
                # 使用文档内容作为唯一标识
                doc_id = doc.page_content
                
                # RRF公式: score = weight * (1 / (k + rank + 1))
                rrf_score = weight / (self.k + rank + 1)
                doc_scores[doc_id] += rrf_score
                
                # 保存文档对象（如果还没有的话）
                if doc_id not in doc_objects:
                    doc_objects[doc_id] = doc
        
        # 按融合分数排序
        fused_results = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            doc = doc_objects[doc_id]
            # 将RRF分数添加到元数据
            doc.metadata['rrf_score'] = score
            fused_results.append((doc, score))
        
        return fused_results

class BM25Retriever:
    """
    简单的BM25检索器实现
    用于基于关键词的精确匹配检索
    """
    
    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75, min_match_ratio: float = 0.3, score_threshold: float = 0.01):
        """
        初始化BM25检索器
        
        Args:
            documents: 文档列表
            k1: BM25参数，控制词频饱和度
            b: BM25参数，控制文档长度的影响
            min_match_ratio: 最小匹配词比例（0.3表示至少匹配30%的查询词）
            score_threshold: 最小分数阈值，低于此分数的文档将被过滤
        """
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.min_match_ratio = min_match_ratio
        self.score_threshold = score_threshold
        self.doc_count = len(documents)
        
        # 预处理文档
        self._preprocess_documents()
    
    def _load_stop_words(self) -> set:
        """加载停用词库"""
        import os
        
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        stop_words_path = os.path.join(current_dir, 'stop_words.txt')
        
        stop_words = set()
        
        try:
            if os.path.exists(stop_words_path):
                with open(stop_words_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word and not word.startswith('#'):  # 跳过空行和注释行
                            stop_words.add(word)
   
            else:
                print(f"⚠️ 停用词文件不存在: {stop_words_path}")
                # 使用基本的停用词作为后备
                stop_words = {
                    '的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '一个', '一些',
                    '上', '也', '很', '到', '说', '要', '去', '会', '着', '看', '好',
                    '自己', '这', '那', '里', '还', '把', '来', '时', '个', '为', '但', '或',
                    '与', '及', '以', '所', '而', '等', '等等', '如', '如果', '因为', '所以'
                }
                print(f"⚠️ BM25检索器使用默认停用词库: {len(stop_words)} 个停用词")
        except Exception as e:
            print(f"❌ BM25检索器加载停用词库失败: {e}")
            # 使用基本的停用词作为后备
            stop_words = {
                '的', '了', '在', '是', '有', '和', '就', '不', '都', '一', '一个', '一些',
                '上', '也', '很', '到', '说', '要', '去', '会', '着', '看', '好'
            }
            print(f"⚠️ BM25检索器使用基本停用词库: {len(stop_words)} 个停用词")
        
        return stop_words
    
    def _smart_tokenize(self, text: str) -> List[str]:
        """通用智能分词 - 适用于各种中文文本，保护引号内容"""
        import jieba
        import re
        
        # 初始化停用词（只在第一次调用时）
        if not hasattr(self, '_stop_words'):
            self.stop_words = self._load_stop_words()
        
        # 第一步：提取并保护引号内的内容
        quoted_terms = []
        remaining_text = text
        
        # 匹配各种引号：中文双引号、英文双引号、中文单引号
        quote_patterns = [
            r'“([^”]*)”',
            r'"([^"]*)"',      # 英文双引号
            r'‘([^’]*)’',
            r"'([^']*)'",      # 英文单引号
            r'《([^》]*)》',    # 书名号 《》
        ]
        
        # 逐个处理每种引号模式
        for pattern in quote_patterns:
            matches = list(re.finditer(pattern, remaining_text))
            for match in matches:
                quoted_content = match.group(1).strip()
                if quoted_content:  # 非空引号内容
                    quoted_terms.append(quoted_content)
            # 移除已处理的引号内容（包括引号本身）
            remaining_text = re.sub(pattern, ' ', remaining_text)
        
        # 第二步：对剩余文本进行正常分词（使用精确模式，避免冗余分词）
        tokens = list(jieba.cut(remaining_text, cut_all=False))
        
        # 第三步：清理和过滤普通分词结果
        processed_tokens = []
        for token in tokens:
            token = token.strip()
            
            # 跳过空字符串和停用词
            if not token or token in self.stop_words:
                continue
            
            # 跳过纯数字（除非是特殊情况，如年份）
            if token.isdigit() and len(token) < 4:
                continue
                
            # 跳过纯英文字母的短词（除非是专有名词）
            if token.isalpha() and token.islower() and len(token) <= 2:
                continue
            
            # 保留有意义的词
            if len(token) >= 1:  # 保持宽松，不过度过滤
                processed_tokens.append(token)
        
        # 第四步：合并引号内容和普通分词结果
        final_tokens = quoted_terms + processed_tokens
        
        return final_tokens

    def _preprocess_documents(self):
        """预处理文档，构建倒排索引"""
        self.doc_tokens = []  # 每个文档的分词结果
        self.doc_lengths = []  # 每个文档的长度
        self.term_doc_freq = defaultdict(int)  # 词项的文档频率
        self.vocab = set()  # 词汇表
        
        # 分词和构建索引
        for doc in self.documents:
            # 使用智能分词
            tokens = self._smart_tokenize(doc.page_content)
            
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            
            # 统计词项文档频率
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.term_doc_freq[token] += 1
                self.vocab.add(token)
        
        # 计算平均文档长度
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0
        
        print(f"📚 BM25索引构建完成: {len(self.documents)}篇文档, {len(self.vocab)}个词汇")
        print(f"🔤 词汇示例: {list(self.vocab)[:10]}")
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        BM25检索 - 支持部分匹配
        
        Args:
            query: 查询文本
            k: 返回文档数量
        
        Returns:
            排序后的文档列表
        """
        import jieba
        # 查询分词 - 使用相同的智能分词
        
        # 查询分词
        #query_tokens = list(jieba.cut(query))
        #query_tokens = [token.strip() for token in query_tokens if token.strip()]

        query_tokens = self._smart_tokenize(query)
        
        if not query_tokens:
            return []
        
        print(f"🔍 BM25查询分词 (优化后): {query_tokens}")
        
        # 计算每个文档的BM25分数
        scores = []
        
        for doc_idx, doc in enumerate(self.documents):
            doc_tokens = self.doc_tokens[doc_idx]
            doc_length = self.doc_lengths[doc_idx]
            
            score = 0.0
            matched_terms = 0
            
            for query_token in query_tokens:
                if query_token not in self.vocab:
                    continue
                
                # 计算词频
                tf = doc_tokens.count(query_token)
                
                if tf > 0:
                    matched_terms += 1
                    
                    # 计算IDF
                    df = self.term_doc_freq[query_token]
                    idf = math.log((self.doc_count - df + 0.5) / (df + 0.5))
                    
                    # 计算BM25分数
                    tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length)))
                    score += idf * tf_component
            
            # 计算匹配比例
            match_ratio = matched_terms / len(query_tokens) if query_tokens else 0
            
            # 只有满足最小匹配比例且分数超过阈值的文档才被保留
            if match_ratio >= self.min_match_ratio and score >= self.score_threshold:
                # 对分数进行匹配比例加权，鼓励匹配更多词的文档
                adjusted_score = score * (0.5 + 0.5 * match_ratio)
                scores.append((doc, adjusted_score))
        
        print(f"📊 BM25检索：{len(scores)}个文档满足匹配条件")
        
        # 按分数排序并返回前k个
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:k]

class HybridRetriever:
    """
    混合检索器：结合向量检索和BM25检索，使用RRF融合
    """
    
    def __init__(self, vector_store_manager, rrf_k: int = 60, bm25_min_match_ratio: float = 0.3, bm25_score_threshold: float = 0.01):
        """
        初始化混合检索器
        
        Args:
            vector_store_manager: 向量存储管理器
            rrf_k: RRF融合参数
            bm25_min_match_ratio: BM25最小匹配词比例
            bm25_score_threshold: BM25最小分数阈值
        """
        self.vector_store = vector_store_manager
        self.rrf_fusion = RRFFusion(k=rrf_k)
        self.bm25_retriever = None
        self.bm25_min_match_ratio = bm25_min_match_ratio
        self.bm25_score_threshold = bm25_score_threshold
        
        # 初始化BM25检索器
        self._init_bm25()
    
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
                        min_match_ratio=self.bm25_min_match_ratio,
                        score_threshold=self.bm25_score_threshold
                    )
                    print(f"✅ BM25检索器初始化成功，共{len(documents)}个文档")
                    print(f"📊 BM25配置：最小匹配比例={self.bm25_min_match_ratio:.1%}，分数阈值={self.bm25_score_threshold}")
                else:
                    print("⚠️ 向量存储为空，无法初始化BM25检索器")
            else:
                print("⚠️ 向量存储未初始化，无法创建BM25检索器")
                
        except Exception as e:
            print(f"❌ BM25检索器初始化失败: {e}")
            self.bm25_retriever = None
    
    def hybrid_search(
        self, 
        query: str, 
        k: int = 10,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        vector_k: int = 15,
        bm25_k: int = 15
    ) -> List[Tuple[Document, float]]:
        """
        混合检索：向量检索 + BM25检索 + RRF融合
        
        Args:
            query: 查询文本
            k: 最终返回的文档数量
            vector_weight: 向量检索权重
            bm25_weight: BM25检索权重
            vector_k: 向量检索返回数量
            bm25_k: BM25检索返回数量
        
        Returns:
            融合后的检索结果
        """
        rankings = []
        weights = []
        
        # 1. 向量检索
        try:
            print(f"🔍 执行向量检索 (k={vector_k})...")
            vector_results = self.vector_store.search_similarity(query, k=vector_k)
            if vector_results:
                # 转换为(Document, score)格式，这里用相似度作为分数
                vector_ranking = [(doc, 1.0) for doc in vector_results]  # chromadb不直接返回分数
                rankings.append(vector_ranking)
                weights.append(vector_weight)
                print(f"📊 向量检索找到 {len(vector_results)} 个文档")
            else:
                print("⚠️ 向量检索无结果")
        except Exception as e:
            print(f"❌ 向量检索失败: {e}")
        
        # 2. BM25检索
        if self.bm25_retriever:
            try:
                print(f"🔍 执行BM25检索 (k={bm25_k})...")
                bm25_results = self.bm25_retriever.search(query, k=bm25_k)
                if bm25_results:
                    # 过滤掉分数为0的结果
                    bm25_ranking = [(doc, score) for doc, score in bm25_results if score > 0]
                    if bm25_ranking:
                        rankings.append(bm25_ranking)
                        weights.append(bm25_weight)
                        print(f"📊 BM25检索找到 {len(bm25_ranking)} 个相关文档")
                    else:
                        print("⚠️ BM25检索无相关结果")
                else:
                    print("⚠️ BM25检索无结果")
            except Exception as e:
                print(f"❌ BM25检索失败: {e}")
        else:
            print("⚠️ BM25检索器未初始化，跳过BM25检索")
        
        # 3. RRF融合
        if not rankings:
            print("❌ 所有检索方法都失败了")
            return []
        
        if len(rankings) == 1:
            print("⚠️ 只有一种检索方法有结果，无需融合")
            final_results = rankings[0][:k]
        else:
            print(f"🔄 使用RRF融合 {len(rankings)} 个检索结果...")
            fused_results = self.rrf_fusion.fuse_rankings(rankings, weights)
            final_results = fused_results[:k]
            print(f"✅ RRF融合完成，返回 {len(final_results)} 个文档")
        
        return final_results 