#!/usr/bin/env python3
"""
改进版 Chunk Size 优化测试脚本
使用人工设计的问题-答案对数据集进行评估
"""

import os
import time
import json
import numpy as np
import shutil
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics.pairwise import cosine_similarity

# 导入现有RAG组件
from embedding.adapter import EmbeddingAdapter
from llm.adapter import LLMAdapter
from vector_store.vector_store import VectorStoreManager
from langchain_core.documents import Document
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

@dataclass
class ImprovedQueryResult:
    """改进的查询结果数据类"""
    query: str
    generated_answer: str           # RAG生成的答案
    reference_answer: str           # 标准答案
    retrieved_docs: List[Document]
    semantic_similarities: List[float]  # 查询与文档的语义相似度
    answer_similarity: float        # 生成答案与标准答案的相似度
    response_time: float

@dataclass
class ImprovedChunkSizeResult:
    """改进的chunk size测试结果"""
    chunk_size: int
    
    # 核心指标
    avg_answer_similarity: float      # 平均答案相似度（与标准答案）
    avg_semantic_similarity: float    # 平均语义相似度（查询与文档）
    avg_response_time: float         # 平均响应时间
    
    # 检索质量指标
    avg_retrieval_relevance: float   # 平均检索相关性
    
    query_results: List[ImprovedQueryResult]

class ImprovedChunkSizeEvaluator:
    """改进的Chunk Size评估器"""
    
    def __init__(self, document_path: str = "./data/1.txt", dataset_path: str = "./evaluation_dataset.json"):
        """初始化评估器"""
        self.document_path = document_path
        self.dataset_path = dataset_path
        self.embedding = EmbeddingAdapter.get_embedding("dashscope", "text-embedding-v3")
        self.llm = LLMAdapter.get_llm("dashscope", "qwen-turbo", temperature=0.1)
        
        # 加载文档内容
        self._load_document()
        
        # 加载评估数据集
        self._load_evaluation_dataset()
        
        print(f"✅ 改进版评估器初始化完成")
        print(f"📄 文档: {self.document_path}")
        print(f"📋 评估数据集: {self.dataset_path}")
        print(f"📝 测试问题数: {len(self.test_qa_pairs)}")
    
    def _load_document(self):
        """加载文档内容"""
        if not os.path.exists(self.document_path):
            raise FileNotFoundError(f"文档文件不存在: {self.document_path}")
        
        with open(self.document_path, 'r', encoding='utf-8') as f:
            self.document_content = f.read()
        
        print(f"📚 文档加载成功，长度: {len(self.document_content)} 字符")
    
    def _load_evaluation_dataset(self):
        """加载评估数据集"""
        if os.path.exists(self.dataset_path):
            # 从文件加载
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.test_qa_pairs = json.load(f)
            print(f"📋 从 {self.dataset_path} 加载了 {len(self.test_qa_pairs)} 个问答对")
        else:
            # 创建默认示例数据集
            self._create_default_dataset()
            print(f"📋 创建了默认评估数据集，包含 {len(self.test_qa_pairs)} 个问答对")
            print(f"💡 您可以修改 {self.dataset_path} 文件来自定义评估数据集")
    
    def _create_default_dataset(self):
        """创建默认的评估数据集"""
        self.test_qa_pairs = [
            {
                "question": "一微半导体是什么公司？",
                "reference_answer": "一微半导体是一家专业从事集成电路设计、研发和销售的高新技术企业，主要业务涵盖芯片设计和半导体相关产品的开发。"
            },
            {
                "question": "员工迟到会有什么处罚？",
                "reference_answer": "员工迟到将根据考勤制度进行相应处罚，通常包括口头警告、书面警告，严重者可能扣除相应工资或奖金。"
            },
            {
                "question": "公司的核心价值观是什么？",
                "reference_answer": "公司秉承创新、诚信、团队合作和客户至上的核心价值观，致力于为客户提供优质的产品和服务。"
            },
            {
                "question": "公司有多少员工？",
                "reference_answer": "公司目前拥有数百名员工，涵盖研发、销售、市场、行政等各个部门。"
            },
            {
                "question": "公司的考勤时间是怎样的？",
                "reference_answer": "公司实行标准工作时间制，通常为周一至周五上午9:00-12:00，下午13:30-18:00，具体时间安排可能根据部门有所调整。"
            }
        ]
        
        # 保存到文件
        with open(self.dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_qa_pairs, f, ensure_ascii=False, indent=2)
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本之间的语义相似度"""
        try:
            # 获取两个文本的嵌入向量
            embedding1 = self.embedding.embed_query(text1)
            embedding2 = self.embedding.embed_query(text2)
            
            # 计算余弦相似度
            similarity = cosine_similarity(
                np.array(embedding1).reshape(1, -1),
                np.array(embedding2).reshape(1, -1)
            )[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"⚠️ 计算语义相似度失败: {e}")
            return 0.0
    
    def _evaluate_retrieval_relevance(self, query: str, retrieved_docs: List[Document]) -> float:
        """评估检索文档的相关性"""
        if not retrieved_docs:
            return 0.0
        
        relevance_scores = []
        for doc in retrieved_docs:
            # 计算查询与文档的语义相似度
            similarity = self._calculate_semantic_similarity(query, doc.page_content)
            relevance_scores.append(similarity)
        
        return np.mean(relevance_scores)
    
    def _evaluate_query_improved(self, vector_store: VectorStoreManager, qa_pair: Dict, k: int = 5) -> ImprovedQueryResult:
        """改进的查询评估"""
        question = qa_pair["question"]
        reference_answer = qa_pair["reference_answer"]
        
        # 执行检索
        start_time = time.time()
        retriever = vector_store._create_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        retrieved_docs = retriever.invoke(question)
        
        # 生成答案
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer_prompt = f"""基于以下上下文信息，准确回答用户问题。

上下文：
{context}

问题：{question}

请给出准确、完整的答案："""
        
        answer_response = self.llm.invoke(answer_prompt)
        generated_answer = answer_response.content if hasattr(answer_response, 'content') else str(answer_response)
        
        response_time = time.time() - start_time
        
        # 计算查询与检索文档的语义相似度
        semantic_similarities = []
        for doc in retrieved_docs:
            similarity = self._calculate_semantic_similarity(question, doc.page_content)
            semantic_similarities.append(similarity)
        
        # 计算生成答案与标准答案的相似度
        answer_similarity = self._calculate_semantic_similarity(generated_answer, reference_answer)
        
        return ImprovedQueryResult(
            query=question,
            generated_answer=generated_answer,
            reference_answer=reference_answer,
            retrieved_docs=retrieved_docs,
            semantic_similarities=semantic_similarities,
            answer_similarity=answer_similarity,
            response_time=response_time
        )
    
    def test_chunk_size_improved(self, chunk_size: int, k: int = 5) -> ImprovedChunkSizeResult:
        """改进的chunk size测试"""
        print(f"\n{'='*60}")
        print(f"🧪 测试 Chunk Size: {chunk_size}")
        print(f"{'='*60}")
        
        # 创建向量存储
        collection_name = f"improved_chunk_test_{chunk_size}"
        persist_dir = f"./improved_chunk_test_db_{chunk_size}"
        
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
        
        vector_store = VectorStoreManager(
            embedding_model=self.embedding,
            collection_name=collection_name,
            persist_directory=persist_dir
        )
        
        document = Document(
            page_content=self.document_content,
            metadata={'source': self.document_path}
        )
        
        vector_store.create_from_documents(
            [document],
            chunk_size=chunk_size,
            chunk_overlap=100
        )
        
        print(f"📦 创建了 {len(vector_store)} 个文档块")
        
        # 执行所有查询
        query_results = []
        
        for i, qa_pair in enumerate(self.test_qa_pairs, 1):
            print(f"📋 问题 {i}/{len(self.test_qa_pairs)}: {qa_pair['question']}")
            
            result = self._evaluate_query_improved(vector_store, qa_pair, k)
            query_results.append(result)
            
            print(f"   ⏱️ 响应时间: {result.response_time:.3f}s")
            print(f"   📊 平均检索相似度: {np.mean(result.semantic_similarities):.3f}")
            print(f"   🎯 答案相似度: {result.answer_similarity:.3f}")
            print(f"   💬 生成答案: {result.generated_answer[:50]}...")
        
        # 计算聚合指标
        avg_answer_similarity = np.mean([r.answer_similarity for r in query_results])
        avg_semantic_similarity = np.mean([np.mean(r.semantic_similarities) for r in query_results])
        avg_response_time = np.mean([r.response_time for r in query_results])
        
        # 计算检索质量指标
        retrieval_relevance_scores = []
        
        for result in query_results:
            # 检索相关性
            relevance = self._evaluate_retrieval_relevance(result.query, result.retrieved_docs)
            retrieval_relevance_scores.append(relevance)
        
        avg_retrieval_relevance = np.mean(retrieval_relevance_scores)
        
        # 清理测试数据
        shutil.rmtree(persist_dir, ignore_errors=True)
        
        print(f"\n📊 Chunk Size {chunk_size} 测试结果:")
        print(f"   🎯 平均答案相似度: {avg_answer_similarity:.3f}")
        print(f"   🔍 平均检索相似度: {avg_semantic_similarity:.3f}")
        print(f"   📋 平均检索相关性: {avg_retrieval_relevance:.3f}")
        print(f"   ⏱️ 平均响应时间: {avg_response_time:.3f}s")
        
        return ImprovedChunkSizeResult(
            chunk_size=chunk_size,
            avg_answer_similarity=avg_answer_similarity,
            avg_semantic_similarity=avg_semantic_similarity,
            avg_response_time=avg_response_time,
            avg_retrieval_relevance=avg_retrieval_relevance,
            query_results=query_results
        )
    
    def comprehensive_evaluation(self, chunk_sizes: List[int] = None, k: int = 5) -> Dict[str, Any]:
        """综合评估多个chunk size"""
        if chunk_sizes is None:
            chunk_sizes = [256, 512, 800, 1024, 1500, 2048]
        
        print(f"\n" + "="*80)
        print(f"🧪 Chunk Size 综合评估 - 基于人工标注数据集")
        print(f"="*80)
        print(f"📋 测试配置:")
        print(f"   📄 文档: {self.document_path}")
        print(f"   📝 评估数据集: {self.dataset_path}")
        print(f"   🔍 检索数量 (k): {k}")
        print(f"   📏 测试问题数: {len(self.test_qa_pairs)}")
        print(f"   🎛️ Chunk Sizes: {chunk_sizes}")
        
        results = []
        
        # 测试每个chunk size
        for chunk_size in chunk_sizes:
            try:
                result = self.test_chunk_size_improved(chunk_size, k)
                results.append(result)
            except Exception as e:
                print(f"❌ Chunk Size {chunk_size} 测试失败: {e}")
        
        # 分析结果
        analysis = self._analyze_results(results)
        
        # 保存结果
        self._save_results(results, analysis)
        
        return {
            "results": results,
            "analysis": analysis
        }
    
    def _analyze_results(self, results: List[ImprovedChunkSizeResult]) -> Dict[str, Any]:
        """分析测试结果"""
        if not results:
            return {"error": "没有有效的测试结果"}
        
        print(f"\n" + "="*80)
        print(f"📊 结果分析")
        print(f"="*80)
        
        # 按不同指标排序
        by_answer_similarity = sorted(results, key=lambda x: x.avg_answer_similarity, reverse=True)
        by_retrieval_relevance = sorted(results, key=lambda x: x.avg_retrieval_relevance, reverse=True)
        by_speed = sorted(results, key=lambda x: x.avg_response_time)
        
        # 计算综合评分
        def calculate_composite_score(result: ImprovedChunkSizeResult) -> float:
            # 归一化各指标
            max_answer_sim = max(r.avg_answer_similarity for r in results)
            max_retrieval_rel = max(r.avg_retrieval_relevance for r in results)
            min_time = min(r.avg_response_time for r in results)
            
            norm_answer_sim = result.avg_answer_similarity / max_answer_sim if max_answer_sim > 0 else 0
            norm_retrieval_rel = result.avg_retrieval_relevance / max_retrieval_rel if max_retrieval_rel > 0 else 0
            norm_speed = min_time / result.avg_response_time if result.avg_response_time > 0 else 0
            
            # 加权计算综合评分
            composite = (norm_answer_sim * 0.5 +      # 答案相似度权重 50%
                        norm_retrieval_rel * 0.25 +   # 检索相关性权重 25%
                        norm_speed * 0.25)             # 速度权重 25%
            
            return composite
        
        # 计算综合评分
        for result in results:
            result.composite_score = calculate_composite_score(result)
        
        by_composite = sorted(results, key=lambda x: x.composite_score, reverse=True)
        
        # 打印详细对比表
        print(f"\n📋 详细对比表:")
        print(f"{'Chunk Size':<11} {'答案相似度':<10} {'检索相关性':<10} {'响应时间':<10} {'综合评分':<8}")
        print("-" * 75)
        
        for result in results:
            print(f"{result.chunk_size:<11} "
                  f"{result.avg_answer_similarity:<10.3f} "
                  f"{result.avg_retrieval_relevance:<10.3f} "
                  f"{result.avg_response_time:<10.3f} "
                  f"{result.composite_score:<8.3f}")
        
        # 最佳配置分析
        print(f"\n🏆 最佳配置分析:")
        print(f"   🎯 最高答案相似度: Chunk Size {by_answer_similarity[0].chunk_size} (相似度: {by_answer_similarity[0].avg_answer_similarity:.3f})")
        print(f"   🔍 最高检索相关性: Chunk Size {by_retrieval_relevance[0].chunk_size} (相关性: {by_retrieval_relevance[0].avg_retrieval_relevance:.3f})")
        print(f"   🚀 最快响应: Chunk Size {by_speed[0].chunk_size} (时间: {by_speed[0].avg_response_time:.3f}s)")
        print(f"   🎖️ 最佳综合: Chunk Size {by_composite[0].chunk_size} (综合评分: {by_composite[0].composite_score:.3f})")
        
        # 推荐配置
        recommended = by_composite[0]
        
        print(f"\n💡 推荐配置:")
        print(f"   📏 建议使用 Chunk Size: {recommended.chunk_size}")
        print(f"   📊 性能指标:")
        print(f"      🎯 答案相似度: {recommended.avg_answer_similarity:.3f}")
        print(f"      🔍 检索相关性: {recommended.avg_retrieval_relevance:.3f}")
        print(f"      ⏱️ 响应时间: {recommended.avg_response_time:.3f}s")
        print(f"      🎖️ 综合评分: {recommended.composite_score:.3f}")
        
        print(f"\n🔧 使用建议:")
        print(f"   - 在RAG系统中设置 chunk_size={recommended.chunk_size}")
        print(f"   - 该配置在答案质量和检索效果之间取得了最佳平衡")
        
        if recommended.chunk_size != by_answer_similarity[0].chunk_size:
            print(f"   - 如果更重视答案质量，可考虑 chunk_size={by_answer_similarity[0].chunk_size}")
        if recommended.chunk_size != by_speed[0].chunk_size:
            print(f"   - 如果更重视响应速度，可考虑 chunk_size={by_speed[0].chunk_size}")
        
        return {
            "best_overall": {
                "chunk_size": recommended.chunk_size,
                "metrics": {
                    "answer_similarity": recommended.avg_answer_similarity,
                    "retrieval_relevance": recommended.avg_retrieval_relevance,
                    "response_time": recommended.avg_response_time,
                    "composite_score": recommended.composite_score
                }
            },
            "best_by_metric": {
                "answer_similarity": {"chunk_size": by_answer_similarity[0].chunk_size, "value": by_answer_similarity[0].avg_answer_similarity},
                "retrieval_relevance": {"chunk_size": by_retrieval_relevance[0].chunk_size, "value": by_retrieval_relevance[0].avg_retrieval_relevance},
                "speed": {"chunk_size": by_speed[0].chunk_size, "value": by_speed[0].avg_response_time}
            },
            "all_results": [
                {
                    "chunk_size": r.chunk_size,
                    "answer_similarity": r.avg_answer_similarity,
                    "retrieval_relevance": r.avg_retrieval_relevance,
                    "response_time": r.avg_response_time,
                    "composite_score": r.composite_score
                }
                for r in results
            ]
        }
    
    def _save_results(self, results: List[ImprovedChunkSizeResult], analysis: Dict[str, Any]):
        """保存结果到文件"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"chunk_size_evaluation_{timestamp}.json"
        
        # 准备保存的数据
        save_data = {
            "timestamp": timestamp,
            "document": self.document_path,
            "dataset": self.dataset_path,
            "test_questions_count": len(self.test_qa_pairs),
            "results": [
                {
                    "chunk_size": r.chunk_size,
                    "answer_similarity": r.avg_answer_similarity,
                    "retrieval_relevance": r.avg_retrieval_relevance,
                    "response_time": r.avg_response_time,
                    "composite_score": getattr(r, 'composite_score', 0),
                    "detailed_results": [
                        {
                            "question": qr.query,
                            "generated_answer": qr.generated_answer,
                            "reference_answer": qr.reference_answer,
                            "answer_similarity": qr.answer_similarity,
                            "response_time": qr.response_time
                        }
                        for qr in r.query_results
                    ]
                }
                for r in results
            ],
            "analysis": analysis
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 详细结果已保存到: {filename}")

def main():
    """主函数"""
    try:
        document_path = "./data/1.txt"
        dataset_path = "./evaluation_dataset.json"
        
        if not os.path.exists(document_path):
            print(f"❌ 数据文件不存在: {document_path}")
            return
        
        evaluator = ImprovedChunkSizeEvaluator(document_path, dataset_path)
        
        # 执行综合评估
        chunk_sizes = [256, 512, 800, 1024, 1500, 2048]
        evaluation_results = evaluator.comprehensive_evaluation(chunk_sizes, k=5)
        
        print(f"\n🎉 评估完成！")
        print(f"建议的最佳 Chunk Size: {evaluation_results['analysis']['best_overall']['chunk_size']}")
        
    except KeyboardInterrupt:
        print(f"\n👋 用户中断，评估停止")
    except Exception as e:
        print(f"❌ 评估出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 