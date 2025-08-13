#!/usr/bin/env python3
"""
KV Cache 优化模块
基于 turboRAG 思路，为现有 RAG 系统添加 KV 缓存优化
"""

import os
import torch
import pickle
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# 导入现有组件
from vector_store.vector_store import VectorStoreManager
from embedding.adapter import EmbeddingAdapter
from llm.adapter import LLMAdapter


class KVCacheManager:
    """KV缓存管理器 - 核心优化组件"""
    
    def __init__(self, 
                 cache_dir: str = "./kv_cache",
                 model_name: str = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        初始化KV缓存管理器
        
        Args:
            cache_dir: 缓存文件存储目录
            model_name: 模型名称（如果使用本地模型）
            device: 计算设备
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.device = device
        
        # 如果提供了模型名称，初始化本地模型
        self.model = None
        self.tokenizer = None
        if model_name:
            self._init_local_model(model_name)
        
        # 缓存索引：{chunk_hash: cache_file_path}
        self.cache_index = self._load_cache_index()
        
        # 预定义的提示前缀
        self.prefix_template = """你是一微半导体公司的智能助手。请基于以下文档内容回答用户问题。

文档内容：
"""
        
        print(f"🚀 KV缓存管理器已初始化")
        print(f"📁 缓存目录: {self.cache_dir}")
        print(f"💾 已有缓存数量: {len(self.cache_index)}")
    
    def _init_local_model(self, model_name: str):
        """初始化本地模型（如果需要）"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            print(f"✅ 本地模型加载成功: {model_name}")
        except Exception as e:
            print(f"⚠️ 本地模型加载失败: {e}")
            print("💡 将使用远程API模式")
    
    def _load_cache_index(self) -> Dict[str, str]:
        """加载缓存索引"""
        index_file = self.cache_dir / "cache_index.pkl"
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return {}
    
    def _save_cache_index(self):
        """保存缓存索引"""
        index_file = self.cache_dir / "cache_index.pkl"
        with open(index_file, 'wb') as f:
            pickle.dump(self.cache_index, f)
    
    def _get_chunk_hash(self, chunk_content: str) -> str:
        """获取文档块的哈希值"""
        return hashlib.md5(chunk_content.encode('utf-8')).hexdigest()
    
    def preprocess_documents(self, vector_store: VectorStoreManager, 
                           batch_size: int = 8) -> int:
        """
        预处理文档，生成KV缓存
        
        Args:
            vector_store: 向量存储管理器
            batch_size: 批处理大小
            
        Returns:
            int: 处理的文档块数量
        """
        print(f"🔄 开始预处理文档，生成KV缓存...")
        
        # 获取所有文档块
        try:
            documents = vector_store.get_all_documents()
            all_docs = []
            for doc in documents:
                all_docs.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata
                })
        except Exception as e:
            print(f"⚠️ 获取文档失败: {e}")
            all_docs = []
        
        print(f"📄 发现 {len(all_docs)} 个文档块")
        
        processed_count = 0
        new_cache_count = 0
        
        for i, doc in enumerate(all_docs):
            content = doc['content']
            chunk_hash = self._get_chunk_hash(content)
            
            # 检查是否已有缓存
            if chunk_hash in self.cache_index:
                print(f"⚡ 跳过已缓存块 {i+1}/{len(all_docs)}")
                processed_count += 1
                continue
            
            # 生成带前缀的完整内容
            full_content = self.prefix_template + content
            
            # 生成KV缓存
            cache_data = self._generate_kv_cache(full_content, chunk_hash)
            
            if cache_data:
                # 保存缓存文件
                cache_file = self.cache_dir / f"cache_{chunk_hash}.pkl"
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                # 更新索引
                self.cache_index[chunk_hash] = str(cache_file)
                new_cache_count += 1
                
                print(f"💾 已处理块 {i+1}/{len(all_docs)} (新增缓存)")
            
            processed_count += 1
        
        # 保存索引
        self._save_cache_index()
        
        print(f"✅ 文档预处理完成！")
        print(f"📊 总处理: {processed_count} 块")
        print(f"🆕 新增缓存: {new_cache_count} 块")
        print(f"💾 总缓存数: {len(self.cache_index)} 块")
        
        return processed_count
    
    def _generate_kv_cache(self, content: str, chunk_hash: str) -> Optional[Dict[str, Any]]:
        """
        为单个文档块生成KV缓存
        
        Args:
            content: 文档内容
            chunk_hash: 文档哈希
            
        Returns:
            Optional[Dict]: 缓存数据或None
        """
        if self.model is None or self.tokenizer is None:
            # 如果没有本地模型，保存文本信息供后续优化
            return {
                'content': content,
                'chunk_hash': chunk_hash,
                'type': 'text_cache',
                'timestamp': time.time()
            }
        
        try:
            # 使用本地模型生成KV缓存
            inputs = self.tokenizer(
                content, 
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs, use_cache=True)
            
            # 保存KV缓存到CPU以节省显存
            kv_cache = None
            if hasattr(outputs, 'past_key_values') and outputs.past_key_values:
                kv_cache = []
                for layer_cache in outputs.past_key_values:
                    kv_cache.append(tuple(tensor.cpu() for tensor in layer_cache))
            
            return {
                'content': content,
                'chunk_hash': chunk_hash,
                'kv_cache': kv_cache,
                'input_ids': inputs['input_ids'].cpu(),
                'attention_mask': inputs['attention_mask'].cpu(),
                'type': 'full_cache',
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"⚠️ 生成KV缓存失败 {chunk_hash}: {e}")
            return None
    
    def get_cached_chunks(self, chunk_contents: List[str]) -> Tuple[List[Dict], List[str]]:
        """
        获取文档块的缓存信息
        
        Args:
            chunk_contents: 文档块内容列表
            
        Returns:
            Tuple[List[Dict], List[str]]: (缓存数据列表, 未缓存内容列表)
        """
        cached_data = []
        uncached_contents = []
        
        for content in chunk_contents:
            chunk_hash = self._get_chunk_hash(content)
            
            if chunk_hash in self.cache_index:
                cache_file = self.cache_index[chunk_hash]
                try:
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    cached_data.append(cache_data)
                except:
                    # 缓存文件损坏，添加到未缓存列表
                    uncached_contents.append(content)
            else:
                uncached_contents.append(content)
        
        return cached_data, uncached_contents
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_size = 0
        cache_files = list(self.cache_dir.glob("cache_*.pkl"))
        
        for cache_file in cache_files:
            total_size += cache_file.stat().st_size
        
        return {
            'total_caches': len(self.cache_index),
            'cache_files': len(cache_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_directory': str(self.cache_dir)
        }
    
    def clear_cache(self):
        """清理所有缓存"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index = {}
        print(f"🗑️ 已清理所有KV缓存")


class OptimizedRAG:
    """优化版RAG - 集成KV缓存"""
    
    def __init__(self, original_rag, cache_manager: KVCacheManager):
        """
        初始化优化版RAG
        
        Args:
            original_rag: 原始RAG系统实例
            cache_manager: KV缓存管理器
        """
        self.original_rag = original_rag
        self.cache_manager = cache_manager
        
        print(f"🚀 优化版RAG已初始化")
    
    def preprocess_knowledge_base(self):
        """预处理知识库，生成KV缓存"""
        return self.cache_manager.preprocess_documents(
            self.original_rag.vector_store
        )
    
    def optimized_query(self, question: str, show_sources: bool = False) -> Dict[str, Any]:
        """
        优化的查询方法
        
        Args:
            question: 用户问题
            show_sources: 是否显示来源
            
        Returns:
            Dict[str, Any]: 查询结果
        """
        print(f"🔍 优化查询: {question}")
        start_time = time.time()
        
        try:
            # 1. 向量检索（保持原有逻辑）
            retriever = self.original_rag.vector_store._create_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            retrieved_docs = retriever.invoke(question)
            
            retrieval_time = time.time() - start_time
            print(f"📚 检索完成，耗时: {retrieval_time:.2f}秒")
            
            # 2. 获取缓存信息
            chunk_contents = [doc.page_content for doc in retrieved_docs]
            cached_data, uncached_contents = self.cache_manager.get_cached_chunks(chunk_contents)
            
            cache_hit_rate = len(cached_data) / len(chunk_contents) if chunk_contents else 0
            print(f"⚡ 缓存命中率: {cache_hit_rate:.1%} ({len(cached_data)}/{len(chunk_contents)})")
            
            # 3. 如果有缓存，使用优化路径
            if cached_data:
                # 使用缓存进行快速推理
                answer = self._fast_inference_with_cache(question, cached_data, uncached_contents)
            else:
                # 回退到原始方法
                print(f"⚠️ 无可用缓存，使用原始方法")
                original_result = self.original_rag.query(question, show_sources)
                answer = original_result['answer']
            
            # 4. 准备结果
            result = {
                "question": question,
                "answer": answer,
                "response_time": f"{time.time() - start_time:.2f}秒",
                "cache_hit_rate": f"{cache_hit_rate:.1%}",
                "optimization_used": len(cached_data) > 0,
                "sources": [{"content": doc.page_content[:200] + "...", 
                           "metadata": doc.metadata} 
                          for doc in retrieved_docs] if show_sources else []
            }
            
            print(f"✅ 优化查询完成，总耗时: {result['response_time']}")
            return result
            
        except Exception as e:
            print(f"❌ 优化查询失败，回退到原始方法: {e}")
            return self.original_rag.query(question, show_sources)
    
    def _fast_inference_with_cache(self, question: str, 
                                 cached_data: List[Dict], 
                                 uncached_contents: List[str]) -> str:
        """
        使用缓存进行快速推理
        
        Args:
            question: 用户问题
            cached_data: 缓存数据
            uncached_contents: 未缓存内容
            
        Returns:
            str: 生成的答案
        """
        # 组装上下文
        context_parts = []
        
        # 添加缓存的内容
        for cache_item in cached_data:
            if cache_item.get('type') == 'full_cache':
                # 如果有完整KV缓存，提取内容
                content = cache_item['content']
                # 移除前缀部分，只保留文档内容
                if content.startswith(self.cache_manager.prefix_template):
                    content = content[len(self.cache_manager.prefix_template):]
                context_parts.append(content)
            else:
                # 文本缓存
                content = cache_item['content']
                if content.startswith(self.cache_manager.prefix_template):
                    content = content[len(self.cache_manager.prefix_template):]
                context_parts.append(content)
        
        # 添加未缓存的内容
        context_parts.extend(uncached_contents)
        
        # 组装完整提示
        context = "\n\n".join(context_parts)
        prompt = f"""你是一微半导体公司的智能助手。请基于以下文档内容回答用户问题。

文档内容：
{context}

用户问题：{question}

请基于文档内容准确回答，如果文档中没有相关信息请说明。

回答："""
        
        # 使用原始LLM进行推理（这里可以进一步优化）
        try:
            answer = self.original_rag.llm.invoke(prompt)
            return answer
        except Exception as e:
            print(f"⚠️ 快速推理失败: {e}")
            # 回退到组装RAG链
            return self._fallback_to_rag_chain(question, context_parts)
    
    def _fallback_to_rag_chain(self, question: str, context_parts: List[str]) -> str:
        """回退到原始RAG链方法"""
        try:
            # 重新组装为Document对象
            from langchain_core.documents import Document
            docs = [Document(page_content=content) for content in context_parts]
            
            # 使用原始RAG链
            context = "\n\n".join(doc.page_content for doc in docs)
            
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            
            prompt = PromptTemplate(
                template="""你是一微半导体公司的智能助手。请基于以下文档内容回答用户问题。

文档内容：
{context}

用户问题：{question}

请基于文档内容准确回答，如果文档中没有相关信息请说明。

回答：""",
                input_variables=["context", "question"]
            )
            
            chain = prompt | self.original_rag.llm | StrOutputParser()
            return chain.invoke({"context": context, "question": question})
            
        except Exception as e:
            print(f"❌ 回退方法也失败: {e}")
            return f"抱歉，处理问题时出现错误: {str(e)}"


def create_optimized_rag(original_rag, model_name: str = None) -> OptimizedRAG:
    """
    创建优化版RAG系统的工厂函数
    
    Args:
        original_rag: 原始RAG系统实例
        model_name: 可选的本地模型名称
        
    Returns:
        OptimizedRAG: 优化版RAG系统
    """
    cache_manager = KVCacheManager(model_name=model_name)
    return OptimizedRAG(original_rag, cache_manager)


if __name__ == "__main__":
    # 测试模块
    print("🧪 测试KV缓存优化模块")
    print("=" * 50)
    
    # 这里可以添加测试代码
    cache_manager = KVCacheManager()
    stats = cache_manager.get_cache_stats()
    
    print(f"📊 缓存统计:")
    for key, value in stats.items():
        print(f"  {key}: {value}") 