#!/usr/bin/env python3
"""
基于1.txt的简单RAG系统
使用现有组件构建的单文件解决方案
"""

import os
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
import asyncio
import threading
from typing import Dict, Any, List
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 导入现有组件
from document_loader.local_document_processor import LocalDocumentProcessor
from embedding.adapter import EmbeddingAdapter
# from embedding.reranker import AdaptiveReranker  # 已禁用重排序
from llm.adapter import LLMAdapter
from vector_store.vector_store import VectorStoreManager
from vector_store.bm25_vec import HybridRetriever
from vector_store.incremental_document_processor import IncrementalDocumentProcessor

# 加载环境变量
from dotenv import load_dotenv
# 获取项目根目录的 .env 文件路径

load_dotenv()


class SimpleRAG:
    """简单RAG系统"""
    
    def __init__(self, document_path: str = "1.txt"):
        """初始化RAG系统"""
        # 如果是相对路径，尝试在脚本所在目录查找
        if not os.path.isabs(document_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(script_dir, document_path)
            if os.path.exists(full_path):
                document_path = full_path
        
        self.document_path = document_path
        self._auto_update_thread = None
        self._stop_auto_update = threading.Event()
        
        print(f"🚀 初始化混合检索RAG系统...")
        print(f"📁 文档路径: {self.document_path}")
        print(f"🔍 检索模式: BGE向量检索 + BM25关键词检索 + RRF融合")
        print(f"⏰ 自动更新: 每5分钟检查一次")
        
        # 检查文档是否存在
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"文档文件不存在: {document_path}")
        
        # 初始化组件
        self._setup_components()
        self._build_knowledge_base()
        self._setup_rag_chain()
        
        # 启动时检查一次更新
        if self.incremental_processor:
            print(f"🔍 启动时检查文档更新...")
            try:
                result = self.update_documents()
                if result["status"] == "success" and result.get("total_processed", 0) > 0:
                    print(f"✅ 启动时发现并处理了 {result['total_processed']} 个文档更新")
                elif result["status"] == "no_changes":
                    print(f"ℹ️ 启动时未发现文档变更")
            except Exception as e:
                print(f"⚠️ 启动时更新检查失败: {e}")
        
        # 启动自动更新后台线程
        if self.incremental_processor:
            self._start_auto_update_thread()
        
        print(f"✅ RAG系统就绪！")
    
    def _setup_components(self):
        """设置组件"""
        # 嵌入模型 - 使用开源免费的 bge-large-zh-v1.5
        self.embedding = EmbeddingAdapter.get_embedding("bge", "BAAI/bge-large-zh-v1.5")
        
        self.llm = LLMAdapter.get_llm("openai", "Qwen/Qwen2.5-7B-Instruct", temperature=0.1)
        # LLM - 使用本地模型
        #self.llm = LLMAdapter.get_llm("local", "models/qwen2.5-1.5b-instruct", temperature=0.1)
        
        # 向量存储（支持持久化）
        self.vector_store = VectorStoreManager(
            embedding_model=self.embedding,
            collection_name="amicro_simple",
            persist_directory="./simple_rag_db"  # 持久化目录
        )
        
        # 增量文档处理器 - 用于监控文档变更
        self.incremental_processor = None
        self._setup_incremental_processor()
        
        # 注意：混合检索器将在知识库构建完成后初始化
        self.hybrid_retriever = None
    
    def _setup_incremental_processor(self):
        """设置增量文档处理器"""
        docs_folder = Path("docs")
        if docs_folder.exists() and docs_folder.is_dir():
            try:
                self.incremental_processor = IncrementalDocumentProcessor.create_with_vector_store(
                    docs_path=str(docs_folder),
                    vector_store_manager=self.vector_store,
                    chunk_size=300,  # 与knowledge base构建保持一致
                    chunk_overlap=50,
                    supported_extensions=['.txt', '.md', '.pdf', '.doc', '.docx']
                )
                print("🔄 增量处理器初始化成功，监控docs文件夹变更")
            except Exception as e:
                print(f"⚠️ 增量处理器初始化失败: {e}")
                self.incremental_processor = None
        else:
            print("ℹ️ 未发现docs文件夹，跳过增量处理器设置")
    
    def _build_knowledge_base(self):
        """构建知识库"""
        
        # 检查是否已有持久化的向量库
        if self.vector_store.is_persistent() and len(self.vector_store) > 0:
            print(f"🔄 发现已有持久化向量库，共 {len(self.vector_store)} 个文档块")
            print(f"⚡ 跳过文档处理，直接加载现有向量库")
            # 即使从持久化加载，也需要初始化混合检索器
            self._setup_hybrid_retriever()
            return
        
        # 检查是否使用docs文件夹
        import os
        from langchain_core.documents import Document
        
        docs_folder = Path("RAG/docs")
        if docs_folder.exists() and any(docs_folder.glob("*.txt")):
            print(f"📚 发现docs文件夹，加载章节文档...")
            documents = []
            
            # 加载所有txt文件
            for file_path in sorted(docs_folder.glob("*.txt")):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if content:
                        doc = Document(
                            page_content=content,
                            metadata={
                                'source': str(file_path),
                                'filename': file_path.name,
                                'chapter': file_path.stem  # 文件名作为章节
                            }
                        )
                        documents.append(doc)
                        
                except Exception as e:
                    print(f"⚠️ 读取文件失败 {file_path.name}: {e}")
            
            print(f"📄 成功加载 {len(documents)} 个章节文档")
            total_chars = sum(len(doc.page_content) for doc in documents)
            print(f"📝 总字符数: {total_chars:,}")
            
        else:
            # 降级到单文件模式
            print(f"📚 处理单文档: {self.document_path}")
            with open(self.document_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            documents = [Document(
                page_content=content,
                metadata={'source': self.document_path}
            )]
            print(f"📄 文档加载完成，长度: {len(content)} 字符")
        
        # 使用语义分块添加到向量存储
        self.vector_store.create_from_documents(
            documents,
            chunk_size=300,  # 使用更小的块以适应语义分块
            chunk_overlap=50
        )
        
        print(f"💾 知识库构建完成，共 {len(self.vector_store)} 个文档块")
        
        # 在知识库构建完成后初始化混合检索器
        self._setup_hybrid_retriever()
    
    def _setup_rag_chain(self):
        """设置RAG链"""
        # 创建检索器
        retriever = self.vector_store._create_retriever(
            search_type="similarity",
            search_kwargs={"k":8}
        )
        #你是一微半导体公司的智能助手。请基于以下文档内容回答用户问题。
        # RAG提示模板
        prompt = PromptTemplate(
            template="""你是一个熟读红楼梦的智能助手。请基于以下文档内容回答用户问题。

文档内容：
{context}

用户问题：{question}

请基于文档内容准确回答，同时用简短的几句话说明你的依据。如果文档中没有相关信息请说明。

回答：""",
            input_variables=["context", "question"]
        )
        
        # 格式化文档
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # 构建RAG链
        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
    
    def query(self, question: str, show_sources: bool = False) -> Dict[str, Any]:
        """查询RAG系统"""
        print(f"❓ 查询: {question}")
        total_start_time = time.time()
        
        # 性能统计字典
        performance_stats = {}
        
        try:
            # 1. 混合检索阶段
            print(f"🔍 开始混合检索...")
            retrieval_start = time.time()
            
            if self.hybrid_retriever:
                # 使用混合检索 (向量 + BM25 + RRF)
                hybrid_results = self.hybrid_retriever.hybrid_search(
                    query=question,
                    k=8,                  # 最终返回5个文档
                    vector_weight=0.6,    # 向量检索权重
                    bm25_weight=0.4,      # BM25检索权重
                    vector_k=10,          # 向量检索返回10个候选
                    bm25_k=10             # BM25检索返回10个候选
                )
                # 提取文档（忽略RRF分数）
                retrieved_docs = [doc for doc, score in hybrid_results]
            else:
                # 降级到基础向量检索
                print("⚠️ 混合检索器不可用，使用基础向量检索")
                retriever = self.vector_store._create_retriever(
                    search_type="similarity",
                    search_kwargs={"k":8}
                )
                retrieved_docs = retriever.invoke(question)
            
            retrieval_time = time.time() - retrieval_start
            performance_stats["retrieval_time"] = retrieval_time
            print(f"📚 混合检索完成，找到{len(retrieved_docs)}个相关文档，耗时: {retrieval_time:.2f}秒")
            
            # 2. 文档后处理阶段
            print(f"📊 文档后处理...")
            rerank_start = time.time()
            
            # 确保不超过5个文档
            retrieved_docs = retrieved_docs[:8]
            print(f"✅ 最终使用{len(retrieved_docs)}个文档")
            
            rerank_time = time.time() - rerank_start
            performance_stats["rerank_time"] = rerank_time
            print(f"📊 后处理耗时: {rerank_time:.2f}秒")
            
            # 3. 文档格式化阶段
            print(f"📝 开始文档格式化...")
            format_start = time.time()
            
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            context = format_docs(retrieved_docs)
            format_time = time.time() - format_start
            performance_stats["format_time"] = format_time
            print(f"📄 文档格式化完成，上下文长度: {len(context)}字符，耗时: {format_time:.2f}秒")
            
            # 4. 提示词组装阶段
            print(f"🔧 开始提示词组装...")
            prompt_start = time.time()
            
            # RAG提示模板
            prompt = PromptTemplate(
                template="""你是一个熟读红楼梦的智能助手。请基于以下文档内容回答用户问题。

文档内容：
{context}

用户问题：{question}

请基于文档内容准确回答，同时用简短的几句话说明你的依据。如果文档中没有相关信息请说明。

回答：""",
                input_variables=["context", "question"]
            )
            
            final_prompt = prompt.format(context=context, question=question)
            prompt_time = time.time() - prompt_start
            performance_stats["prompt_time"] = prompt_time
            print(f"🔧 提示词组装完成，最终提示长度: {len(final_prompt)}字符，耗时: {prompt_time:.2f}秒")
            
            # 5. LLM推理阶段
            print(f"🤖 开始LLM推理...")
            llm_start = time.time()
            
            answer = self.llm.invoke(final_prompt)
            
            llm_time = time.time() - llm_start
            performance_stats["llm_time"] = llm_time
            print(f"🤖 LLM推理完成，回答长度: {len(str(answer))}字符，耗时: {llm_time:.2f}秒")
            
            # 6. 来源获取阶段（可选）
            sources = []
            source_time = 0
            if show_sources:
                print(f"📚 开始获取来源文档...")
                source_start = time.time()
                sources = self.vector_store.search_similarity(question, k=5)
                source_time = time.time() - source_start
                print(f"📚 来源获取完成，获得{len(sources)}个来源，耗时: {source_time:.2f}秒")
            
            performance_stats["source_time"] = source_time
            
            # 计算总耗时
            total_time = time.time() - total_start_time
            performance_stats["total_time"] = total_time
            
            # 打印性能统计
            print(f"\n📊 性能统计:")
            print(f"  🔍 混合检索: {retrieval_time:.2f}秒 ({retrieval_time/total_time*100:.1f}%)")
            print(f"  📊 后处理: {rerank_time:.2f}秒 ({rerank_time/total_time*100:.1f}%)")
            print(f"  📝 文档格式化: {format_time:.2f}秒 ({format_time/total_time*100:.1f}%)")
            print(f"  🔧 提示词组装: {prompt_time:.2f}秒 ({prompt_time/total_time*100:.1f}%)")
            print(f"  🤖 LLM推理: {llm_time:.2f}秒 ({llm_time/total_time*100:.1f}%)")
            if show_sources:
                print(f"  📚 来源获取: {source_time:.2f}秒 ({source_time/total_time*100:.1f}%)")
            print(f"  ⏱️ 总耗时: {total_time:.2f}秒")
            
            # 找出最耗时的环节
            time_stages = {
                #"向量检索": retrieval_time,
                #"文档筛选": rerank_time, 
                "混合检索": retrieval_time,
                "后处理": rerank_time, 
                "LLM推理": llm_time,
                "其他": format_time + prompt_time + source_time
            }
            max_stage = max(time_stages.items(), key=lambda x: x[1])
            print(f"  🎯 最耗时环节: {max_stage[0]}")
            
            result = {
                "question": question,
                "answer": answer,
                "response_time": f"{total_time:.2f}秒",
                "performance_stats": {
                    #"retrieval_time": f"{retrieval_time:.2f}秒",
                    #"filter_time": f"{rerank_time:.2f}秒",
                    "hybrid_retrieval_time": f"{retrieval_time:.2f}秒",
                    "post_process_time": f"{rerank_time:.2f}秒",
                    "format_time": f"{format_time:.2f}秒", 
                    "prompt_time": f"{prompt_time:.2f}秒",
                    "llm_time": f"{llm_time:.2f}秒",
                    "source_time": f"{source_time:.2f}秒",
                    "total_time": f"{total_time:.2f}秒",
                    "bottleneck": max_stage[0]
                },
                "sources": [{"content": doc.page_content[:200] + "...", 
                           "chunk_id": doc.metadata.get("chunk_id")} 
                          for doc in sources] if show_sources else []
            }
            
            print(f"✅ 查询完成！")
            return result
            
        except Exception as e:
            error_time = time.time() - total_start_time
            print(f"❌ 查询出错，总耗时: {error_time:.2f}秒")
            return {
                "question": question,
                "answer": f"抱歉，处理问题时出错: {str(e)}",
                "error": str(e),
                "response_time": f"{error_time:.2f}秒",
                "performance_stats": {"error": True}
            }
    
    
    def interactive(self):
        """交互模式"""
        print("\n" + "="*60)
        print("💬 交互模式")
        print("📝 命令说明:")
        print("  - 直接输入问题进行查询")
        print("  - '/update' - 检查并更新文档")
        print("  - '/status' - 查看文档库状态")
        print("  - '/help' - 显示帮助")
        print("  - 'quit' - 退出")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n🤔 请输入问题或命令: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                # 处理命令
                if user_input.startswith('/'):
                    command = user_input[1:].lower()
                    
                    if command == 'update':
                        print("🔄 执行增量文档更新...")
                        update_result = self.update_documents()
                        print(f"📊 更新结果: {update_result['message']}")
                        if update_result['status'] == 'success':
                            print(f"📈 处理统计: 新增{update_result.get('new_files', 0)}个，修改{update_result.get('modified_files', 0)}个文件")
                    
                    elif command == 'status':
                        print("📊 文档库状态:")
                        status = self.get_docs_status()
                        for key, value in status.items():
                            print(f"  📌 {key}: {value}")
                    
                    elif command == 'help':
                        print("📖 可用命令:")
                        print("  /update - 检查并更新docs文件夹中的文档")
                        print("  /status - 查看当前文档库状态")
                        print("  /help   - 显示此帮助信息")
                        print("  quit    - 退出程序")
                        print("\n💡 提示: 将文档放在 RAG/docs/ 文件夹中可以使用增量更新功能")
                    
                    else:
                        print(f"❓ 未知命令: {command}，输入 '/help' 查看帮助")
                    
                    continue
                
                # 正常问答
                result = self.query(user_input, show_sources=True)
                print(f"\n💬 回答:\n{result['answer'].content}")
                print(f"\n⏱️ 耗时: {result['response_time']}")
                
                # 显示来源
                if result['sources']:
                    print(f"\n📚 相关文档片段:")
                    for i, source in enumerate(result['sources'], 1):
                        print(f"  {i}. {source['content']}")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
    
    def update_documents(self) -> Dict[str, Any]:
        """
        触发增量文档更新（简单接口）
        """
        if not self.incremental_processor:
            return {
                "status": "error",
                "message": "增量处理器未初始化，请确保docs文件夹存在"
            }
        
        # 调用增量处理器的方法
        result = self.incremental_processor.update_documents()
        
        # 如果有文档更新，重新初始化混合检索器
        if result.get("status") == "success" and result.get("total_processed", 0) > 0:
            self._refresh_hybrid_retriever()
        
        return result
    
    def get_docs_status(self) -> Dict[str, Any]:
        """
        获取文档库状态信息（简单接口）
        """
        base_status = {
            "vector_store_size": len(self.vector_store),
            "incremental_processor_available": self.incremental_processor is not None,
            "hybrid_retriever_available": self.hybrid_retriever is not None,
        }
        
        if self.incremental_processor:
            incremental_status = self.incremental_processor.get_comprehensive_status()
            base_status.update(incremental_status)
        
        return base_status
    
    def _setup_hybrid_retriever(self):
        """初始化混合检索器（在向量库有数据后）"""
        try:
            print("🔄 初始化混合检索器...")
            # 使用更宽松的BM25匹配条件
            self.hybrid_retriever = HybridRetriever(
                self.vector_store,
                bm25_min_match_ratio=0.2,  # 降低到20%的词匹配即可
                bm25_score_threshold=0.001  # 降低分数阈值
            )
            print("✅ 混合检索器初始化成功 (向量检索 + BM25 + RRF)")
        except Exception as e:
            print(f"⚠️ 混合检索器初始化失败: {e}")
            print("⚠️ 将使用基础向量检索模式")
            self.hybrid_retriever = None
    
    def _refresh_hybrid_retriever(self):
        """重新初始化混合检索器（用于增量更新后）"""
        print("🔄 重新初始化混合检索器（增量更新后）...")
        self._setup_hybrid_retriever()
    
    def _start_auto_update_thread(self):
        """启动自动更新后台线程"""
        if self._auto_update_thread is None or not self._auto_update_thread.is_alive():
            self._stop_auto_update.clear()
            self._auto_update_thread = threading.Thread(
                target=self._auto_update_worker,
                daemon=True,
                name="AutoUpdateThread"
            )
            self._auto_update_thread.start()
            print(f"🔄 自动更新线程已启动")
    
    def _auto_update_worker(self):
        """自动更新工作线程"""
        while not self._stop_auto_update.is_set():
            try:
                # 等待5分钟，如果收到停止信号则立即退出
                if self._stop_auto_update.wait(timeout=300):  # 5分钟
                    break
                
                # 执行更新检查
                result = self.update_documents()
                
                if result["status"] == "success" and result.get("total_processed", 0) > 0:
                    total_processed = result.get("total_processed", 0)
                    new_files = result.get("new_files", 0)
                    modified_files = result.get("modified_files", 0)
                    print(f"🎉 自动更新完成: 新增{new_files}个文件，修改{modified_files}个文件，处理{total_processed}个文档块")
                
                # 无变更时静默处理，不打印日志
                    
            except Exception as e:
                print(f"❌ 自动更新出错: {e}")
                # 出错后等待1分钟再继续
                if self._stop_auto_update.wait(timeout=60):
                    break
    
    def stop_auto_update(self):
        """停止自动更新"""
        if self._auto_update_thread and self._auto_update_thread.is_alive():
            self._stop_auto_update.set()
            self._auto_update_thread.join(timeout=5)
    
    def __del__(self):
        """析构时确保自动更新线程停止"""
        try:
            if hasattr(self, '_stop_auto_update'):
                self.stop_auto_update()
        except:
            pass

def main():
    """主函数"""
    rag = None
    try:
        # 初始化RAG系统
        rag = SimpleRAG("docs/1.txt")
        rag.interactive()
            
    except KeyboardInterrupt:
        print("\n🛑 收到退出信号...")
    except Exception as e:
        print(f"❌ 系统错误: {e}")
    finally:
        # 确保停止自动更新线程
        if rag:
            rag.stop_auto_update()
        print("👋 程序已退出")

if __name__ == "__main__":
    main() 