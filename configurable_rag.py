#!/usr/bin/env python3
"""
可配置RAG系统
支持多种检索策略的灵活组合
"""

import os
os.environ['TRANSFORMERS_OFFLINE'] = '0'
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import time
from typing import Dict, Any, List, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 导入现有组件
from document_loader.local_document_processor import LocalDocumentProcessor
from embedding.adapter import EmbeddingAdapter
# from embedding.reranker import AdaptiveReranker  # 已禁用重排序
from llm.adapter import LLMAdapter
from vector_store.vector_store import VectorStoreManager
from vector_store.configurable_retriever import ConfigurableRetriever

# 导入配置模块
from retrieval_config import RetrievalConfig, get_config, list_configs

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()


class ConfigurableRAG:
    """可配置RAG系统"""
    
    def __init__(self, document_path: str = "2.txt", retrieval_config: str = "comprehensive"):
        """
        初始化可配置RAG系统
        
        Args:
            document_path: 文档路径
            retrieval_config: 检索配置名称或RetrievalConfig对象
        """
        # 如果是相对路径，尝试在脚本所在目录查找
        if not os.path.isabs(document_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(script_dir, document_path)
            if os.path.exists(full_path):
                document_path = full_path
        
        self.document_path = document_path
        
        # 设置检索配置
        if isinstance(retrieval_config, str):
            self.config = get_config(retrieval_config)
        elif isinstance(retrieval_config, RetrievalConfig):
            self.config = retrieval_config
        else:
            raise ValueError("retrieval_config 必须是配置名称字符串或RetrievalConfig对象")
        
        print(f"🚀 初始化可配置RAG系统...")
        print(f"📁 文档路径: {self.document_path}")
        print(f"🔍 检索配置: {self.config.get_description()}")
        print(f"🎯 启用方法: {', '.join(self.config.get_enabled_methods())}")
        
        # 检查文档是否存在
        if not os.path.exists(document_path):
            raise FileNotFoundError(f"文档文件不存在: {document_path}")
        
        # 初始化组件
        self._setup_components()
        self._build_knowledge_base()
        self._setup_rag_chain()
        
        print(f"✅ RAG系统就绪！")
    
    def _setup_components(self):
        """设置组件"""
        # 嵌入模型 - 使用开源免费的 bge-large-zh-v1.5
        self.embedding = EmbeddingAdapter.get_embedding("bge", "BAAI/bge-large-zh-v1.5")
        
        # LLM
        self.llm = LLMAdapter.get_llm("dashscope", "qwen-turbo", temperature=0.1)
        
        # 向量存储（支持持久化）
        self.vector_store = VectorStoreManager(
            embedding_model=self.embedding,
            collection_name="configurable_rag",
            persist_directory="./configurable_rag_db"  # 持久化目录
        )
        
        # 可配置检索器
        self.retriever = None
        try:
            self.retriever = ConfigurableRetriever(self.vector_store, self.config)
            print("✅ 可配置检索器初始化成功")
        except Exception as e:
            print(f"⚠️ 可配置检索器初始化失败: {e}")
            print("⚠️ 将使用基础向量检索模式")
    
    def _build_knowledge_base(self):
        """构建知识库"""
        print(f"📚 处理文档: {self.document_path}")
        
        # 检查是否已有持久化的向量库
        if self.vector_store.is_persistent() and len(self.vector_store) > 0:
            print(f"🔄 发现已有持久化向量库，共 {len(self.vector_store)} 个文档块")
            print(f"⚡ 跳过文档处理，直接加载现有向量库")
            return
        
        # 直接读取单个文件
        with open(self.document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 创建文档对象
        from langchain_core.documents import Document
        document = Document(
            page_content=content,
            metadata={'source': self.document_path}
        )
        
        print(f"📄 文档加载完成，长度: {len(content)} 字符")
        
        # 添加到向量存储（自动分块）
        self.vector_store.create_from_documents(
            [document],
            chunk_size=1024,
            chunk_overlap=100
        )
        
        print(f"💾 知识库构建完成，共 {len(self.vector_store)} 个文档块")
    
    def _setup_rag_chain(self):
        """设置RAG链"""
        # 创建基础检索器作为备用
        self.fallback_retriever = self.vector_store._create_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # RAG提示模板
        self.prompt_template = PromptTemplate(
            template="""你是一个熟读红楼梦的智能助手。请基于以下文档内容回答用户问题。

文档内容（可能包含小说情节和人物信息）：
{context}

用户问题：{question}

请基于文档内容准确回答。如果文档中包含具体的人物信息（如姓名、年龄、职位等），请优先使用这些准确信息。如果文档中没有相关信息请说明。

回答：""",
            input_variables=["context", "question"]
        )
    
    def query(self, question: str, show_sources: bool = False, show_config: bool = False) -> Dict[str, Any]:
        """查询RAG系统"""
        print(f"❓ 查询: {question}")
        total_start_time = time.time()
        
        # 性能统计字典
        performance_stats = {}
        
        try:
            # 1. 检索阶段
            print(f"🔍 开始检索...")
            retrieval_start = time.time()
            
            if self.retriever:
                # 使用可配置检索器
                hybrid_results = self.retriever.search(question)
                retrieved_docs = [doc for doc, score in hybrid_results]
            else:
                # 降级到基础向量检索
                print("⚠️ 可配置检索器不可用，使用基础向量检索")
                retrieved_docs = self.fallback_retriever.invoke(question)
            
            retrieval_time = time.time() - retrieval_start
            performance_stats["retrieval_time"] = retrieval_time
            print(f"📚 检索完成，找到{len(retrieved_docs)}个相关文档，耗时: {retrieval_time:.2f}秒")
            
            # 2. 文档后处理阶段
            print(f"📊 文档后处理...")
            postprocess_start = time.time()
            
            # 确保不超过配置的k值
            retrieved_docs = retrieved_docs[:self.config.k]
            print(f"✅ 最终使用{len(retrieved_docs)}个文档")
            
            postprocess_time = time.time() - postprocess_start
            performance_stats["postprocess_time"] = postprocess_time
            print(f"📊 后处理耗时: {postprocess_time:.2f}秒")
            
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
            
            final_prompt = self.prompt_template.format(context=context, question=question)
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
            print(f"  🔍 检索: {retrieval_time:.2f}秒 ({retrieval_time/total_time*100:.1f}%)")
            print(f"  📊 后处理: {postprocess_time:.2f}秒 ({postprocess_time/total_time*100:.1f}%)")
            print(f"  📝 文档格式化: {format_time:.2f}秒 ({format_time/total_time*100:.1f}%)")
            print(f"  🔧 提示词组装: {prompt_time:.2f}秒 ({prompt_time/total_time*100:.1f}%)")
            print(f"  🤖 LLM推理: {llm_time:.2f}秒 ({llm_time/total_time*100:.1f}%)")
            if show_sources:
                print(f"  📚 来源获取: {source_time:.2f}秒 ({source_time/total_time*100:.1f}%)")
            print(f"  ⏱️ 总耗时: {total_time:.2f}秒")
            
            # 找出最耗时的环节
            time_stages = {
                "检索": retrieval_time,
                "后处理": postprocess_time, 
                "LLM推理": llm_time,
                "其他": format_time + prompt_time + source_time
            }
            max_stage = max(time_stages.items(), key=lambda x: x[1])
            print(f"  🎯 最耗时环节: {max_stage[0]}")
            
            # 构建返回结果
            result = {
                "question": question,
                "answer": answer,
                "response_time": f"{total_time:.2f}秒",
                "performance_stats": {
                    "retrieval_time": f"{retrieval_time:.2f}秒",
                    "postprocess_time": f"{postprocess_time:.2f}秒",
                    "format_time": f"{format_time:.2f}秒", 
                    "prompt_time": f"{prompt_time:.2f}秒",
                    "llm_time": f"{llm_time:.2f}秒",
                    "source_time": f"{source_time:.2f}秒",
                    "total_time": f"{total_time:.2f}秒",
                    "bottleneck": max_stage[0],
                    "retrieval_mode": self.config.get_description()
                },
                "sources": [{"content": doc.page_content[:200] + "...", 
                           "chunk_id": doc.metadata.get("chunk_id"),
                           "source_type": doc.metadata.get("search_type", "unknown")} 
                          for doc in sources] if show_sources else []
            }
            
            # 添加配置信息（可选）
            if show_config:
                result["config"] = self.retriever.get_config_summary() if self.retriever else {}
            
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
    
    def switch_config(self, new_config: str):
        """
        切换检索配置
        
        Args:
            new_config: 新的配置名称
        """
        print(f"🔄 切换检索配置: {new_config}")
        
        try:
            # 获取新配置
            self.config = get_config(new_config)
            
            # 重新初始化检索器
            self.retriever = ConfigurableRetriever(self.vector_store, self.config)
            
            print(f"✅ 检索配置已切换到: {self.config.get_description()}")
            print(f"🎯 启用方法: {', '.join(self.config.get_enabled_methods())}")
            
        except Exception as e:
            print(f"❌ 配置切换失败: {e}")
    
    def list_available_configs(self):
        """列出所有可用配置"""
        print("📋 可用检索配置:")
        configs = list_configs()
        for name, description in configs.items():
            current = " (当前)" if name == getattr(self.config, 'name', None) else ""
            print(f"  • {name}: {description}{current}")
    
    def interactive(self):
        """交互模式"""
        print("\n" + "="*70)
        print("💬 可配置RAG交互模式")
        print("="*70)
        print("命令说明:")
        print("  - 直接输入问题进行查询")
        print("  - 输入 'config' 查看当前配置")
        print("  - 输入 'switch <配置名>' 切换配置")
        print("  - 输入 'list' 列出所有配置")
        print("  - 输入 'quit' 退出")
        print("="*70)
        
        while True:
            try:
                user_input = input(f"\n🤔 请输入 [{self.config.get_description()}]: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                elif user_input.lower() == 'config':
                    print(f"📋 当前配置: {self.config.get_description()}")
                    print(f"🎯 启用方法: {', '.join(self.config.get_enabled_methods())}")
                    if self.retriever:
                        config_summary = self.retriever.get_config_summary()
                        weights = config_summary.get('weights', {})
                        print(f"⚖️ 权重分配: 向量={weights.get('vector', 0):.2f}, BM25={weights.get('bm25', 0):.2f}, SQL={weights.get('sql', 0):.2f}")
                elif user_input.lower() == 'list':
                    self.list_available_configs()
                elif user_input.lower().startswith('switch '):
                    config_name = user_input[7:].strip()
                    if config_name:
                        self.switch_config(config_name)
                    else:
                        print("❌ 请指定配置名称，例如: switch semantic")
                else:
                    # 执行查询
                    result = self.query(user_input, show_sources=True, show_config=False)
                    print(f"\n💬 回答:\n{result['answer'].content}")
                    print(f"\n⏱️ 耗时: {result['response_time']}")
                    print(f"🔍 检索模式: {result['performance_stats'].get('retrieval_mode', 'unknown')}")
                    
                    # 显示来源
                    if result['sources']:
                        print(f"\n📚 相关文档片段:")
                        for i, source in enumerate(result['sources'], 1):
                            source_type = source.get('source_type', 'unknown')
                            print(f"  {i}. [{source_type}] {source['content']}")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")


def main():
    """主函数"""
    import sys
    
    # 解析命令行参数
    config_name = "comprehensive"  # 默认配置
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    
    try:
        # 显示可用配置
        print("📋 可用检索配置:")
        configs = list_configs()
        for name, description in configs.items():
            print(f"  • {name}: {description}")
        print()
        
        # 初始化RAG系统
        rag = ConfigurableRAG("2.txt", retrieval_config=config_name)
        rag.interactive()
            
    except Exception as e:
        print(f"❌ 系统错误: {e}")


if __name__ == "__main__":
    main() 