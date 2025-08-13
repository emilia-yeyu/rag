#!/usr/bin/env python3
"""
基于1.txt的简单RAG系统
使用现有组件构建的单文件解决方案
"""

import os
import time
from typing import Dict, Any, List
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 导入现有组件
from document_loader.local_document_processor import LocalDocumentProcessor
from embedding.adapter import EmbeddingAdapter
from llm.adapter import LLMAdapter
from vector_store.vector_store import VectorStoreManager

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
        print(f"🚀 初始化简单RAG系统...")
        print(f"📁 文档路径: {self.document_path}")
        
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
        # 嵌入模型
        self.embedding = EmbeddingAdapter.get_embedding("dashscope", "text-embedding-v3")
        
        # LLM
        self.llm = LLMAdapter.get_llm("dashscope", "qwen-plus-latest", temperature=0.1)
        
        # 向量存储（支持持久化）
        self.vector_store = VectorStoreManager(
            embedding_model=self.embedding,
            collection_name="amicro_simple",
            persist_directory="./simple_rag_db"  # 持久化目录
        )
    
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
            chunk_size=800,
            chunk_overlap=100
        )
        
        print(f"💾 知识库构建完成，共 {len(self.vector_store)} 个文档块")
    
    def _setup_rag_chain(self):
        """设置RAG链"""
        # 创建检索器
        retriever = self.vector_store._create_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
        
        # RAG提示模板
        prompt = PromptTemplate(
            template="""你是一微半导体公司的智能助手。请基于以下文档内容回答用户问题。

文档内容：
{context}

用户问题：{question}

请基于文档内容准确回答，如果文档中没有相关信息请说明。

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
        start_time = time.time()
        
        try:
            # 生成回答
            answer = self.rag_chain.invoke(question)
            
            # 获取相关文档（用于显示来源）
            sources = []
            if show_sources:
                sources = self.vector_store.search_similarity(question, k=3)
            
            result = {
                "question": question,
                "answer": answer,
                "response_time": f"{time.time() - start_time:.2f}秒",
                "sources": [{"content": doc.page_content[:200] + "...", 
                           "chunk_id": doc.metadata.get("chunk_id")} 
                          for doc in sources] if show_sources else []
            }
            
            print(f"✅ 回答完成，耗时: {result['response_time']}")
            return result
            
        except Exception as e:
            return {
                "question": question,
                "answer": f"抱歉，处理问题时出错: {str(e)}",
                "error": str(e),
                "response_time": f"{time.time() - start_time:.2f}秒"
            }
    
    def demo(self):
        """演示功能"""
        print("\n" + "="*60)
        print("🎯 AMICRO RAG系统演示")
        print("="*60)
        
        # 测试问题
        questions = [
            "一微半导体是什么公司？",
            "员工迟到会有什么处罚？",
            "公司的核心价值观是什么？",
            "公司有多少员工？",
            "公司的考勤时间是怎样的？"
        ]
        
        for i, q in enumerate(questions, 1):
            print(f"\n🔍 问题{i}: {q}")
            print("-" * 40)
            result = self.query(q)
            print(f"💬 回答: {result['answer']}")
            print(f"⏱️ 耗时: {result['response_time']}")
        
        print(f"\n✅ 演示完成！")
    
    def interactive(self):
        """交互模式"""
        print("\n" + "="*60)
        print("💬 交互模式 (输入 'quit' 退出)")
        print("="*60)
        
        while True:
            try:
                question = input("\n🤔 请输入问题: ").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                
                result = self.query(question, show_sources=True)
                print(f"\n💬 回答:\n{result['answer']}")
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

def main():
    """主函数"""
    try:
        # 初始化RAG系统
        rag = SimpleRAG("1.txt")
        
        # 选择模式
        print("\n选择模式:")
        print("1. 演示模式")
        print("2. 交互模式")
        
        choice = input("请选择 (1/2): ").strip()
        
        if choice == "1":
            rag.demo()
        elif choice == "2":
            rag.interactive()
        else:
            print("❌ 无效选择，运行演示模式")
            rag.demo()
            
    except Exception as e:
        print(f"❌ 系统错误: {e}")

if __name__ == "__main__":
    main() 