#!/usr/bin/env python3
"""
RAG 工具类 - 供 Agent 调用
基于现有的 SimpleRAG 类封装为 LangChain 工具
"""

import os
import time
from typing import Dict, Any, Optional, Type
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

# 导入现有的 SimpleRAG 类
from .rag import SimpleRAG


class RAGInput(BaseModel):
    """RAG 工具的输入模型"""
    question: str = Field(description="需要查询的问题")
    include_sources: bool = Field(default=False, description="是否包含来源信息")


class RAGTool(BaseTool):
    """
    RAG 工具类 - 符合 LangChain 工具标准
    可以被 Agent 调用进行文档检索和问答
    """
    
    name: str = "rag_search"
    description: str = "基于一微半导体公司文档内容回答问题的工具。输入问题，返回基于文档内容的准确答案。"
    args_schema: Type[BaseModel] = RAGInput
    
    # RAG 系统配置
    document_path: str = "1.txt"
    _rag_system: Optional[SimpleRAG] = None
    _initialized: bool = False
    
    def __init__(self, document_path: str = "1.txt", **kwargs):
        """
        初始化 RAG 工具
        
        Args:
            document_path: 文档文件路径
            **kwargs: 其他参数
        """
        super().__init__(**kwargs)
        self.document_path = document_path
        self._ensure_initialized()
    
    def _ensure_initialized(self):
        """确保 RAG 系统已初始化"""
        if self._initialized and self._rag_system is not None:
            return
            
        try:
            print(f"🚀 初始化 RAG 工具...")
            print(f"📁 文档路径: {self.document_path}")
            
            # 创建 SimpleRAG 实例
            self._rag_system = SimpleRAG(document_path=self.document_path)
            self._initialized = True
            
            print(f"✅ RAG 工具就绪！")
            
        except Exception as e:
            print(f"❌ RAG 工具初始化失败: {e}")
            raise RuntimeError(f"RAG 工具初始化失败: {e}")
    
    def _run(
        self,
        question: str,
        include_sources: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        执行 RAG 查询 - 这是 Agent 调用的主要方法
        
        Args:
            question: 用户问题
            include_sources: 是否包含来源信息
            run_manager: 回调管理器
            
        Returns:
            str: 基于文档内容的回答
        """
        try:
            self._ensure_initialized()
            if self._rag_system is None:
                raise RuntimeError("RAG system is not initialized.")
            
            print(f"🔍 Agent 查询: {question}")
            start_time = time.time()
            
            # 使用 SimpleRAG 进行查询
            result = self._rag_system.query(question, show_sources=include_sources)
            
            elapsed_time = time.time() - start_time
            print(f"✅ 查询完成，耗时: {elapsed_time:.2f}秒")
            
            # 格式化返回结果
            if include_sources and result.get('sources'):
                # 包含来源信息的详细回答
                response = f"回答: {result['answer']}\n\n"
                response += f"相关文档片段:\n"
                for i, source in enumerate(result['sources'], 1):
                    response += f"{i}. {source['content']}\n"
                return response
            else:
                # 简单回答
                return result['answer']
                
        except Exception as e:
            error_msg = f"RAG 查询失败: {str(e)}"
            print(f"❌ {error_msg}")
            return f"抱歉，查询文档时出现错误: {str(e)}"
    
    def query_with_details(self, question: str) -> Dict[str, Any]:
        """
        查询并返回详细信息
        这个方法可以被直接调用，不通过 Agent
        
        Args:
            question: 用户问题
            
        Returns:
            Dict[str, Any]: 包含答案、来源和其他信息的字典
        """
        try:
            self._ensure_initialized()
            if self._rag_system is None:
                raise RuntimeError("RAG system is not initialized.")
            return self._rag_system.query(question, show_sources=True)
        except Exception as e:
            return {
                "question": question,
                "answer": f"抱歉，处理问题时出错: {str(e)}",
                "error": str(e),
                "response_time": "0.00秒",
                "sources": []
            }
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取 RAG 系统信息"""
        if not self._initialized:
            return {"status": "未初始化"}
        
        return {
            "status": "已初始化",
            "document_path": self.document_path,
            "vector_store_size": len(self._rag_system.vector_store) if self._rag_system else 0,
            "tool_name": self.name,
            "tool_description": self.description
        }


def create_rag_tool(document_path: str = "1.txt") -> RAGTool:
    """
    创建 RAG 工具实例的工厂函数
    
    Args:
        document_path: 文档文件路径
        
    Returns:
        RAGTool: 配置好的 RAG 工具实例
    """
    return RAGTool(document_path=document_path)


# 便捷的全局实例（使用默认配置）
default_rag_tool = None

def get_default_rag_tool() -> RAGTool:
    """获取默认的 RAG 工具实例（单例模式）"""
    global default_rag_tool
    if default_rag_tool is None:
        default_rag_tool = create_rag_tool()
    return default_rag_tool


# 为了向后兼容，提供简单的函数接口
def rag_search(question: str, include_sources: bool = False) -> str:
    """
    简单的 RAG 搜索函数
    
    Args:
        question: 问题
        include_sources: 是否包含来源
        
    Returns:
        str: 回答
    """
    tool = get_default_rag_tool()
    return tool._run(question, include_sources)


if __name__ == "__main__":
    # 测试工具
    print("🧪 测试 RAG 工具")
    print("=" * 50)
    
    # 创建工具实例
    rag_tool = create_rag_tool()
    
    # 测试查询
    test_questions = [
        "一微半导体是什么公司？",
        "员工迟到会有什么处罚？",
        "公司的核心价值观是什么？"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 测试 {i}: {question}")
        print("-" * 40)
        
        # 测试简单查询
        answer = rag_tool._run(question)
        print(f"💬 回答: {answer}")
        
        # 测试详细查询
        if i == 1:  # 只对第一个问题测试详细信息
            print(f"\n📚 详细信息:")
            detailed = rag_tool.query_with_details(question)
            print(f"耗时: {detailed['response_time']}")
            if detailed.get('sources'):
                print(f"来源数量: {len(detailed['sources'])}")
    
    # 显示系统信息
    print(f"\n🔧 系统信息:")
    info = rag_tool.get_system_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print(f"\n✅ 测试完成！") 