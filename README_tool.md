# RAG 工具使用指南

## 概述

基于现有的 `SimpleRAG` 类，我们创建了一个符合 LangChain 工具标准的 `RAGTool` 类，可以被 Agent 直接调用。

## 文件结构

```
kzk/RAG/
├── rag.py              # 原始 SimpleRAG 类
├── rag_tool.py         # 新的 RAGTool 工具类
├── agent_example.py    # Agent 使用示例
├── 1.txt              # 文档数据
└── README_tool.md     # 本文档
```

## 快速开始

### 1. 基本使用

```python
from rag_tool import create_rag_tool

# 创建 RAG 工具
rag_tool = create_rag_tool("1.txt")

# 直接调用
answer = rag_tool._run("一微半导体是什么公司？")
print(answer)
```

### 2. Agent 集成

```python
from langchain.agents import initialize_agent, AgentType
from llm.adapter import LLMAdapter
from rag_tool import create_rag_tool

# 创建工具和 LLM
rag_tool = create_rag_tool("1.txt")
llm = LLMAdapter.get_llm("dashscope", "qwen-plus-latest")

# 创建 Agent
agent = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 使用 Agent
response = agent.run("一微半导体是什么公司？")
print(response)
```

## 主要特性

### ✅ 符合 LangChain 工具标准
- 继承 `BaseTool` 类
- 实现 `_run()` 方法
- 定义输入模式 `RAGInput`

### ✅ 多种使用方式
1. **直接调用**: `rag_tool._run(question)`
2. **Agent 集成**: `initialize_agent(tools=[rag_tool])`
3. **函数式调用**: `rag_search(question)`
4. **详细查询**: `rag_tool.query_with_details(question)`

### ✅ 灵活配置
- 自定义文档路径
- 支持来源信息显示
- 单例模式支持

## 使用示例

### 示例1: 简单 Agent

```python
from rag_tool import create_rag_tool
from langchain.agents import initialize_agent, AgentType
from llm.adapter import LLMAdapter

# 创建组件
rag_tool = create_rag_tool("1.txt")
llm = LLMAdapter.get_llm("dashscope", "qwen-plus-latest")

# 创建 Agent
agent = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# 查询
response = agent.run("公司的考勤制度是什么？")
```

### 示例2: 带记忆的 Agent

```python
from langchain.memory import ConversationBufferMemory

# 创建记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 创建带记忆的 Agent
agent = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory
)

# 连续对话
agent.run("一微半导体是什么公司？")
agent.run("刚才提到的公司有什么特色？")  # 会记住之前的对话
```

### 示例3: 自定义工具装饰器

```python
from langchain.tools import tool
from rag_tool import rag_search

@tool
def amicro_rag_search(question: str) -> str:
    """基于一微半导体公司文档内容回答问题"""
    return rag_search(question)

@tool
def amicro_rag_search_with_sources(question: str) -> str:
    """基于一微半导体公司文档内容回答问题，包含来源信息"""
    return rag_search(question, include_sources=True)

# 使用多个工具
agent = initialize_agent(
    tools=[amicro_rag_search, amicro_rag_search_with_sources],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)
```

## API 参考

### RAGTool 类

```python
class RAGTool(BaseTool):
    name: str = "rag_search"
    description: str = "基于一微半导体公司文档内容回答问题的工具"
    
    def __init__(self, document_path: str = "1.txt")
    def _run(self, question: str, include_sources: bool = False) -> str
    def query_with_details(self, question: str) -> Dict[str, Any]
    def get_system_info(self) -> Dict[str, Any]
```

### 工厂函数

```python
def create_rag_tool(document_path: str = "1.txt") -> RAGTool
def get_default_rag_tool() -> RAGTool
def rag_search(question: str, include_sources: bool = False) -> str
```

## 运行示例

```bash
# 进入 RAG 目录
cd kzk/RAG

# 测试工具
python rag_tool.py

# 运行 Agent 示例
python agent_example.py
```

## 配置要求

### 环境变量
确保设置了以下环境变量：
```bash
export DASHSCOPE_API_KEY="your_api_key"
```

### 依赖项
- langchain
- langchain-core
- pydantic
- 现有的 RAG 系统依赖

## 错误处理

工具提供了完善的错误处理：

```python
try:
    rag_tool = create_rag_tool("nonexistent.txt")
except RuntimeError as e:
    print(f"初始化失败: {e}")

# 查询时的错误处理
answer = rag_tool._run("问题")
if "抱歉，查询文档时出现错误" in answer:
    print("查询失败，请检查系统状态")
```

## 性能优化

- **懒加载**: RAG 系统在首次使用时才初始化
- **单例模式**: `get_default_rag_tool()` 返回同一个实例
- **持久化**: 向量库支持持久化存储

## 扩展开发

### 自定义工具

```python
class CustomRAGTool(RAGTool):
    def _run(self, question: str, include_sources: bool = False) -> str:
        # 自定义处理逻辑
        result = super()._run(question, include_sources)
        # 后处理
        return self.post_process(result)
```

### 添加新功能

1. 继承 `RAGTool` 类
2. 重写相关方法
3. 更新工具描述和参数

## 故障排除

### 常见问题

1. **文档找不到**
   ```
   FileNotFoundError: 文档文件不存在: xxx.txt
   ```
   解决：检查文档路径是否正确

2. **初始化失败**
   ```
   RuntimeError: RAG 工具初始化失败
   ```
   解决：检查环境变量和依赖项

3. **Agent 调用失败**
   ```
   Agent 错误: xxx
   ```
   解决：检查 LLM 配置和网络连接

### 调试模式

```python
# 获取系统信息
rag_tool = create_rag_tool()
info = rag_tool.get_system_info()
print(f"系统状态: {info}")

# 详细查询
result = rag_tool.query_with_details("测试问题")
print(f"详细结果: {result}")
```

## 最佳实践

1. **使用单例模式**: 对于同一文档，使用 `get_default_rag_tool()`
2. **错误处理**: 始终检查返回结果中的错误信息
3. **内存管理**: 长时间运行时注意内存使用
4. **日志记录**: 在生产环境中启用详细日志

## 许可证

本工具遵循项目的许可证协议。 