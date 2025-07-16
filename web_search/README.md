# TavilyManager 简明说明

本模块封装了 Tavily 的 Web 搜索与网页内容提取功能，便于在项目中统一调用。

## 主要功能
- **网页搜索**：通过 Tavily Search API，根据自然语言问题检索互联网上的相关网页信息。
- **网页内容提取**：通过 Tavily Extract API，从指定 URL 批量提取网页正文内容。

## 类与接口

### TavilyManager

#### 初始化
```python
from backend.web_search.tavily_manager import TavilyManager

# 需先设置环境变量 TAVILY_API_KEY
manager = TavilyManager(
    max_search_results=5,        # 搜索结果最大数量，默认5
    extract_depth='basic',       # 网页提取深度，'basic' 或 'advanced'，默认'basic'
    search_depth='basic'         # 搜索深度，'basic' 或 'advanced'，默认'basic'
)
```

#### 搜索接口
```python
result = manager.search("梅州市金田村土地资源现状")
# result 为 Tavily Search API 的完整响应字典，包含如下主要字段：
# {
#   'query': '...',
#   'results': [
#       {'url': '...', 'title': '...', 'content': '...', ...},
#       ...
#   ],
#   'response_time': ...,
#   ...
# }
```

#### 网页内容提取接口
```python
urls = [item['url'] for item in result.get('results', [])]
extract_result = manager.extract(urls)
# extract_result 为 Tavily Extract API 的完整响应字典，包含：
# {
#   'results': [
#       {'url': '...', 'raw_content': '...', ...},
#       ...
#   ],
#   'failed_results': [...],
#   'response_time': ...
# }
```

## 输入输出说明
- **search(query: str) -> dict**
  - 输入：自然语言问题字符串
  - 输出：Tavily Search API 的完整响应字典（含网页元信息、标题、摘要等）
- **extract(urls: List[str]) -> dict**
  - 输入：URL 字符串列表
  - 输出：Tavily Extract API 的完整响应字典（含网页正文、失败列表等）

## 依赖
- 需先安装 `langchain-tavily`：
  ```bash
  pip install -U langchain-tavily
  ```
- 需设置环境变量 `TAVILY_API_KEY`。

## 参考
- [Tavily 官方文档](https://docs.tavily.com/documentation/integrations/langchain) 