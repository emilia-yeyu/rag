import os
from typing import List, Dict, Any, Optional
from langchain_tavily import TavilySearch, TavilyExtract

class TavilyManagerError(Exception):
    """自定义 Tavily 管理器错误。"""
    pass

class TavilyManager:
    """
    封装 Tavily Search 和 Extract API 的配置和调用。
    """
    def __init__(self,
                 max_search_results: int,
                 extract_depth: str,
                 search_depth: str
                ):
        """
        初始化 TavilyManager。

        Args:
            max_search_results: Tavily Search 返回的最大结果数。
            extract_depth: Tavily Extract 的提取深度 ('basic' 或 'advanced')。
            search_depth: Tavily Search 的搜索深度 ('basic' 或 'advanced')。

        Raises:
            TavilyManagerError: 如果 TAVILY_API_KEY 环境变量未设置或导入/初始化工具时出错。
        """
        # 仅从环境变量获取 API Key
        self.api_key = os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise TavilyManagerError("未设置 TAVILY_API_KEY 环境变量。请设置该环境变量后重试。")

        # 使用传入的参数
        self.max_search_results = max_search_results
        self.extract_depth = extract_depth
        self.search_depth = search_depth

        try:
            self.search_tool = TavilySearch(
                max_results=self.max_search_results,
                search_depth=self.search_depth
            )
            self.extract_tool = TavilyExtract(extract_depth=self.extract_depth)
        except ImportError:
            raise TavilyManagerError(
                "无法导入 langchain-tavily，请执行: pip install -U langchain-tavily"
            )
        except Exception as e:
            raise TavilyManagerError(f"初始化 Tavily 工具出错: {e}")

    #TODO 查询优化
    def search(self, query: str) -> Dict[str, Any]:
        """
        使用 Tavily Search API 执行搜索查询。

        Args:
            query: 搜索查询字符串。

        Returns:
            Tavily Search API 返回的完整结果字典。
            如果出错则返回包含错误信息的字典。
        """
        try:
            print(f"  [Tavily] 正在搜索: '{query}' (max_results={self.max_search_results}, search_depth={self.search_depth})")
            results = self.search_tool.invoke({"query": query})
            print(f"  [Tavily] 搜索完成，找到 {len(results.get('results', []))} 个结果。")
            return results
        except Exception as e:
            print(f"错误: Tavily 搜索查询 '{query}' 时出错: {e}")
            return {"error": f"Tavily 搜索失败: {e}", "query": query}

    def extract(self, urls: List[str]) -> Dict[str, Any]:
        """
        使用 Tavily Extract API 从 URL 列表提取内容。

        Args:
            urls: 需要提取内容的 URL 列表。

        Returns:
            Tavily Extract API 返回的完整结果字典。
            如果出错则返回包含错误信息的字典。
        """
        if not urls:
            return {"results": [], "failed_results": [], "response_time": 0, "info": "输入 URL 列表为空。"}

        try:
            print(f"  [Tavily] 正在从 {len(urls)} 个 URL 提取内容 (depth={self.extract_depth})...")
            results = self.extract_tool.invoke({"urls": urls})
            success_count = len(results.get('results', []))
            fail_count = len(results.get('failed_results', []))
            print(f"  [Tavily] 提取完成。成功: {success_count}, 失败: {fail_count}")
            if fail_count > 0:
                 print(f"  [Tavily] 提取失败的 URLs: {results.get('failed_results')}")
            return results
        except Exception as e:
            print(f"错误: Tavily 提取 URL 时出错: {e}")
            return {"error": f"Tavily 提取失败: {e}", "urls": urls}