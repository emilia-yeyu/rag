import os
from typing import List, Type, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader

class LocalDocumentProcessor:
    """
    处理本地文档加载。
    仅负责文档加载，不涉及切分、向量化和数据库存储。
    """
    def __init__(
        self,
        docs_path: str,
        loader_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化 LocalDocumentProcessor。

        参数:
            docs_path: 包含文档的目录路径。
            loader_kwargs: (可选) 文档加载器的关键字参数字典。
        """
        if not os.path.isdir(docs_path):
            raise ValueError(f"文档路径不存在或不是一个目录: {docs_path}")

        self.docs_path = docs_path
        self.loader_kwargs = loader_kwargs or {}

    def load_documents(self) -> List[Document]:
        """从指定目录加载原始文档。"""
        try:
            loader = DirectoryLoader(
                self.docs_path,
                loader_kwargs=self.loader_kwargs,
                show_progress=True, 
                use_multithreading=True,
                silent_errors=True  # 跳过无法加载的文件
            )
            documents = loader.load()
            return documents
        except Exception as e:
            raise RuntimeError(f"加载文档时出错: {e}") from e 