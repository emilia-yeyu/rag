import os
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

class VectorStoreManager:
    """
    向量存储管理器类，负责创建、管理和查询 Chroma 向量数据库。
    每个实例管理一个独立的向量库，支持增量更新和多种检索方法。
    支持内存模式和持久化模式。
    """
    #TODO 删除能力 查询日志log（）   ID管理
    def __init__(
        self,
        embedding_model: Embeddings,
        collection_name: str = "default_collection",
        persist_directory: Optional[str] = None  # 如果为None则使用内存模式，否则持久化到指定目录
    ):
        """
        初始化向量存储管理器。

        参数:
            embedding_model: LangChain Embeddings 模型实例，用于文本向量化。
            collection_name: Chroma 集合名称。
            persist_directory: 持久化目录，如果为None则使用内存模式。
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory

        # 根据是否提供持久化目录选择模式
        if self.persist_directory:
            # 确保持久化目录存在
            os.makedirs(self.persist_directory, exist_ok=True)
            print(f"[VectorStore] 使用持久化模式，目录: {self.persist_directory}")
        else:
            print(f"[VectorStore] 使用内存模式")

        # 初始化 Chroma 向量库
        try:
            print(f"[VectorStore] 正在初始化Chroma向量库...")
            if self.persist_directory:
                self._vector_store = Chroma(
                    embedding_function=self.embedding_model,
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory
                )
            else:
                self._vector_store = Chroma(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name
                )
                print(f"[VectorStore] Chroma初始化成功: {self._vector_store is not None}")
        except Exception as e:
            print(f"[VectorStore] ❌ Chroma初始化失败: {e}")
            print(f"[VectorStore] 异常类型: {type(e).__name__}")
            self._vector_store = None
            raise


    #TODO 需要改进切片策略
    def create_from_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> None:
        """
        从文档列表创建新的 Chroma 向量存储，覆盖任何现有实例。
        如果提供了 chunk_size，则会在创建前对文档进行切片。

        参数:
            documents: Document 对象列表，包含文本和元数据。
            chunk_size: （可选）文本块大小，如果提供则进行切片。
            chunk_overlap: （可选）文本块重叠大小，需与 chunk_size 同时提供。
        """
        docs_to_create = documents
        # 如果需要切片
        if chunk_size and chunk_size > 0 and documents:
            print(f"  [VectorStore] 对 {len(documents)} 个文档进行切片 (size={chunk_size}, overlap={chunk_overlap}) for creation...")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap or 0
            )
            docs_to_create = text_splitter.split_documents(documents)
            print(f"  [VectorStore] 切片完成，生成 {len(docs_to_create)} 个文档块。")

        if not docs_to_create:
            print("警告: 没有文档（或切片后为空）可用于创建向量存储")
            # 保留已有的空库或重新创建空库
            if self.persist_directory:
                self._vector_store = Chroma(
                    embedding_function=self.embedding_model,
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory
                )
            else:
                self._vector_store = Chroma(
                embedding_function=self.embedding_model,
                collection_name=self.collection_name
            )
            return

        try:
            mode_desc = "持久化" if self.persist_directory else "内存"
            print(f"正在从 {len(docs_to_create)} 个文档块创建新的 {mode_desc} Chroma 向量库 (集合: {self.collection_name})")
            # 使用 from_documents 创建或覆盖
            if self.persist_directory:
                self._vector_store = Chroma.from_documents(
                    documents=docs_to_create,
                    embedding=self.embedding_model, # Chroma v0.2+ uses 'embedding'
                    collection_name=self.collection_name,
                    persist_directory=self.persist_directory
                )
                print(f"[VectorStore] 向量库已保存到: {self.persist_directory}")
            else:
                self._vector_store = Chroma.from_documents(
                documents=docs_to_create,
                embedding=self.embedding_model, # Chroma v0.2+ uses 'embedding'
                collection_name=self.collection_name
            )

        except Exception as e:
            # 初始化失败时，确保 self._vector_store 为 None 或重新创建空库
            try:
                if self.persist_directory:
                    self._vector_store = Chroma(
                        embedding_function=self.embedding_model, 
                        collection_name=self.collection_name,
                        persist_directory=self.persist_directory
                    )
                else:
                    self._vector_store = Chroma(
                        embedding_function=self.embedding_model, 
                        collection_name=self.collection_name
                    )
            except Exception:
                 self._vector_store = None # 如果连空库都创建失败
            raise RuntimeError(f"创建 Chroma 向量存储时发生错误: {e}") from e

    def add_documents(
        self,
        documents: List[Document],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> None:
        """
        向现有 Chroma 向量存储添加新文档。如果向量存储尚未创建，则创建新的。
        如果提供了 chunk_size，则会在添加前对文档进行切片。

        参数:
            documents: Document 对象列表，包含文本和元数据。
            chunk_size: （可选）文本块大小，如果提供则进行切片。
            chunk_overlap: （可选）文本块重叠大小，需与 chunk_size 同时提供。
        """
        if not documents:
            return

        try:
            docs_to_add = documents
            # 如果需要切片
            if chunk_size and chunk_size > 0:
                print(f"  [VectorStore] 对 {len(documents)} 个文档进行切片 (size={chunk_size}, overlap={chunk_overlap})...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap or 0 # 如果 overlap 未提供则默认为 0
                )
                docs_to_add = text_splitter.split_documents(documents)
                print(f"  [VectorStore] 切片完成，生成 {len(docs_to_add)} 个文档块。")

            if not self._vector_store:
                print("向量存储尚未初始化，将从提供的文档创建新的内存向量库。")
                return self.create_from_documents(docs_to_add)

            mode_desc = "持久化" if self.persist_directory else "内存"
            print(f"正在向 {mode_desc} Chroma 向量库 (集合: {self.collection_name}) 添加 {len(docs_to_add)} 个文档块。")
            self._vector_store.add_documents(docs_to_add)
            
            if self.persist_directory:
                print(f"[VectorStore] 文档已保存到: {self.persist_directory}")

        except Exception as e:
            raise RuntimeError(f"添加文档到 Chroma 向量存储时发生错误: {e}") from e

    def get_all_documents(self) -> List[Document]:
        """
        获取 Chroma 向量存储中的所有文档。

        返回:
            Document 对象列表。
        """
        if not self._vector_store or not hasattr(self._vector_store, '_collection'):
            return []

        try:
            collection_data = self._vector_store._collection.get(include=["documents", "metadatas"])
            if collection_data and collection_data.get('documents') and collection_data.get('metadatas'):
                 docs = collection_data.get('documents', []) or []
                 metas = collection_data.get('metadatas', []) or []
                 if len(docs) == len(metas):
                     return [
                         Document(page_content=doc, metadata=meta or {})
                         for doc, meta in zip(docs, metas)
                         if doc is not None
                     ]
                 else:
                     print(f"警告: 文档和元数据数量不匹配 ({len(docs)} vs {len(metas)})，无法可靠地获取所有文档。")
                     return []
            return []
        except Exception as e:
            print(f"获取所有文档时发生错误: {e}")
            return []

    def _create_retriever(self, search_type: str, search_kwargs: Dict[str, Any]):
        """
        内部方法：创建指定搜索类型的检索器。

        参数:
            search_type: 搜索类型 ('similarity', 'mmr', 'similarity_score_threshold')
            search_kwargs: 搜索参数

        返回:
            检索器实例
        """
        if not self._vector_store:
            raise ValueError("Chroma 向量存储尚未初始化。请先调用 create_from_documents 或 add_documents。")

        return self._vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )

    def search_similarity(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        基于相似度在向量库中搜索文档。

        参数:
            query: 搜索查询文本。
            k: 最大返回结果数。

        返回:
            匹配的 Document 对象列表。
        """
        if not self._vector_store:
            raise ValueError("Chroma 向量存储尚未初始化。请先调用 create_from_documents 或 add_documents。")

        search_params = {'k': k}
        retriever = self._create_retriever('similarity', search_params)

        try:
            return retriever.invoke(query)
        except Exception as e:
            raise RuntimeError(f"执行相似度搜索时出错: {e}") from e

    def search_mmr(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        使用最大边际相关性(MMR)算法在向量库中搜索文档。
        MMR算法会在相关性和多样性之间取得平衡。

        参数:
            query: 搜索查询文本。
            k: 最大返回结果数。

        返回:
            匹配的 Document 对象列表。
        """
        if not self._vector_store:
            raise ValueError("Chroma 向量存储尚未初始化。请先调用 create_from_documents 或 add_documents。")

        search_params = {'k': k}
        retriever = self._create_retriever('mmr', search_params)

        try:
            return retriever.invoke(query)
        except Exception as e:
            raise RuntimeError(f"执行 MMR 搜索时出错: {e}") from e

    def search_with_threshold(
        self,
        query: str,
        similarity_threshold: float = 0.5,
    ) -> List[Document]:
        """
        使用相似度阈值在向量库中搜索文档。只返回相似度分数超过指定阈值的文档，返回所有满足条件的文档。

        参数:
            query: 搜索查询文本。
            similarity_threshold: 相似度阈值 (0-1)。

        返回:
            匹配的 Document 对象列表。
        """
        if not self._vector_store:
            raise ValueError("Chroma 向量存储尚未初始化。请先调用 create_from_documents 或 add_documents。")

        search_params = {
            'score_threshold': similarity_threshold
        }

        count = len(self)
        search_params['k'] = count if count > 0 else 100  # Use collection count or a large default

        retriever = self._create_retriever('similarity_score_threshold', search_params)

        try:
            return retriever.invoke(query)
        except Exception as e:
            raise RuntimeError(f"执行相似度阈值搜索时出错: {e}") from e

    def search_batch(
        self,
        queries: List[str],
        method: str = 'similarity',
        similarity_threshold: float = 0.5,
        k: int = 4
    ) -> List[Document]:
        """
        批量执行检索查询并对结果去重。

        参数:
            queries: 查询字符串列表。
            method: 检索方法，可选值：'similarity'、'mmr'、'threshold'。
            similarity_threshold: 相似度阈值（仅在 method='threshold' 时使用）。
            k: 每次检索返回的最大结果数（仅在 method='similarity' 或 'mmr' 时使用）。

        返回:
            去重后的 Document 对象列表。
        """
        if not self._vector_store:
            raise ValueError("Chroma 向量存储尚未初始化。请先调用 create_from_documents 或 add_documents。")

        if not queries:
            return []

        # 定义检索方法映射
        method_map = {
            'similarity': lambda q: self.search_similarity(q, k=k),
            'mmr': lambda q: self.search_mmr(q, k=k),
            'threshold': lambda q: self.search_with_threshold(q, similarity_threshold=similarity_threshold)
        }

        if method not in method_map:
            print(f"警告: 未知的检索方法 '{method}'，使用 'similarity'。")
            method = 'similarity'

        search_fn = method_map[method]
        results = []
        seen_texts = set()

        for query in queries:
            try:
                docs = search_fn(query)
                for doc in docs:
                    text = doc.page_content
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        results.append(doc)
            except Exception as e:
                print(f"警告: 执行查询 '{query}' 时出错: {e}")
                continue

        print(f"批量检索完成，共收集到 {len(results)} 条去重后的结果。")
        return results

    def __len__(self) -> int:
        """返回 Chroma 向量存储中的文档数量。如果未初始化，返回 0。"""
        if not self._vector_store or not hasattr(self._vector_store, '_collection'):
            return 0

        try:
            return self._vector_store._collection.count()
        except Exception as e:
            print(f"获取 Chroma 集合大小时出错: {e}")
            return 0

    def is_initialized(self) -> bool:
        """检查 Chroma 向量存储是否已初始化。"""
        return self._vector_store is not None

    @property
    def vector_store(self) -> Optional[Chroma]:
        """获取底层的 Chroma 向量存储实例。如果未初始化则返回 None。"""
        return self._vector_store

    def is_persistent(self) -> bool:
        """检查是否使用持久化模式。"""
        return self.persist_directory is not None

    def get_persist_directory(self) -> Optional[str]:
        """获取持久化目录路径。"""
        return self.persist_directory

    def get_db_info(self) -> Dict[str, Any]:
        """获取数据库信息。"""
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "is_persistent": self.is_persistent(),
            "is_initialized": self.is_initialized(),
            "document_count": len(self),
            "mode": "持久化模式" if self.is_persistent() else "内存模式"
        }