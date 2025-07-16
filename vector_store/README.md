# VectorStoreManager 简明接口说明

本目录用于管理和检索内存中的 Chroma 向量数据库。

## 主要接口

通过 `VectorStoreManager` 类实现文档的向量化存储与多种检索。

### 初始化

```python
VectorStoreManager(
    embedding_model: Embeddings,
    collection_name: str = "default_collection"
)
```
- **embedding_model**：LangChain Embeddings 实例。
- **collection_name**：集合名称（可选）。

### 核心方法

- `create_from_documents(documents: List[Document], chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None)`：用文档列表新建向量库。如果提供 `chunk_size`，则在创建前对文档进行切片。
- `add_documents(documents: List[Document], chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None)`：向现有库增量添加文档。如果提供 `chunk_size`，则在添加前对文档进行切片。
- `get_all_documents() -> List[Document]`：获取全部文档。
- `search_similarity(query: str, k: int = 4) -> List[Document]`：相似度检索。
- `search_mmr(query: str, k: int = 4) -> List[Document]`：MMR多样性检索。
- `search_with_threshold(query: str, similarity_threshold: float = 0.5) -> List[Document]`：使用相似度阈值检索，自动返回所有相似度分数超过指定阈值的文档，无需指定返回数量。

**返回**：均为 LangChain `Document` 对象列表。 