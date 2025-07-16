# LocalDocumentProcessor 简明接口说明

本目录用于本地文档的加载处理。

## 主要接口

通过 `LocalDocumentProcessor` 类实现本地文档的批量加载。

### 初始化
```python
LocalDocumentProcessor(
    docs_path: str,
    loader_kwargs: dict = None,
)
```
- **docs_path**：文档目录路径。
- **loader_kwargs**: (可选) 传递给 DirectoryLoader 的额外参数。

### 核心方法

- `load_documents() -> List[Document]`：加载目录下所有支持的文档，返回原始 Document 列表。

**返回**：为 LangChain `Document` 对象列表，可用于后续处理（如切片、向量化等）。 