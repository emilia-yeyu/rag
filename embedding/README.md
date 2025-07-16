# Embedding Adapter 简明接口说明

本目录用于统一管理和适配不同厂商的文本嵌入（Embedding）模型。

## 主要接口

通过 `EmbeddingAdapter.get_embedding(provider: str, model_name: Optional[str] = None)` 获取指定厂商和模型的嵌入实例。

- **provider**：嵌入模型提供商名称（如 `"openai"`、`"dashscope"`、`"google"`，不区分大小写）。
- **model_name**：可选，具体模型名称，不填则用默认。

**返回**：配置好的 LangChain `Embeddings` 实例，可直接用于 `.embed_query()` 等方法。

## 支持的提供商与默认模型

- `openai`：默认 `text-embedding-3-small`
- `dashscope`：默认 `text-embedding-v3`
- `google`：默认 `text-embedding-004`

## 环境变量

- OpenAI: `OPENAI_API_KEY`（可选 `OPENAI_BASE_URL`）
- DashScope: `DASHSCOPE_API_KEY`
- Google: `GOOGLE_API_KEY`

未设置密钥会报错。 