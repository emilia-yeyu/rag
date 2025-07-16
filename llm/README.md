# LLM Adapter 简明接口说明

本目录用于统一管理和适配不同厂商的大语言模型（LLM），便于在项目中灵活切换和调用。

## 主要接口

### 获取 LLM 实例

通过 `LLMAdapter.get_llm(provider: str, model_name: Optional[str] = None, **kwargs)` 获取指定厂商和模型的 LLM 实例。

- **provider**：模型提供商名称（如 `"openai"`、`"dashscope"`、`"google"` 等，支持的厂商详见 adapter.py 代码）。
- **model_name**：可选，具体模型名称，不填则用默认。
- **kwargs**：可选，传递给底层 LLM 构造的其他参数（如温度、最大长度等）。

**返回**：配置好的 LangChain LLM 实例，可直接用于 `.invoke()`、`.generate()` 等方法。

## 用法演示

```python
from backend.llm.adapter import LLMAdapter

# 获取 OpenAI LLM 实例
llm = LLMAdapter.get_llm(provider="openai", model_name="gpt-3.5-turbo")

# 直接调用生成
response = llm.invoke("请用一句话介绍金田村。")
print(response)
```

## 环境变量

不同厂商需要设置相应的 API Key 环境变量，例如：

- OpenAI: `OPENAI_API_KEY`
- DashScope: `DASHSCOPE_API_KEY`
- Google: `GOOGLE_API_KEY`

未设置密钥会报错。

---

如需更多高级用法（如自定义参数、流式输出等），请参考 `adapter.py` 代码注释或实际接口文档。 