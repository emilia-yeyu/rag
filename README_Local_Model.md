# 本地Qwen模型使用指南

本指南将帮助你下载并配置本地Qwen2.5-1.5B-Instruct模型，实现离线RAG系统。

## 🌟 优势

- **离线运行**: 无需API密钥，完全本地化
- **数据隐私**: 数据不会发送到外部服务器
- **成本控制**: 一次下载，永久使用
- **微调友好**: 可以基于本地模型进行微调
- **响应稳定**: 不受网络波动影响

## 📋 系统要求

### 最低配置
- **内存**: 8GB RAM
- **存储**: 5GB 可用空间
- **Python**: 3.8+
- **PyTorch**: 2.0+

### 推荐配置
- **内存**: 16GB+ RAM
- **GPU**: NVIDIA GPU with 4GB+ VRAM (可选)
- **存储**: 10GB+ 可用空间

### GPU支持 (可选)
如果有NVIDIA GPU，模型将自动使用GPU加速：
```bash
# 安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# 额外的模型相关依赖
pip install torch>=2.0.0 transformers>=4.32.0 accelerate>=0.21.0
```

### 2. 下载模型

```bash
# 运行下载脚本
python download_model.py
```

下载过程：
- 📦 模型大小: ~3GB
- ⏱️ 预计时间: 5-15分钟
- 📁 保存位置: `./models/qwen2.5-1.5b-instruct/`

### 3. 测试模型

```bash
# 运行测试脚本
python test_local_model.py
```

### 4. 启动RAG系统

```bash
# 使用本地模型启动RAG
python rag.py
```

## 🔧 配置说明

### 模型配置参数

在 `llm/local_model.py` 中可以调整以下参数：

```python
class LocalQwenModel(BaseChatModel):
    temperature: float = 0.7      # 控制随机性 (0-1)
    max_new_tokens: int = 512     # 最大生成token数
    do_sample: bool = True        # 是否采样
    top_p: float = 0.8           # 核采样参数
    top_k: int = 50              # top-k采样参数
    repetition_penalty: float = 1.1  # 重复惩罚
```

### RAG系统配置

在 `rag.py` 中修改LLM配置：

```python
# 基本配置
self.llm = LLMAdapter.get_llm(
    "local", 
    "./models/qwen2.5-1.5b-instruct", 
    temperature=0.1
)

# 高级配置
self.llm = LLMAdapter.get_llm(
    "local", 
    "./models/qwen2.5-1.5b-instruct", 
    temperature=0.7,
    max_new_tokens=256,
    top_p=0.9,
    repetition_penalty=1.05
)
```

## 📊 性能优化

### GPU加速

如果有NVIDIA GPU，模型会自动使用GPU：

```python
# 检查GPU状态
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前GPU: {torch.cuda.get_device_name()}")
```

### 内存优化

对于内存受限的环境：

```python
# 在local_model.py中修改
self._model = AutoModelForCausalLM.from_pretrained(
    self.model_path,
    torch_dtype=torch.float16,     # 使用半精度
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,        # 低CPU内存使用
    load_in_8bit=True             # 8bit量化 (需要安装bitsandbytes)
)
```

### 推理加速

```python
# 减少生成长度
max_new_tokens=128

# 禁用采样(更快但可能重复)
do_sample=False

# 使用贪心解码
temperature=0.0
```

## 🛠️ 故障排除

### 常见问题

1. **内存不足**
   ```
   解决方案：
   - 关闭其他应用程序
   - 使用8bit量化
   - 减小max_new_tokens
   ```

2. **模型下载失败**
   ```bash
   # 设置镜像源
   export HF_ENDPOINT=https://hf-mirror.com
   
   # 重新下载
   python download_model.py
   ```

3. **CUDA版本不匹配**
   ```bash
   # 重新安装对应版本的PyTorch
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **生成质量不佳**
   ```python
   # 调整参数
   temperature=0.7        # 增加随机性
   top_p=0.9             # 增加多样性
   repetition_penalty=1.1 # 减少重复
   ```

### 调试日志

启用详细日志：

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🔄 模型微调

### 准备微调环境

```bash
# 安装微调相关依赖
pip install peft datasets trl
```

### 微调脚本示例

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

# 加载模型
model_path = "./models/qwen2.5-1.5b-instruct"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 配置LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

# 应用LoRA
model = get_peft_model(model, lora_config)

# 微调训练...
```

## 📈 性能基准

### 硬件配置对比

| 配置 | 加载时间 | 推理速度 | 内存使用 |
|------|---------|---------|---------|
| CPU only (16GB RAM) | ~30秒 | ~2-3秒/回答 | 4-6GB |
| RTX 3060 (12GB) | ~15秒 | ~0.5-1秒/回答 | 3-4GB |
| RTX 4080 (16GB) | ~10秒 | ~0.3-0.5秒/回答 | 3-4GB |

### 质量评估

- **事实准确性**: ⭐⭐⭐⭐ (基于RAG检索)
- **语言流畅性**: ⭐⭐⭐⭐
- **上下文理解**: ⭐⭐⭐⭐
- **中文支持**: ⭐⭐⭐⭐⭐

## 🔍 使用示例

### 基本查询

```python
from rag import SimpleRAG

# 初始化
rag = SimpleRAG("2.txt")

# 查询
result = rag.query("贾宝玉的性格特点是什么？")
print(result['answer'].content)
```

### 批量处理

```python
questions = [
    "红楼梦的主要人物有哪些？",
    "大观园是什么地方？",
    "林黛玉和薛宝钗的关系如何？"
]

for question in questions:
    result = rag.query(question)
    print(f"Q: {question}")
    print(f"A: {result['answer'].content}\n")
```

### 配置定制

```python
# 创建自定义配置的RAG
rag = SimpleRAG("2.txt")

# 修改LLM参数
rag.llm = LLMAdapter.get_llm(
    "local", 
    "./models/qwen2.5-1.5b-instruct",
    temperature=0.8,           # 更有创造性
    max_new_tokens=200,        # 更长回答
    repetition_penalty=1.15    # 更少重复
)
```

## 📚 扩展使用

### 1. 多模型支持

可以同时支持多个本地模型：

```python
# 配置不同模型用于不同任务
creative_llm = LLMAdapter.get_llm("local", "./models/qwen2.5-1.5b-instruct", temperature=0.9)
factual_llm = LLMAdapter.get_llm("local", "./models/qwen2.5-1.5b-instruct", temperature=0.1)
```

### 2. 对话历史

添加对话历史支持：

```python
from langchain_core.messages import HumanMessage, AIMessage

# 维护对话历史
conversation_history = []

def chat_with_history(question):
    conversation_history.append(HumanMessage(content=question))
    response = rag.llm._generate(conversation_history)
    answer = response.generations[0].message.content
    conversation_history.append(AIMessage(content=answer))
    return answer
```

### 3. 流式输出

实现流式回答：

```python
# 在local_model.py中添加流式支持
def _stream_generate(self, messages, **kwargs):
    # 实现流式生成逻辑
    pass
```

## 🎯 最佳实践

1. **首次使用**：运行测试脚本确保一切正常
2. **性能监控**：定期检查内存和GPU使用情况
3. **模型更新**：关注Qwen模型的新版本
4. **数据安全**：本地模型确保数据隐私
5. **备份模型**：下载完成后备份模型文件

## 📝 更新日志

### v1.0.0
- ✨ 初始版本
- ✨ 支持Qwen2.5-1.5B-Instruct
- ✨ GPU自动检测和加速
- ✨ 完整的RAG集成

## 🤝 贡献

欢迎提交Issue和PR来改进本地模型支持！

## ⚠️ 注意事项

1. **模型许可**: 请遵守Qwen模型的使用许可
2. **商业使用**: 商业使用前请查看模型的商业许可条款
3. **内容过滤**: 本地模型可能需要额外的内容安全过滤
4. **版本兼容**: 不同版本的transformers可能有兼容性问题