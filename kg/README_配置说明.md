# 知识图谱配置说明

## 🚨 问题解决

### 1. API限流问题 (429 Too Many Requests)

**问题**: DeepSeek API请求频率过高导致限流
**解决方案**:
- 降低并发数: `best_model_max_async=2, cheap_model_max_async=2`
- 添加随机延迟和重试机制
- 减少实体提取迭代次数: `entity_extract_max_gleaning=1`

### 2. Neo4j配置问题

**Neo4j数据库配置有两种方式**:

#### 方式1: 环境变量配置 (推荐)
创建 `.env` 文件：
```bash
# Neo4j配置
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# DeepSeek API配置
OPENAI_API_KEY=your_deepseek_api_key
```

#### 方式2: 代码中配置
```python
rag = GraphRAG(
    working_dir=WORKING_DIR,
    addon_params={
        "neo4j_url": "bolt://localhost:7687",
        "neo4j_auth": ("neo4j", "your_password")
    }
)
```

## 🛠️ 本地Neo4j安装

### Docker方式 (推荐)
```bash
docker run \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -v neo4j_data:/data \
    -v neo4j_logs:/logs \
    -v neo4j_import:/var/lib/neo4j/import \
    -v neo4j_plugins:/plugins \
    --env NEO4J_AUTH=neo4j/your_password \
    neo4j:latest
```

### 直接安装
1. 下载Neo4j: https://neo4j.com/download/
2. 解压并启动
3. 访问 http://localhost:7474
4. 默认用户名/密码: neo4j/neo4j

## 📊 配置参数说明

### 性能调优参数
```python
rag = GraphRAG(
    # 基本配置
    working_dir="./cache_dir",
    enable_llm_cache=True,
    
    # API限流控制
    best_model_max_async=2,          # 降低并发数
    cheap_model_max_async=2,
    
    # 实体提取控制
    entity_extract_max_gleaning=1,   # 减少迭代次数
    entity_summary_to_max_tokens=300, # 减少token数量
    
    # 分块设置
    chunk_token_size=800,            # 减少分块大小
    chunk_overlap_token_size=50,
    
    # 图聚类设置
    max_graph_cluster_size=8,        # 减少聚类大小
)
```

### 数据源配置
```python
# 使用本地文件
with open("your_text_file.txt", "r", encoding="utf-8") as f:
    text_content = f.read()

rag.insert(text_content)
```

## 🔧 故障排除

### 1. 内存不足
```python
# 减少并发和批处理大小
rag = GraphRAG(
    best_model_max_async=1,
    cheap_model_max_async=1,
    embedding_batch_num=16,
)
```

### 2. API配额不足
```python
# 使用缓存减少API调用
rag = GraphRAG(
    enable_llm_cache=True,  # 启用缓存
    entity_extract_max_gleaning=1,  # 减少API调用
)
```

### 3. 网络连接问题
- 检查Neo4j服务是否运行
- 确认防火墙设置
- 验证网络连接

## 📝 使用示例

### 完整配置示例
```python
import os
from nano_graphrag import GraphRAG, QueryParam

# 配置检查
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    print("请设置NEO4J_PASSWORD环境变量")
    exit(1)

# 创建GraphRAG实例
rag = GraphRAG(
    working_dir="./my_cache",
    enable_llm_cache=True,
    best_model_max_async=2,
    cheap_model_max_async=2,
    entity_extract_max_gleaning=1,
    addon_params={
        "neo4j_url": NEO4J_URI,
        "neo4j_auth": (NEO4J_USERNAME, NEO4J_PASSWORD)
    }
)

# 插入数据
with open("data.txt", "r") as f:
    rag.insert(f.read())

# 查询
result = rag.query("你的问题", param=QueryParam(mode="global"))
print(result)
```

## 🚀 性能优化建议

1. **使用SSD存储** - 提高I/O性能
2. **增加内存** - 减少磁盘交换
3. **使用GPU** - 加速embedding计算
4. **启用缓存** - 减少重复计算
5. **调整批处理大小** - 平衡内存和性能

## 📞 支持

如果遇到问题，请检查：
1. 环境变量是否正确设置
2. Neo4j服务是否正常运行
3. API密钥是否有效
4. 网络连接是否正常
