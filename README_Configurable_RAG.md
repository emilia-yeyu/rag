# 可配置RAG系统

这是一个支持多种检索策略灵活组合的RAG系统，可以根据不同的使用场景选择最适合的检索模式。

## 🌟 特性

- **多种检索策略**: 支持向量检索、BM25关键词检索、SQL精确检索
- **灵活配置**: 7种预定义配置，支持单一策略或混合策略
- **动态切换**: 运行时可以切换不同的检索配置
- **性能优化**: RRF融合算法，智能权重分配
- **结构化数据**: 内置红楼梦人物数据库，支持精确查询

## 📋 检索模式

### 1. 单一检索模式

- **semantic** (仅向量语义检索)
  - 适用于: 概念性问题、语义相似性查询
  - 示例: "红楼梦的主题思想是什么？"

- **keyword** (仅BM25关键词检索)  
  - 适用于: 关键词匹配、精确短语查询
  - 示例: "大观园中的诗词"

- **structured** (仅SQL精确检索)
  - 适用于: 结构化信息查询、数据统计
  - 示例: "贾宝玉的年龄是多少？"

### 2. 混合检索模式

- **semantic_keyword** (向量检索 + BM25检索)
  - 权重: 向量70% + BM25 30%
  - 适用于: 需要语义理解和关键词匹配的查询

- **semantic_structured** (向量检索 + SQL检索)
  - 权重: 向量60% + SQL 40%
  - 适用于: 需要语义理解和精确数据的查询

- **keyword_structured** (BM25检索 + SQL检索)
  - 权重: BM25 50% + SQL 50%
  - 适用于: 关键词匹配和数据查询结合

- **comprehensive** (全混合检索)
  - 权重: 向量50% + BM25 30% + SQL 20%
  - 适用于: 复杂查询，需要多种检索策略

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 基本使用

```python
from configurable_rag import ConfigurableRAG

# 初始化RAG系统（使用综合模式）
rag = ConfigurableRAG("2.txt", retrieval_config="comprehensive")

# 执行查询
result = rag.query("贾宝玉多大年纪？", show_sources=True)
print(result['answer'].content)
```

### 3. 使用不同配置

```python
# 使用仅向量检索
rag_semantic = ConfigurableRAG("2.txt", retrieval_config="semantic")

# 使用仅SQL检索
rag_sql = ConfigurableRAG("2.txt", retrieval_config="structured")

# 使用向量+BM25混合
rag_hybrid = ConfigurableRAG("2.txt", retrieval_config="semantic_keyword")
```

### 4. 动态切换配置

```python
# 初始化
rag = ConfigurableRAG("2.txt", retrieval_config="comprehensive")

# 切换到语义检索模式
rag.switch_config("semantic")

# 切换到SQL检索模式  
rag.switch_config("structured")
```

## 💬 交互模式

```bash
# 使用默认配置启动
python configurable_rag.py

# 使用特定配置启动
python configurable_rag.py semantic
```

### 交互命令:

- 直接输入问题进行查询
- `config` - 查看当前配置
- `list` - 列出所有可用配置
- `switch <配置名>` - 切换配置
- `quit` - 退出

## 🧪 测试

```bash
# 运行完整测试套件
python test_configurable_rag.py
```

测试包括:
- SQL组件测试
- 检索配置测试  
- 不同检索模式测试
- 配置切换测试
- 性能对比测试

## 📊 配置对比

| 配置名称 | 检索方式 | 适用场景 | 优势 |
|---------|---------|---------|------|
| semantic | 向量检索 | 概念理解、语义查询 | 理解能力强 |
| keyword | BM25检索 | 关键词匹配 | 精确匹配 |
| structured | SQL检索 | 结构化数据查询 | 数据准确 |
| semantic_keyword | 向量+BM25 | 平衡语义和关键词 | 覆盖面广 |
| semantic_structured | 向量+SQL | 语义理解+精确数据 | 智能且准确 |
| keyword_structured | BM25+SQL | 关键词+结构化数据 | 高效精确 |
| comprehensive | 全混合 | 复杂综合查询 | 最全面 |

## 🗄️ 数据库结构

系统内置红楼梦人物数据库:

```sql
CREATE TABLE guests (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,      -- 姓名
    phone TEXT,              -- 电话
    position TEXT,           -- 职位  
    age INTEGER,             -- 年龄
    company TEXT,            -- 公司/家族
    email TEXT,              -- 邮箱
    address TEXT,            -- 地址
    notes TEXT,              -- 备注
    created_date TEXT        -- 创建时间
);
```

示例数据包含贾宝玉、林黛玉、薛宝钗等15个主要人物的详细信息。

## 📈 性能优化

### RRF融合算法

使用倒数排名融合(Reciprocal Rank Fusion)算法融合多个检索结果:

```
RRF_score = Σ(weight_i / (k + rank_i + 1))
```

### 权重优化

- 向量检索: 擅长语义理解
- BM25检索: 擅长关键词匹配  
- SQL检索: 擅长精确数据查询

权重根据查询类型和配置自动调整。

## 🔧 自定义配置

```python
from retrieval_config import RetrievalConfig, RetrievalMode

# 创建自定义配置
custom_config = RetrievalConfig(
    mode=RetrievalMode.VECTOR_SQL,
    k=10,
    vector_weight=0.7,
    sql_weight=0.3,
    vector_k=15,
    sql_k=8
)

# 使用自定义配置
rag = ConfigurableRAG("2.txt", retrieval_config=custom_config)
```

## 🎯 使用建议

### 选择检索模式的建议:

1. **纯信息查询** → `structured` (SQL检索)
   - "贾宝玉多大？"
   - "谁的电话是138xxx？"

2. **概念理解** → `semantic` (向量检索)  
   - "红楼梦的主题是什么？"
   - "林黛玉的性格特点？"

3. **关键词搜索** → `keyword` (BM25检索)
   - "大观园的诗词"
   - "荣国府管家"

4. **综合查询** → `comprehensive` (全混合)
   - "介绍一下王熙凤这个人物"
   - "贾府的主要人物关系"

## 🐛 故障排除

### 常见问题:

1. **SQL检索无结果**: 检查查询是否包含人物信息相关关键词
2. **向量检索慢**: 首次运行需要下载BGE模型
3. **BM25检索无结果**: 调整`bm25_min_match_ratio`参数
4. **数据库连接失败**: 检查SQLite数据库文件权限

### 日志分析:

系统会输出详细的检索过程日志，包括:
- 各检索器的结果数量
- RRF融合权重
- 最终结果来源分布
- 性能统计信息

## 📝 更新日志

### v1.0.0 (2024-01-XX)
- ✨ 新增可配置检索系统
- ✨ 支持7种预定义检索配置
- ✨ 集成SQL精确检索功能
- ✨ 实现RRF融合算法
- ✨ 支持动态配置切换
- ✨ 添加性能对比测试

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request来帮助改进这个项目！ 