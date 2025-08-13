# 知识图谱集成方案

## 🎯 集成目标
将nano_graphrag知识图谱系统与现有RAG系统集成，提供统一的接口和配置管理。

## 📊 问题分析

### 1. 目录结构问题
- **当前**: kg/独立目录，与RAG系统分离
- **问题**: 缺少统一的项目结构
- **解决**: 重构目录结构，创建统一的入口

### 2. 配置管理问题  
- **当前**: setup.py中硬编码配置
- **问题**: API密钥混乱，路径硬编码
- **解决**: 创建统一的配置管理系统

### 3. 接口不统一
- **当前**: 独立的GraphRAG类和查询方法
- **问题**: 与现有RAG系统接口不兼容
- **解决**: 创建适配器模式的统一接口

### 4. 依赖管理
- **当前**: 使用DeepSeek特定配置
- **问题**: 与现有模型配置不兼容
- **解决**: 支持多种模型后端

## 🔧 具体修改方案

### 步骤1: 重构配置系统
```python
# 创建 kg_config.py
class KnowledgeGraphConfig:
    def __init__(self):
        self.working_dir = os.path.join("RAG", "kg", "cache")
        self.model_config = self.load_model_config()
        self.api_config = self.load_api_config()
```

### 步骤2: 创建统一接口
```python
# 创建 kg_adapter.py  
class KnowledgeGraphAdapter:
    def __init__(self, config):
        self.config = config
        self.graph_rag = None
        
    def query(self, question: str, mode: str = "local"):
        # 统一查询接口
        pass
        
    def insert_documents(self, documents: List[str]):
        # 统一文档插入接口
        pass
```

### 步骤3: 集成到现有RAG系统
```python
# 修改 configurable_rag.py
class ConfigurableRAG:
    def __init__(self):
        # 添加知识图谱支持
        self.kg_adapter = KnowledgeGraphAdapter()
        
    def query_with_kg(self, question: str):
        # 结合传统RAG和知识图谱
        pass
```

### 步骤4: 统一数据管理
- 将kg缓存目录移动到统一的数据目录
- 支持现有的文档加载器
- 集成到现有的向量存储系统

## 📁 建议的新目录结构

```
RAG/
├── kg/
│   ├── __init__.py
│   ├── kg_config.py          # 知识图谱配置
│   ├── kg_adapter.py         # 适配器接口
│   ├── kg_manager.py         # 知识图谱管理器
│   ├── nano_graphrag/        # 原始库（保持不变）
│   └── cache/               # 缓存目录
├── unified_rag.py           # 统一的RAG入口
├── config/
│   ├── kg_config.json       # 知识图谱配置文件
│   └── model_config.json    # 模型配置文件
└── examples/
    └── kg_integration_demo.py
```

## 🚀 实施优先级

1. **高优先级**: 配置系统重构
2. **高优先级**: 创建适配器接口  
3. **中优先级**: 集成到现有RAG系统
4. **低优先级**: 优化和性能调优

## 💡 关键修改点

### 1. setup.py → kg_config.py
- 移除硬编码配置
- 支持环境变量配置
- 支持多种模型后端

### 2. 创建适配器层
- 统一查询接口
- 统一文档管理
- 错误处理和日志

### 3. 集成现有系统
- 与configurable_rag.py集成
- 支持现有的文档加载器
- 兼容现有的向量存储

### 4. 配置文件化
- JSON配置文件
- 环境变量支持
- 开发/生产环境分离

## 🧪 测试计划

1. **单元测试**: 各个组件独立测试
2. **集成测试**: RAG+KG联合查询
3. **性能测试**: 查询速度和内存使用
4. **兼容性测试**: 与现有系统的兼容性
