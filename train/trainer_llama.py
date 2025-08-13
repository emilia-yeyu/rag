from llama_index.legacy.finetuning import SentenceTransformersFinetuneEngine
from llama_index.legacy.finetuning import EmbeddingQAFinetuneDataset
import os
def get_local_model_path(model_name: str) -> str:
    """
    获取本地模型路径，如果不存在则返回HuggingFace模型名称。
    与EmbeddingAdapter保持一致的逻辑。
    
    Args:
        model_name: 模型名称，例如 "BAAI/bge-large-zh-v1.5"
    
    Returns:
        本地模型路径或HuggingFace模型名称
    """
    # 将huggingface模型名称转换为本地缓存目录名称
    cache_name = model_name.replace("/", "--")
    cache_name = f"models--{cache_name}"
    
    # 获取项目根目录并构建模型路径
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 回到 RAG 目录
    model_base_path = os.path.join(script_dir, "models", "embeddings", cache_name)
    
    # 检查是否存在main引用文件
    main_ref_path = os.path.join(model_base_path, "refs", "main")
    if os.path.exists(main_ref_path):
        # 读取main引用指向的snapshot
        try:
            with open(main_ref_path, 'r') as f:
                snapshot_hash = f.read().strip()
            
            # 构建完整的snapshot路径
            snapshot_path = os.path.join(model_base_path, "snapshots", snapshot_hash)
            
            # 验证snapshot路径存在且包含必要的配置文件
            if os.path.exists(snapshot_path) and os.path.exists(os.path.join(snapshot_path, "config.json")):
                print(f"🎯 使用本地模型: {snapshot_path}")
                return snapshot_path
        except Exception as e:
            print(f"⚠️  读取本地模型引用失败: {e}")
    
    print(f"🌐 使用在线模型: {model_name}")
    return model_name

import json
project_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_CORPUS_FPATH = os.path.join(project_dir, "data/ft_train_corpus.json")
VAL_CORPUS_FPATH = os.path.join(project_dir, "data/ft_val_corpus.json")

# 加载JSON数据
train_data = json.load(open(TRAIN_CORPUS_FPATH, "r", encoding="utf-8"))
val_data = json.load(open(VAL_CORPUS_FPATH, "r", encoding="utf-8"))

# 转换为EmbeddingQAFinetuneDataset对象
print("📚 正在加载数据集...")
train_dataset = EmbeddingQAFinetuneDataset.from_json(TRAIN_CORPUS_FPATH)
val_dataset = EmbeddingQAFinetuneDataset.from_json(VAL_CORPUS_FPATH)

print(f"✅ 训练数据集加载成功，大小: {len(train_dataset.queries)} queries")
print(f"✅ 验证数据集加载成功，大小: {len(val_dataset.queries)} queries")

model_name = "BAAI/bge-large-zh-v1.5"  # 使用标准的HuggingFace模型名称
model_path = get_local_model_path(model_name)

eval_name = model_name.replace("/", "_")
output_dir_name = f"ft_{eval_name}"  # 使用简化的名称作为输出目录

print("🔧 创建微调引擎...")
finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id=get_local_model_path(model_name),
    model_output_path=output_dir_name,
    val_dataset=val_dataset,



    # 添加调试参数

)

finetune_engine.finetune()