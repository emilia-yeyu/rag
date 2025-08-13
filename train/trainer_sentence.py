# -*- coding: utf-8 -*-
# @file: ft_sentence_transformers_trainer.py
import os
import json
import time
import torch
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import SentenceTransformerTrainer

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

start_time = time.time()
project_dir = os.path.dirname(os.path.abspath(__file__))

# load eval dataset
with open(os.path.join(project_dir, "data/ft_val_corpus_clean.json"), "r", encoding="utf-8") as f:
    eval_content = json.loads(f.read())

corpus, queries, relevant_docs = eval_content['corpus'], eval_content['queries'], eval_content['relevant_docs']
# load train dataset
with open(os.path.join(project_dir, "data/ft_train_corpus_clean.json"), "r", encoding="utf-8") as f:
    train_content = json.loads(f.read())

train_anchor, train_positive = [], []
for query_id, context_id in train_content['relevant_docs'].items():
    train_anchor.append(train_content['queries'][query_id])
    train_positive.append(train_content['corpus'][context_id[0]])

train_dataset = Dataset.from_dict({"anchor": train_anchor, "positive": train_positive})

print(train_dataset)
print(train_dataset[0:5])

# Load a model - 使用统一的模型路径管理
model_name = "BAAI/bge-large-zh-v1.5"  # 使用标准的HuggingFace模型名称
model_path = get_local_model_path(model_name)

model = SentenceTransformer(model_path, device="cuda:0" if torch.cuda.is_available() else "cpu")
print("✅ Model loaded successfully")

# # Evaluate the model
# 为评估器创建一个简短的名称
eval_name = model_name.replace("/", "_")
evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name=eval_name,
    score_functions={"cosine": cos_sim}
)
train_loss = MultipleNegativesRankingLoss(model)

# define training arguments
output_dir_name = f"ft_{eval_name}"  # 使用简化的名称作为输出目录
args = SentenceTransformerTrainingArguments(
    output_dir=output_dir_name,  # output directory and hugging face model ID
    num_train_epochs=5,  # number of epochs
    per_device_train_batch_size=2,  # train batch size
    gradient_accumulation_steps=2,  # for a global batch size of 512
    per_device_eval_batch_size=4,  # evaluation batch size
    warmup_ratio=0.1,  # warmup ratio
    learning_rate=2e-5,  # learning rate, 2e-5 is a good value
    lr_scheduler_type="cosine",  # use constant learning rate scheduler
    optim="adamw_torch_fused",  # use fused adamw optimizer
    tf32=True,  # use tf32 precision
    bf16=True,  # use bf16 precision
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="epoch",  # evaluate after each epoch
    save_strategy="epoch",  # save after each epoch
    logging_steps=10,  # log every 10 steps
    save_total_limit=3,  # save only the last 3 models
    load_best_model_at_end=True,  # load the best model when training ends
    metric_for_best_model=f"eval_{eval_name}_cosine_ndcg@10",  # Optimizing for the best ndcg@10 score
    dataloader_num_workers=0,
)

# train the model
trainer = SentenceTransformerTrainer(
    model=model,    # the model to train
    args=args,      # training arguments
    train_dataset=train_dataset.select_columns(
        ["anchor", "positive"]
    ),  # training dataset
    loss=train_loss,
    evaluator=evaluator
)

trainer.train()
trainer.save_model()
print(f"cost time: {time.time() - start_time:.2f}s")