# -*- coding: utf-8 -*-
# @file: estimate_resources.py
"""
这个脚本用于估算 sentence-transformers 训练任务所需的资源，
主要关注 GPU 显存（VRAM）和 CPU 内存（RAM）。
它不会进行完整的训练，而是模拟最耗费资源的几个步骤并进行测量。

如何使用：
1.  根据你的情况修改下面的【配置参数】部分。
2.  使用指定了单个 GPU 的命令来运行此脚本，例如：
    CUDA_VISIBLE_DEVICES=1 python estimate_resources.py
"""
import os
import json
import time
import torch
import psutil
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.losses import MultipleNegativesRankingLoss
from torch.optim import AdamW

# --- 1. 配置参数 (请根据您的需求修改) ---
MODEL_NAME = "BAAI/bge-large-zh-v1.5"
# 设置您打算用于实际训练的批次大小
BATCH_SIZE = 8  

project_dir = os.path.dirname(os.path.abspath(__file__))
TRAIN_FILE_PATH = os.path.join(project_dir, "data/ft_train_corpus_clean.json")
VAL_FILE_PATH = os.path.join(project_dir, "data/ft_val_corpus_clean.json")

# --- 辅助函数 ---
def format_bytes(size):
    """将字节转换为更易读的格式 (MB or GB)"""
    if size is None:
        return "N/A"
    power = 1024
    n = 0
    power_labels = {0: 'B', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size > power and n < len(power_labels) -1 :
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"

def get_process_memory():
    """获取当前 Python 进程的 CPU 内存使用情况"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss


# --- 2. 开始诊断 ---

print("="*50)
print("🚀 开始进行训练资源需求估算...")
print("="*50)

# 检查 GPU 是否可用
if not torch.cuda.is_available():
    print("❌ 错误：未检测到可用的 CUDA 设备。请在 GPU 环境下运行此脚本。")
    exit()

device = torch.device("cuda:0")
print(f"✅ 检测到 GPU: {torch.cuda.get_device_name(0)}")

# --- 步骤 1: 测量模型加载所需显存 ---
print("\n--- [诊断步骤 1/4] 测量模型加载资源 ---")
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)
mem_before_load = torch.cuda.memory_allocated(device)

model = SentenceTransformer(MODEL_NAME, device=device)

mem_after_load = torch.cuda.memory_allocated(device)
model_vram = mem_after_load - mem_before_load

print(f"🧠 模型基准显存占用: {format_bytes(model_vram)}")
print(f"   (这是模型权重本身加载到 GPU 上所需的最小显存)")

# --- 步骤 2: 测量数据加载所需 CPU 内存 ---
print("\n--- [诊断步骤 2/4] 测量数据加载资源 ---")
cpu_mem_before = get_process_memory()

with open(TRAIN_FILE_PATH, "r", encoding="utf-8") as f:
    train_content = json.loads(f.read())
with open(VAL_FILE_PATH, "r", encoding="utf-8") as f:
    eval_content = json.loads(f.read())

corpus, queries = eval_content['corpus'], eval_content['queries']
train_anchor, train_positive = [], []
for query_id, context_id in train_content['relevant_docs'].items():
    train_anchor.append(train_content['queries'][query_id])
    train_positive.append(train_content['corpus'][context_id[0]])
train_dataset = Dataset.from_dict({"anchor": train_anchor, "positive": train_positive})

cpu_mem_after = get_process_memory()
data_ram = cpu_mem_after - cpu_mem_before
print(f"📊 数据集加载到 CPU 内存增量: {format_bytes(data_ram)}")
print(f"   (您的训练和验证数据总共占用了大约这么多系统内存)")


# --- 步骤 3: 测量评估器预编码所需峰值显存 ---
print("\n--- [诊断步骤 3/4] 测量评估器语料库编码峰值资源 ---")
print("   (这是 trainer.train() 开始时卡住的那一步)")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)
start_time = time.time()

# 模拟 Evaluator 的操作：对所有 corpus 文档进行编码
print(f"   正在对验证集中的 {len(corpus)} 个文档进行编码...")
_ = model.encode(list(corpus.values()), batch_size=BATCH_SIZE, show_progress_bar=True)

end_time = time.time()
eval_peak_vram = torch.cuda.max_memory_allocated(device)
eval_time = end_time - start_time

print(f"📉 评估器预编码峰值显存: {format_bytes(eval_peak_vram)}")
print(f"⏱️  评估器预编码耗时: {eval_time:.2f} 秒")
print(f"   (这是运行评估前一次性发生的资源消耗)")


# --- 步骤 4: 测量单个训练步骤所需峰值显存 ---
print("\n--- [诊断步骤 4/4] 测量单个训练步骤峰值资源 ---")
print(f"   (使用您设置的 BATCH_SIZE = {BATCH_SIZE})")

loss_func = MultipleNegativesRankingLoss(model)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 从数据集中取一个批次
train_batch = train_dataset.select(range(BATCH_SIZE))
# SentenceTransformerTrainer 会自动处理 tokenizer 和格式化
# 这里我们直接使用文本输入，因为模型内部会处理
batch_features = [{"anchor": item["anchor"], "positive": item["positive"]} for item in train_batch]
# 使用模型的 tokenize 方法来模拟 trainer 的行为
tokenized_batch = model.tokenize(batch_features)
# 将 tokenized batch 移动到 GPU
for key in tokenized_batch:
    tokenized_batch[key] = tokenized_batch[key].to(device)


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(device)

# --- 模拟训练循环的核心 ---
# 1. 前向传播 (Forward Pass)
features = model(tokenized_batch)
# 2. 计算损失
loss = loss_func(features, labels=None) # MNRLoss 不需要显式 labels
# 3. 反向传播 (Backward Pass) - 这是最耗显存的一步！
loss.backward()
# 4. 清空梯度 (为下一步做准备)
optimizer.zero_grad()
# ---------------------------

train_step_peak_vram = torch.cuda.max_memory_allocated(device)
print(f"📈 单个训练步骤峰值显存: {format_bytes(train_step_peak_vram)}")
print(f"   (这是训练循环中每个 step 所需的最高显存，决定了是否会 OOM)")

# --- 3. 结论与建议 ---
print("\n" + "="*50)
print("✅ 诊断完成！结论如下：")
print("="*50)
print(f"  - 模型基准显存: {format_bytes(model_vram)}")
print(f"  - 训练峰值显存 (Batch Size={BATCH_SIZE}): {format_bytes(train_step_peak_vram)}")
print(f"  - 评估峰值显存: {format_bytes(eval_peak_vram)}")
print("-" * 50)
final_recommendation = max(train_step_peak_vram, eval_peak_vram)
print(f"👉 综合建议：")
print(f"   要以 BATCH_SIZE={BATCH_SIZE} 顺利运行完整的训练和评估，")
print(f"   您选择的 GPU 至少需要有 【{format_bytes(final_recommendation)}】 的可用显存。")
print(f"\n   如果 '训练峰值显存' 超出预算，请减小 BATCH_SIZE 或增加 gradient_accumulation_steps。")
print(f"   如果 '评估峰值显存' 超出预算，请减小评估时的 per_device_eval_batch_size。")