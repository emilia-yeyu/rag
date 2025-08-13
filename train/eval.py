# -*- coding: utf-8 -*-
# @file: bge_base_zh_eval.py
import os
import json
import time
import torch
from pprint import pprint
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import cos_sim

project_dir = os.path.dirname(os.path.abspath(__file__))

# data process
# load dataset, get corpus, queries, relevant_docs
with open(os.path.join(project_dir, "data/ft_val_corpus_clean.json"), "r", encoding="utf-8") as f:
    content = json.loads(f.read())

corpus = content['corpus']
queries = content['queries']
relevant_docs = content['relevant_docs']

# # Load a model
# 使用完整的snapshot路径或直接使用huggingface model id
model_name = "models--BAAI--bge-large-zh-v1.5"
# 指向包含完整模型文件的snapshot目录
model_path = os.path.join(project_dir, f"../models/embeddings/{model_name}/snapshots/79e7739b6ab944e86d6171e44d24c997fc1e0116")

# 如果本地模型路径不存在，直接使用huggingface model id
if not os.path.exists(model_path):
    model_path = "BAAI/bge-large-zh-v1.5"
    print(f"本地模型路径不存在，使用huggingface model id: {model_path}")

model = SentenceTransformer(model_path, device="cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded")
s_time = time.time()

# # Evaluate the model
evaluator = InformationRetrievalEvaluator(
    queries=queries,
    corpus=corpus,
    relevant_docs=relevant_docs,
    name=f"{os.path.basename(model_path)}",
    score_functions={"cosine": cos_sim}
)

# Evaluate the model
result = evaluator(model)
pprint(result)
print(f"Time cost: {time.time() - s_time:.2f}s")