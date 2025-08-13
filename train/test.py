import json
import os
project_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(project_dir, "data/ft_val_corpus.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

for qid, question in data["queries"].items():
    print("问题:", question)
    doc_id = data["relevant_docs"][qid][0]

