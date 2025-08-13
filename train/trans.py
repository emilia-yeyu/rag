import json
import json
import os
project_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(project_dir, "data/ft_train_corpus.json"), "r", encoding="utf-8") as f:
    content = json.load(f)


# 保存新的文件（中文）
with open(os.path.join(project_dir, "data/ft_train_corpus.json"), "w", encoding="utf-8") as f:
    json.dump(content, f, ensure_ascii=False, indent=2)

with open(os.path.join(project_dir, "data/ft_val_corpus.json"), "r", encoding="utf-8") as f:
    content = json.load(f)


# 保存新的文件（中文）
with open(os.path.join(project_dir, "data/ft_val_corpus.json"), "w", encoding="utf-8") as f:
    json.dump(content, f, ensure_ascii=False, indent=2)

print("✅ 转换完成，已写入 output.json")
