import json
import os
project_dir = os.path.dirname(os.path.abspath(__file__))

def clean_qa_dataset(input_path, output_path):
    """
    清洗 LlamaIndex 生成的 QA 数据集。
    删除无效的 query 以及它们关联的所有数据。
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_queries = data.get("queries", {})
    original_corpus = data.get("corpus", {})
    original_relevant_docs = data.get("relevant_docs", {})

    # 1. 识别无效的 query_id
    invalid_query_ids = set()
    invalid_keywords = ["题目"]
    for q_id, question in original_queries.items():
        if len(question.strip()) < 5 or not (question.strip().endswith("？") or question.strip().endswith("?")) or any(keyword in question for keyword in invalid_keywords):
            invalid_query_ids.add(q_id)
            
    print(f"发现了 {len(invalid_query_ids)} 个无效的 query。")

    # 2. 创建干净的 queries 和 relevant_docs
    cleaned_queries = {}
    cleaned_relevant_docs = {}
    for q_id, question in original_queries.items():
        if q_id not in invalid_query_ids:
            cleaned_queries[q_id] = question
            # 确保这个 query 在 relevant_docs 中有记录
            if q_id in original_relevant_docs:
                cleaned_relevant_docs[q_id] = original_relevant_docs[q_id]

    # 3. 识别需要保留的 chunk_id
    required_chunk_ids = set()
    for doc_ids in cleaned_relevant_docs.values():
        for doc_id in doc_ids:
            required_chunk_ids.add(doc_id)

    # 4. 创建干净的 corpus
    cleaned_corpus = {}
    for chunk_id, text in original_corpus.items():
        if chunk_id in required_chunk_ids:
            cleaned_corpus[chunk_id] = text

    # 构建最终的干净数据集
    cleaned_data = {
        "queries": cleaned_queries,
        "corpus": cleaned_corpus,
        "relevant_docs": cleaned_relevant_docs,
        "mode": data.get("mode", "text") # 保留其他元数据
    }

    # 保存到新文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

    print(f"清洗完成！结果已保存到 {output_path}")
    print(f"原始 Query 数量: {len(original_queries)}, 清洗后: {len(cleaned_queries)}")
    print(f"原始 Corpus 数量: {len(original_corpus)}, 清洗后: {len(cleaned_corpus)}")


# --- 使用方法 ---
# 假设您的文件名为 "ft_train_corpus.json"
input_file = os.path.join(project_dir, "data/ft_train_corpus.json")
output_file = os.path.join(project_dir, "data/ft_train_corpus_clean.json")
clean_qa_dataset(input_file, output_file)

input_file = os.path.join(project_dir, "data/ft_val_corpus.json")
output_file = os.path.join(project_dir, "data/ft_val_corpus_clean.json")
clean_qa_dataset(input_file, output_file)
