import os
import logging
import numpy as np
from openai import AsyncOpenAI
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag.base import BaseKVStorage
from nano_graphrag._utils import compute_args_hash, wrap_embedding_func_with_attrs
from nano_graphrag._storage import Neo4jStorage
from sentence_transformers import SentenceTransformer
import dotenv

dotenv.load_dotenv()

logging.basicConfig(level=logging.WARNING)
logging.getLogger("nano-graphrag").setLevel(logging.INFO)

DEEPSEEK_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "deepseek-ai/DeepSeek-V3"

# Neo4j配置检查
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

USE_NEO4J = bool(NEO4J_URI and NEO4J_USERNAME and NEO4J_PASSWORD and NEO4J_PASSWORD != "your_password")

if USE_NEO4J:
    print(f"✓ 使用Neo4j数据库: {NEO4J_URI}")
else:
    print("⚠ Neo4j配置不完整，将使用本地文件存储")
    print("如需使用Neo4j，请配置环境变量：NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD")


async def deepseepk_model_if_cache(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    import asyncio
    import random
    
    openai_async_client = AsyncOpenAI(
        api_key=DEEPSEEK_API_KEY, base_url="https://api-inference.modelscope.cn/v1"
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Get the cached response if having-------------------
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(MODEL, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None:
            return if_cache_return["return"]
    # -----------------------------------------------------

    # 添加随机延迟以减少API限流
    delay = random.uniform(0.5, 2.0)
    print(f"🔄 API调用延迟: {delay:.2f}秒")
    await asyncio.sleep(delay)
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"📡 发起API请求 (尝试 {attempt + 1}/{max_retries})")
            print(f"📝 提示词长度: {len(prompt)} 字符")
            
            response = await openai_async_client.chat.completions.create(
                model=MODEL, messages=messages, **kwargs
            )
            
            print(f"✅ API请求成功，响应长度: {len(response.choices[0].message.content)} 字符")
            
            # Cache the response if having-------------------
            if hashing_kv is not None:
                await hashing_kv.upsert(
                    {args_hash: {"return": response.choices[0].message.content, "model": MODEL}}
                )
            # -----------------------------------------------------
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"❌ API调用异常: {type(e).__name__}: {str(e)}")
            if "429" in str(e) or "Too Many Requests" in str(e):
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"⏳ API限流，等待 {wait_time:.2f} 秒后重试 (尝试 {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait_time)
                if attempt == max_retries - 1:
                    print("🚨 达到最大重试次数，返回默认响应")
                    return "无法获取响应，请稍后重试。"
            else:
                print(f"💥 API调用失败: {e}")
                return "API调用失败"


def remove_if_exist(file):
    if os.path.exists(file):
        os.remove(file)


WORKING_DIR = "./nano_graphrag_cache_deepseek_TEST"

# 尝试使用现有的embedding系统
try:
    import sys
    sys.path.append("../embedding")
    from adapter import EmbeddingAdapter
    
    print("🔄 尝试使用现有的BGE embedding系统...")
    bge_embedding = EmbeddingAdapter.get_embedding("bge", "BAAI/bge-large-zh-v1.5")
    
    @wrap_embedding_func_with_attrs(
        embedding_dim=1024,  # BGE-large-zh-v1.5的维度
        max_token_size=512,  # BGE模型的最大序列长度
    )
    async def local_embedding(texts: list[str]) -> np.ndarray:
        print(f"🔤 使用现有BGE系统计算 {len(texts)} 个文本的向量")
        embeddings = bge_embedding.embed_documents(texts)
        embeddings_array = np.array(embeddings)
        print(f"✅ BGE向量计算完成，形状: {embeddings_array.shape}")
        return embeddings_array
    
    print("✅ 现有BGE embedding系统加载成功")
    
except Exception as e:
    print(f"⚠ 无法使用现有embedding系统: {e}")
    print("🔄 使用简化的embedding函数...")
    
    @wrap_embedding_func_with_attrs(
        embedding_dim=1024,
        max_token_size=512,
    )
    async def local_embedding(texts: list[str]) -> np.ndarray:
        print(f"🔤 使用简化方法计算 {len(texts)} 个文本的向量")
        # 使用简单的哈希方法生成确定性向量
        import hashlib
        embeddings = []
        for text in texts:
            hash_obj = hashlib.md5(text.encode('utf-8'))
            seed = int.from_bytes(hash_obj.digest()[:4], 'big')
            np.random.seed(seed)
            embedding = np.random.normal(0, 1, 1024)
            embeddings.append(embedding)
        
        embeddings_array = np.array(embeddings)
        print(f"✅ 简化向量计算完成，形状: {embeddings_array.shape}")
        return embeddings_array


def query():
    rag = GraphRAG(
        working_dir=WORKING_DIR,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
        # 降低并发数以避免API限流
        best_model_max_async=2,
        cheap_model_max_async=2,
        # 使用本地BGE embedding
        embedding_func=local_embedding,
    )
    print(
        rag.query(
            "公司的组织架构如何?", param=QueryParam(mode="global")
        )
    )


def insert():
    from time import time

    with open("./tests/1.txt", encoding="utf-8-sig") as f:
        FAKE_TEXT = f.read()

    remove_if_exist(f"{WORKING_DIR}/vdb_entities.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_full_docs.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_text_chunks.json")
    remove_if_exist(f"{WORKING_DIR}/kv_store_community_reports.json")
    remove_if_exist(f"{WORKING_DIR}/graph_chunk_entity_relation.graphml")

    rag = GraphRAG(
        working_dir=WORKING_DIR,
        enable_llm_cache=True,
        best_model_func=deepseepk_model_if_cache,
        cheap_model_func=deepseepk_model_if_cache,
        # 降低并发数以避免API限流
        best_model_max_async=2,
        cheap_model_max_async=2,
        # 减少实体提取的迭代次数
        entity_extract_max_gleaning=1,
        # 使用本地BGE embedding
        embedding_func=local_embedding,
    )
    start = time()
    rag.insert(FAKE_TEXT)
    print("indexing time:", time() - start)
    # rag = GraphRAG(working_dir=WORKING_DIR, enable_llm_cache=True)
    # rag.insert(FAKE_TEXT[half_len:])


if __name__ == "__main__":
    #insert()
    query()