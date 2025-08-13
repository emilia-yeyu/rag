#!/usr/bin/env python3
"""
混合检索RAG系统
结合SQL数据库和向量检索，专门优化结构化信息查询
"""

import os
import sqlite3
import time
import json
import re
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime

# 导入现有RAG组件
from rag import SimpleRAG
from embedding.adapter import EmbeddingAdapter
from llm.adapter import LLMAdapter
from vector_store.vector_store import VectorStoreManager

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()


@dataclass
class StructuredInfo:
    """结构化信息数据类"""
    id: str
    content: str
    location: Optional[str] = None
    coordinates: Optional[Tuple[float, float]] = None  # (latitude, longitude)
    date: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class SQLStructuredStore:
    """SQL结构化数据存储"""
    
    def __init__(self, db_path: str = "./structured_data.db"):
        """
        初始化SQL存储
        
        Args:
            db_path: SQLite数据库路径
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # 允许通过列名访问
        self._init_schema()
        print(f"✅ SQL结构化存储初始化完成: {db_path}")
    
    def _init_schema(self):
        """初始化数据库schema"""
        cursor = self.conn.cursor()
        
        # 创建主表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS structured_info (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                location TEXT,
                latitude REAL,
                longitude REAL,
                date TEXT,
                category TEXT,
                tags TEXT,  -- JSON格式存储
                metadata TEXT,  -- JSON格式存储
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建索引提高查询效率
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_location ON structured_info(location)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_coordinates ON structured_info(latitude, longitude)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON structured_info(date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON structured_info(category)")
        
        self.conn.commit()
        print("✅ 数据库schema初始化完成")
    
    def insert(self, info: StructuredInfo):
        """插入结构化信息"""
        cursor = self.conn.cursor()
        
        # 处理坐标
        lat, lon = (info.coordinates[0], info.coordinates[1]) if info.coordinates else (None, None)
        
        cursor.execute("""
            INSERT OR REPLACE INTO structured_info 
            (id, content, location, latitude, longitude, date, category, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            info.id,
            info.content,
            info.location,
            lat,
            lon,
            info.date,
            info.category,
            json.dumps(info.tags) if info.tags else None,
            json.dumps(info.metadata) if info.metadata else None
        ))
        
        self.conn.commit()
    
    def search_by_location(self, location: str, fuzzy: bool = True) -> List[Dict]:
        """按地点搜索"""
        cursor = self.conn.cursor()
        
        if fuzzy:
            # 模糊匹配
            cursor.execute("""
                SELECT * FROM structured_info 
                WHERE location LIKE ? 
                ORDER BY id
            """, (f"%{location}%",))
        else:
            # 精确匹配
            cursor.execute("""
                SELECT * FROM structured_info 
                WHERE location = ? 
                ORDER BY id
            """, (location,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def search_by_coordinates(self, lat: float, lon: float, radius_km: float = 10) -> List[Dict]:
        """按坐标范围搜索"""
        cursor = self.conn.cursor()
        
        # 简单的矩形范围搜索（实际应用中可使用PostGIS等专业地理数据库）
        lat_delta = radius_km / 111.0  # 大约每度111km
        lon_delta = radius_km / (111.0 * abs(lat))  # 经度随纬度变化
        
        cursor.execute("""
            SELECT *, 
                   ((latitude - ?) * (latitude - ?) + (longitude - ?) * (longitude - ?)) as distance_sq
            FROM structured_info 
            WHERE latitude BETWEEN ? AND ? 
            AND longitude BETWEEN ? AND ?
            AND latitude IS NOT NULL 
            AND longitude IS NOT NULL
            ORDER BY distance_sq
        """, (
            lat, lat, lon, lon,
            lat - lat_delta, lat + lat_delta,
            lon - lon_delta, lon + lon_delta
        ))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def search_by_date_range(self, start_date: str, end_date: str = None) -> List[Dict]:
        """按日期范围搜索"""
        cursor = self.conn.cursor()
        
        if end_date:
            cursor.execute("""
                SELECT * FROM structured_info 
                WHERE date BETWEEN ? AND ? 
                ORDER BY date DESC
            """, (start_date, end_date))
        else:
            cursor.execute("""
                SELECT * FROM structured_info 
                WHERE date >= ? 
                ORDER BY date DESC
            """, (start_date,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def search_by_category(self, category: str) -> List[Dict]:
        """按类别搜索"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT * FROM structured_info 
            WHERE category = ? 
            ORDER BY id
        """, (category,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def complex_search(self, 
                      location: str = None,
                      category: str = None,
                      date_start: str = None,
                      date_end: str = None,
                      limit: int = 10) -> List[Dict]:
        """复合条件搜索"""
        cursor = self.conn.cursor()
        
        conditions = []
        params = []
        
        if location:
            conditions.append("location LIKE ?")
            params.append(f"%{location}%")
        
        if category:
            conditions.append("category = ?")
            params.append(category)
        
        if date_start:
            conditions.append("date >= ?")
            params.append(date_start)
        
        if date_end:
            conditions.append("date <= ?")
            params.append(date_end)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
            SELECT * FROM structured_info 
            WHERE {where_clause}
            ORDER BY date DESC
            LIMIT ?
        """
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        cursor = self.conn.cursor()
        
        # 总数
        cursor.execute("SELECT COUNT(*) as total FROM structured_info")
        total = cursor.fetchone()["total"]
        
        # 按类别统计
        cursor.execute("""
            SELECT category, COUNT(*) as count 
            FROM structured_info 
            WHERE category IS NOT NULL 
            GROUP BY category 
            ORDER BY count DESC
        """)
        categories = dict(cursor.fetchall())
        
        # 按地点统计（前10）
        cursor.execute("""
            SELECT location, COUNT(*) as count 
            FROM structured_info 
            WHERE location IS NOT NULL 
            GROUP BY location 
            ORDER BY count DESC 
            LIMIT 10
        """)
        locations = dict(cursor.fetchall())
        
        return {
            "total_records": total,
            "categories": categories,
            "top_locations": locations
        }


class HybridRAG:
    """
    混合检索RAG系统
    结合SQL精确查询和向量语义检索
    """
    
    def __init__(self, document_path: str = "1.txt", db_path: str = "./structured_data.db"):
        """
        初始化混合RAG系统
        
        Args:
            document_path: 文档路径
            db_path: 结构化数据库路径
        """
        print(f"🚀 初始化混合RAG系统...")
        
        # 初始化SQL结构化存储
        self.sql_store = SQLStructuredStore(db_path)
        
        # 初始化传统RAG组件
        self.embedding = EmbeddingAdapter.get_embedding("dashscope", "text-embedding-v3")
        self.llm = LLMAdapter.get_llm("dashscope", "qwen-turbo", temperature=0.1)
        
        # 向量存储（用于非结构化内容）
        self.vector_store = VectorStoreManager(
            embedding_model=self.embedding,
            collection_name="hybrid_rag",
            persist_directory="./hybrid_rag_db"
        )
        
        # 构建知识库
        self._build_knowledge_base(document_path)
        
        print(f"✅ 混合RAG系统就绪！")
    
    def _build_knowledge_base(self, document_path: str):
        """构建混合知识库"""
        print(f"📚 构建混合知识库...")
        
        # 检查是否已有数据
        if self.vector_store.is_persistent() and len(self.vector_store) > 0:
            print(f"🔄 发现已有向量库，共 {len(self.vector_store)} 个文档块")
        else:
            # 构建向量知识库
            self._build_vector_knowledge_base(document_path)
        
        # 构建结构化数据（示例）
        self._build_structured_data()
    
    def _build_vector_knowledge_base(self, document_path: str):
        """构建向量知识库"""
        if not os.path.exists(document_path):
            print(f"❌ 文档文件不存在: {document_path}")
            return
        
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        from langchain_core.documents import Document
        document = Document(
            page_content=content,
            metadata={'source': document_path}
        )
        
        self.vector_store.create_from_documents(
            [document],
            chunk_size=800,
            chunk_overlap=100
        )
        
        print(f"💾 向量知识库构建完成，共 {len(self.vector_store)} 个文档块")
    
    def _build_structured_data(self):
        """构建结构化数据示例"""
        print(f"🏗️ 构建结构化数据...")
        
        # 检查是否已有数据
        stats = self.sql_store.get_stats()
        if stats["total_records"] > 0:
            print(f"🔄 发现已有结构化数据 {stats['total_records']} 条")
            return
        
        # 示例结构化数据
        sample_data = [
            StructuredInfo(
                id="office_001",
                content="一微半导体总部办公室，位于北京中关村，员工约500人",
                location="北京中关村",
                coordinates=(39.9042, 116.4074),
                date="2023-01-01",
                category="办公地点",
                tags=["总部", "办公室", "北京"],
                metadata={"building": "科技大厦", "floor": "10-15层"}
            ),
            StructuredInfo(
                id="office_002", 
                content="一微半导体上海分公司，位于浦东新区，主要负责销售业务",
                location="上海浦东新区",
                coordinates=(31.2304, 121.4737),
                date="2023-03-15",
                category="办公地点",
                tags=["分公司", "销售", "上海"],
                metadata={"building": "金融中心", "floor": "28层"}
            ),
            StructuredInfo(
                id="factory_001",
                content="一微半导体生产基地，位于深圳宝安区，主要生产芯片",
                location="深圳宝安区",
                coordinates=(22.5431, 113.8288),
                date="2023-06-01",
                category="生产基地",
                tags=["工厂", "生产", "芯片", "深圳"],
                metadata={"area": "50000平方米", "capacity": "月产100万片"}
            ),
            StructuredInfo(
                id="event_001",
                content="2023年度技术峰会在北京举办，展示最新AI芯片技术",
                location="北京国际会议中心",
                coordinates=(39.9388, 116.3974),
                date="2023-11-15",
                category="会议活动",
                tags=["峰会", "AI芯片", "技术展示"],
                metadata={"participants": 1000, "duration": "2天"}
            ),
            StructuredInfo(
                id="product_001",
                content="AMICRO-AI-001芯片，专为AI推理优化，功耗低性能强",
                location="深圳宝安区",  # 生产地
                coordinates=(22.5431, 113.8288),
                date="2023-09-20",
                category="产品",
                tags=["AI芯片", "推理", "低功耗"],
                metadata={"model": "AMICRO-AI-001", "process": "7nm", "power": "5W"}
            )
        ]
        
        # 插入示例数据
        for data in sample_data:
            self.sql_store.insert(data)
        
        stats = self.sql_store.get_stats()
        print(f"✅ 结构化数据构建完成，共 {stats['total_records']} 条记录")
    
    def _parse_query_intent(self, query: str) -> Dict[str, Any]:
        """解析查询意图，判断是否包含结构化查询需求"""
        intent = {
            "has_location": False,
            "has_date": False,
            "has_category": False,
            "location_terms": [],
            "date_terms": [],
            "category_terms": [],
            "is_structured": False
        }
        
        # 地点关键词
        location_keywords = ["在哪", "位置", "地点", "地址", "哪里", "北京", "上海", "深圳", "广州", "杭州"]
        for keyword in location_keywords:
            if keyword in query:
                intent["has_location"] = True
                intent["location_terms"].append(keyword)
        
        # 时间关键词
        date_keywords = ["时间", "日期", "什么时候", "何时", "年", "月", "日"]
        for keyword in date_keywords:
            if keyword in query:
                intent["has_date"] = True
                intent["date_terms"].append(keyword)
        
        # 类别关键词
        category_keywords = ["办公室", "工厂", "分公司", "总部", "会议", "活动", "产品", "芯片"]
        for keyword in category_keywords:
            if keyword in query:
                intent["has_category"] = True
                intent["category_terms"].append(keyword)
        
        # 判断是否为结构化查询
        intent["is_structured"] = intent["has_location"] or intent["has_date"] or intent["has_category"]
        
        return intent
    
    def _structured_search(self, query: str, intent: Dict[str, Any]) -> List[Dict]:
        """执行结构化搜索"""
        print(f"🔍 执行结构化搜索...")
        
        results = []
        
        # 根据意图执行不同的搜索策略
        if intent["has_location"]:
            # 地点搜索
            for term in intent["location_terms"]:
                if term in ["北京", "上海", "深圳", "广州", "杭州"]:
                    location_results = self.sql_store.search_by_location(term)
                    results.extend(location_results)
        
        if intent["has_category"]:
            # 类别搜索
            category_map = {
                "办公室": "办公地点",
                "工厂": "生产基地", 
                "分公司": "办公地点",
                "总部": "办公地点",
                "会议": "会议活动",
                "活动": "会议活动",
                "产品": "产品",
                "芯片": "产品"
            }
            
            for term in intent["category_terms"]:
                if term in category_map:
                    category_results = self.sql_store.search_by_category(category_map[term])
                    results.extend(category_results)
        
        # 如果没有特定的结构化搜索，尝试复合搜索
        if not results:
            # 提取可能的地点名称
            location = None
            for city in ["北京", "上海", "深圳", "广州", "杭州"]:
                if city in query:
                    location = city
                    break
            
            results = self.sql_store.complex_search(location=location, limit=5)
        
        # 去重
        seen_ids = set()
        unique_results = []
        for result in results:
            if result["id"] not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result["id"])
        
        print(f"📊 结构化搜索找到 {len(unique_results)} 条结果")
        return unique_results
    
    def _vector_search(self, query: str, k: int = 5) -> List[Dict]:
        """执行向量搜索"""
        print(f"🔍 执行向量搜索...")
        
        try:
            retriever = self.vector_store._create_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
            
            docs = retriever.invoke(query)
            results = []
            
            for i, doc in enumerate(docs):
                results.append({
                    "id": f"vector_{i}",
                    "content": doc.page_content,
                    "source": "vector_search",
                    "metadata": doc.metadata
                })
            
            print(f"📊 向量搜索找到 {len(results)} 条结果")
            return results
            
        except Exception as e:
            print(f"❌ 向量搜索失败: {e}")
            return []
    
    def hybrid_query(self, query: str, show_sources: bool = False) -> Dict[str, Any]:
        """混合查询方法"""
        print(f"🧠 开始混合查询: {query}")
        start_time = time.time()
        
        try:
            # 1. 解析查询意图
            intent = self._parse_query_intent(query)
            print(f"🎯 查询意图: {intent}")
            
            # 2. 执行混合搜索
            structured_results = []
            vector_results = []
            
            if intent["is_structured"]:
                # 结构化搜索
                structured_results = self._structured_search(query, intent)
                
                # 如果结构化搜索结果不足，补充向量搜索
                if len(structured_results) < 3:
                    vector_results = self._vector_search(query, k=3)
            else:
                # 主要使用向量搜索
                vector_results = self._vector_search(query, k=5)
                
                # 补充结构化搜索
                structured_results = self.sql_store.complex_search(limit=2)
            
            # 3. 合并结果
            all_results = structured_results + vector_results
            
            if not all_results:
                return {
                    "question": query,
                    "answer": "抱歉，没有找到相关信息。",
                    "response_time": f"{time.time() - start_time:.2f}秒",
                    "search_strategy": "hybrid",
                    "sources": []
                }
            
            # 4. 构建上下文
            context_parts = []
            for result in all_results[:8]:  # 限制上下文长度
                context_parts.append(result["content"])
            
            context = "\n\n".join(context_parts)
            
            # 5. 生成回答
            prompt = f"""你是一微半导体公司的智能助手。请基于以下信息回答用户问题。

信息来源包括：
- 结构化数据（地点、时间、类别等精确信息）
- 文档内容（详细描述和背景信息）

相关信息：
{context}

用户问题: {query}

请综合以上信息给出准确、详细的回答。如果涉及地点、时间等具体信息，请明确说明。

回答："""

            answer = self.llm.invoke(prompt)
            answer_text = answer.content if hasattr(answer, 'content') else str(answer)
            
            # 6. 准备返回结果
            sources = []
            if show_sources:
                for result in all_results[:5]:
                    source_info = {
                        "content": result["content"][:200] + "...",
                        "type": "structured" if "location" in result else "vector",
                        "metadata": result.get("metadata", {})
                    }
                    if "location" in result:
                        source_info["location"] = result["location"]
                    if "category" in result:
                        source_info["category"] = result["category"]
                    sources.append(source_info)
            
            result = {
                "question": query,
                "answer": answer_text,
                "response_time": f"{time.time() - start_time:.2f}秒",
                "search_strategy": "hybrid",
                "intent_analysis": intent,
                "structured_results_count": len(structured_results),
                "vector_results_count": len(vector_results),
                "sources": sources
            }
            
            print(f"✅ 混合查询完成，耗时: {result['response_time']}")
            return result
            
        except Exception as e:
            return {
                "question": query,
                "answer": f"抱歉，查询过程中出错: {str(e)}",
                "error": str(e),
                "response_time": f"{time.time() - start_time:.2f}秒"
            }
    
    def demo(self):
        """演示混合RAG功能"""
        print("\n" + "="*60)
        print("🎯 混合RAG系统演示")
        print("="*60)
        
        # 显示数据库统计
        stats = self.sql_store.get_stats()
        print(f"📊 结构化数据统计:")
        print(f"  总记录数: {stats['total_records']}")
        print(f"  类别分布: {stats['categories']}")
        print(f"  主要地点: {list(stats['top_locations'].keys())[:5]}")
        print(f"  向量库文档: {len(self.vector_store)} 个块")
        
        # 测试查询
        test_queries = [
            "一微半导体在北京有哪些办公室？",  # 结构化查询
            "深圳的工厂主要生产什么？",      # 结构化+向量
            "公司有哪些AI芯片产品？",       # 主要向量查询
            "2023年有什么重要活动？",       # 时间结构化查询
            "公司的核心价值观是什么？"       # 纯向量查询
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 测试 {i}: {query}")
            print("-" * 50)
            
            result = self.hybrid_query(query, show_sources=True)
            
            print(f"💬 回答: {result['answer']}")
            print(f"⏱️ 耗时: {result['response_time']}")
            print(f"🎯 搜索策略: {result['search_strategy']}")
            
            if 'intent_analysis' in result:
                intent = result['intent_analysis']
                if intent['is_structured']:
                    print(f"🏗️ 结构化特征: 地点={intent['has_location']}, 时间={intent['has_date']}, 类别={intent['has_category']}")
            
            print(f"📊 结果统计: 结构化={result.get('structured_results_count', 0)}, 向量={result.get('vector_results_count', 0)}")
        
        print(f"\n✅ 演示完成！")
    
    def interactive(self):
        """交互模式"""
        print("\n" + "="*60)
        print("💬 混合RAG交互模式")
        print("🔧 特殊命令: 'stats' - 数据统计, 'search <地点>' - 地点搜索, 'quit' - 退出")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n🤔 请输入问题: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                elif user_input.lower() == 'stats':
                    stats = self.sql_store.get_stats()
                    print(f"\n📊 数据库统计:")
                    print(f"  总记录: {stats['total_records']}")
                    print(f"  类别: {stats['categories']}")
                    print(f"  地点: {stats['top_locations']}")
                elif user_input.startswith('search '):
                    location = user_input[7:].strip()
                    results = self.sql_store.search_by_location(location)
                    print(f"\n🔍 地点 '{location}' 的搜索结果:")
                    for result in results:
                        print(f"  - {result['content']}")
                else:
                    # 普通查询
                    result = self.hybrid_query(user_input, show_sources=True)
                    print(f"\n💬 回答:\n{result['answer']}")
                    print(f"\n⏱️ 耗时: {result['response_time']}")
                    
                    # 显示来源
                    if result.get('sources'):
                        print(f"\n📚 信息来源:")
                        for i, source in enumerate(result['sources'], 1):
                            source_type = "📍" if source.get('type') == 'structured' else "📄"
                            print(f"  {source_type} {i}. {source['content']}")
                            if source.get('location'):
                                print(f"      地点: {source['location']}")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")


def main():
    """主函数"""
    try:
        # 初始化混合RAG系统
        hybrid_rag = HybridRAG("1.txt")
        
        print("\n选择模式:")
        print("1. 演示模式")
        print("2. 交互模式")
        
        choice = input("请选择 (1/2): ").strip()
        
        if choice == "1":
            hybrid_rag.demo()
        elif choice == "2":
            hybrid_rag.interactive()
        else:
            print("❌ 无效选择，运行演示模式")
            hybrid_rag.demo()
            
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 