#!/usr/bin/env python3
"""
SQL检索器模块
用于处理结构化数据的精确查询
llm去生成sql查询语句
"""
import sqlite3
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from llm.adapter import LLMAdapter


class DatabaseManager:
    """SQLite数据库管理器"""
    
    def __init__(self, db_path: str = "guest_info.db"):
        """
        初始化数据库管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """初始化数据库和表结构"""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row  # 使结果可以通过列名访问
            
            # 创建客人信息表
            self._create_guest_table()
            
            # 插入示例数据
            self._insert_sample_data()
            
            print(f"✅ SQLite数据库初始化成功: {self.db_path}")
            
        except Exception as e:
            print(f"❌ 数据库初始化失败: {e}")
            raise
    
    def _create_guest_table(self):
        """创建客人信息表"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS guests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT,
            position TEXT,
            age INTEGER,
            company TEXT,
            email TEXT,
            address TEXT,
            notes TEXT,
            created_date TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
        
        self.conn.execute(create_table_sql)
        self.conn.commit()
        print("📊 客人信息表创建成功")
    
    def _insert_sample_data(self):
        """插入示例数据"""
        # 检查是否已有数据
        cursor = self.conn.execute("SELECT COUNT(*) FROM guests")
        count = cursor.fetchone()[0]
        
        if count > 0:
            print(f"📊 数据库已有 {count} 条记录，跳过示例数据插入")
            return
        
        sample_data = [
            ("贾宝玉", "13800001001", "公子", 17, "荣国府", "baoyu@rongguo.com", "北京大观园", "性情温和，喜读书"),
            ("林黛玉", "13800001002", "小姐", 16, "林府", "daiyu@lin.com", "苏州林府", "才华横溢，多愁善感"),
            ("薛宝钗", "13800001003", "小姐", 18, "薛家", "baochai@xue.com", "金陵薛府", "端庄贤淑，善于理财"),
            ("王熙凤", "13800001004", "少奶奶", 25, "荣国府", "xifeng@rongguo.com", "北京荣国府", "精明能干，管理有方"),
            ("贾迎春", "13800001005", "二小姐", 16, "荣国府", "yingchun@rongguo.com", "北京荣国府", "性格温顺，不善言辞"),
            ("贾探春", "13800001006", "三小姐", 15, "荣国府", "tanchun@rongguo.com", "北京荣国府", "精明干练，有管理才能"),
            ("贾惜春", "13800001007", "四小姐", 14, "宁国府", "xichun@ningguo.com", "北京宁国府", "性格孤僻，喜画画"),
            ("史湘云", "13800001008", "小姐", 16, "史府", "xiangyun@shi.com", "金陵史府", "豪爽开朗，喜饮酒作诗"),
            ("妙玉", "13800001009", "道姑", 20, "栊翠庵", "miaoyu@longcui.com", "大观园栊翠庵", "清高孤傲，精通茶道"),
            ("刘姥姥", "13800001010", "村妇", 75, "乡下", "laolao@country.com", "乡下农村", "质朴善良，见多识广"),
            ("贾政", "13800001011", "员外郎", 45, "荣国府", "zheng@rongguo.com", "北京荣国府", "正直严肃，重视教育"),
            ("王夫人", "13800001012", "夫人", 42, "荣国府", "wangfuren@rongguo.com", "北京荣国府", "慈善温和，信佛"),
            ("贾赦", "13800001013", "将军", 50, "荣国府", "she@rongguo.com", "北京荣国府", "贪财好色，不务正业"),
            ("邢夫人", "13800001014", "夫人", 45, "荣国府", "xingfuren@rongguo.com", "北京荣国府", "小心谨慎，畏惧贾母"),
            ("贾琏", "13800001015", "少爷", 30, "荣国府", "lian@rongguo.com", "北京荣国府", "风流倜傥，善于交际")
        ]
        
        insert_sql = """
        INSERT INTO guests (name, phone, position, age, company, email, address, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        self.conn.executemany(insert_sql, sample_data)
        self.conn.commit()
        
        print(f"✅ 成功插入 {len(sample_data)} 条示例数据")
    
    def execute_query(self, sql: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
        """
        执行SQL查询
        
        Args:
            sql: SQL查询语句
            params: 查询参数
        
        Returns:
            查询结果列表
        """
        try:
            if params:
                cursor = self.conn.execute(sql, params)
            else:
                cursor = self.conn.execute(sql)
            
            results = []
            for row in cursor.fetchall():
                # 将sqlite3.Row转换为字典
                results.append(dict(row))
            
            return results
            
        except Exception as e:
            print(f"❌ SQL查询执行失败: {e}")
            print(f"SQL: {sql}")
            if params:
                print(f"参数: {params}")
            return []
    
    def get_table_schema(self) -> str:
        """获取表结构信息"""
        schema_sql = "PRAGMA table_info(guests)"
        columns = self.execute_query(schema_sql)
        
        schema_info = "客人信息表 (guests) 结构:\n"
        for col in columns:
            schema_info += f"- {col['name']} ({col['type']}): "
            if col['name'] == 'id':
                schema_info += "主键，自动递增\n"
            elif col['name'] == 'name':
                schema_info += "客人姓名\n"
            elif col['name'] == 'phone':
                schema_info += "电话号码\n"
            elif col['name'] == 'position':
                schema_info += "职位\n"
            elif col['name'] == 'age':
                schema_info += "年龄\n"
            elif col['name'] == 'company':
                schema_info += "公司/家族\n"
            elif col['name'] == 'email':
                schema_info += "邮箱\n"
            elif col['name'] == 'address':
                schema_info += "地址\n"
            elif col['name'] == 'notes':
                schema_info += "备注信息\n"
            elif col['name'] == 'created_date':
                schema_info += "创建时间\n"
            else:
                schema_info += "\n"
        
        return schema_info
    
    def get_sample_data(self, limit: int = 3) -> str:
        """获取示例数据"""
        sample_sql = f"SELECT * FROM guests LIMIT {limit}"
        results = self.execute_query(sample_sql)
        
        sample_info = f"示例数据 (前{limit}条):\n"
        for i, row in enumerate(results, 1):
            sample_info += f"{i}. 姓名: {row['name']}, 年龄: {row['age']}, 职位: {row['position']}, 公司: {row['company']}\n"
        
        return sample_info
    
    def close(self):
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()


class SQLQueryGenerator:
    """SQL查询生成器"""
    
    def __init__(self, llm_adapter=None):
        """
        初始化SQL查询生成器
        
        Args:
            llm_adapter: LLM适配器
        """
        if llm_adapter is None:
            self.llm = LLMAdapter.get_llm("openai", "Qwen/Qwen2.5-7B-Instruct", temperature=0.1)
        else:
            self.llm = llm_adapter
        
        self.db_manager = DatabaseManager()
        
        # SQL生成提示模板
        self.sql_prompt = PromptTemplate(
            template="""你是一个SQL查询专家。根据用户的自然语言问题，生成对应的SQLite查询语句。

数据库表结构：
{schema}

{sample_data}

用户问题：{question}

请根据问题生成相应的SQL查询语句。注意：
1. 只返回SQL语句，不要包含其他解释
2. 使用LIKE进行模糊匹配，例如：name LIKE '%张%'  
3. 对于年龄范围查询，使用BETWEEN或比较操作符
4. 如果问题不涉及客人信息查询，返回：NO_SQL_NEEDED
5. 确保SQL语句语法正确，适用于SQLite

SQL查询：""",
            input_variables=["schema", "sample_data", "question"]
        )
    
    def is_sql_query(self, question: str) -> bool:
        """
        判断问题是否需要SQL查询
        这里可能还要改一下，比如收到了某个人名就检测
        Args:
            question: 用户问题
        
        Returns:
            是否需要SQL查询
        """
        # 关键词检测
        sql_keywords = [
            "姓名", "名字", "叫什么", "年龄", "多大", "电话", "手机", "联系方式",
            "职位", "工作", "公司", "家族", "地址", "住址", "邮箱", "email",
            "几岁", "多少岁", "什么工作", "什么职位", "哪里工作", "住在哪",
            "客人", "人员", "联系", "信息", "查找", "搜索"
        ]
        
        return any(keyword in question for keyword in sql_keywords)
    
    def generate_sql(self, question: str) -> Optional[str]:
        """
        生成SQL查询语句
        
        Args:
            question: 用户问题
        
        Returns:
            SQL查询语句或None
        """
        try:
            # 首先判断是否需要SQL查询
            if not self.is_sql_query(question):
                return None
            
            # 获取数据库结构信息
            schema = self.db_manager.get_table_schema()
            sample_data = self.db_manager.get_sample_data()
            
            # 生成SQL
            prompt_text = self.sql_prompt.format(
                schema=schema,
                sample_data=sample_data,
                question=question
            )
            
            response = self.llm.invoke(prompt_text)
            sql = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # 清理SQL语句
            sql = self._clean_sql(sql)
            
            if sql == "NO_SQL_NEEDED":
                return None
            
            print(f"🔍 生成SQL查询: {sql}")
            return sql
            
        except Exception as e:
            print(f"❌ SQL生成失败: {e}")
            return None
    
    def _clean_sql(self, sql: str) -> str:
        """清理SQL语句"""
        # 移除多余的空白和换行
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        # 移除可能的代码块标记
        sql = re.sub(r'```sql\s*', '', sql)
        sql = re.sub(r'```\s*', '', sql)
        
        # 移除分号（如果有）
        sql = sql.rstrip(';')
        
        return sql


class SQLRetriever:
    """SQL检索器"""
    
    def __init__(self, llm_adapter=None):
        """
        初始化SQL检索器
        
        Args:
            llm_adapter: LLM适配器
        """
        self.query_generator = SQLQueryGenerator(llm_adapter)
        self.db_manager = self.query_generator.db_manager
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """
        SQL检索
        
        Args:
            query: 查询文本
            k: 最大返回数量
        
        Returns:
            检索结果
        """
        try:
            # 生成SQL查询
            sql = self.query_generator.generate_sql(query)
            
            if not sql:
                print("⚠️ 该问题不需要SQL查询")
                return []
            
            # 执行SQL查询
            results = self.db_manager.execute_query(sql)
            
            if not results:
                print("📊 SQL查询无结果")
                return []
            
            print(f"📊 SQL查询找到 {len(results)} 条记录")
            
            # 将结果转换为Document格式
            documents = []
            for i, row in enumerate(results[:k]):
                # 构建文档内容
                content = self._format_result(row)
                
                # 创建元数据
                metadata = {
                    'source': 'sql_database',
                    'table': 'guests',
                    'record_id': row.get('id'),
                    'sql_query': sql,
                    'search_type': 'sql'
                }
                
                doc = Document(page_content=content, metadata=metadata)
                # SQL检索的分数固定为1.0（精确匹配）
                documents.append((doc, 1.0))
            
            return documents
            
        except Exception as e:
            print(f"❌ SQL检索失败: {e}")
            return []
    
    def _format_result(self, row: Dict[str, Any]) -> str:
        """
        格式化SQL查询结果
        
        Args:
            row: 数据库行记录
        
        Returns:
            格式化后的文本
        """
        # 构建客人信息描述
        content_parts = []
        
        if row.get('name'):
            content_parts.append(f"姓名：{row['name']}")
        
        if row.get('age'):
            content_parts.append(f"年龄：{row['age']}岁")
        
        if row.get('position'):
            content_parts.append(f"职位：{row['position']}")
        
        if row.get('company'):
            content_parts.append(f"所属：{row['company']}")
        
        if row.get('phone'):
            content_parts.append(f"电话：{row['phone']}")
        
        if row.get('email'):
            content_parts.append(f"邮箱：{row['email']}")
        
        if row.get('address'):
            content_parts.append(f"地址：{row['address']}")
        
        if row.get('notes'):
            content_parts.append(f"备注：{row['notes']}")
        
        return "；".join(content_parts)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        try:
            # 总记录数
            total_sql = "SELECT COUNT(*) as total FROM guests"
            total_result = self.db_manager.execute_query(total_sql)
            total_count = total_result[0]['total'] if total_result else 0
            
            # 年龄分布
            age_sql = "SELECT MIN(age) as min_age, MAX(age) as max_age, AVG(age) as avg_age FROM guests"
            age_result = self.db_manager.execute_query(age_sql)
            age_stats = age_result[0] if age_result else {}
            
            # 职位分布
            position_sql = "SELECT position, COUNT(*) as count FROM guests GROUP BY position ORDER BY count DESC LIMIT 5"
            position_results = self.db_manager.execute_query(position_sql)
            
            return {
                "total_records": total_count,
                "age_stats": age_stats,
                "top_positions": position_results
            }
            
        except Exception as e:
            print(f"❌ 获取统计信息失败: {e}")
            return {} 