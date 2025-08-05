import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

class DatabaseManager:
    def __init__(self, db_path: str = "greeting_robot.db"):
        """初始化数据库管理器"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """初始化数据库，创建表结构"""
        # 读取SQL脚本
        sql_file_path = os.path.join(os.path.dirname(__file__), "create_tables.sql")
        
        if not os.path.exists(sql_file_path):
            raise FileNotFoundError(f"SQL文件不存在: {sql_file_path}")
        
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        # 执行SQL脚本
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(sql_script)
            conn.commit()
    
    def get_person_info(self, name: str) -> Optional[Dict[str, Any]]:
        """根据姓名获取人物信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # 查询人物基本信息
            cursor.execute("""
                SELECT * FROM person_info WHERE name = ?
            """, (name,))
            
            person = cursor.fetchone()
            if not person:
                return None
            
            # 查询该人物的会议日程
            cursor.execute("""
                SELECT * FROM meeting_schedule 
                WHERE person_id = ? 
                ORDER BY meeting_date DESC, start_time ASC
            """, (person['id'],))
            
            meetings = cursor.fetchall()
            
            # 构建返回数据
            person_info = {
                'id': person['id'],
                'name': person['name'],
                'title': person['title'],
                'department': person['department'],
                'office_location': person['office_location'],
                'phone': person['phone'],
                'email': person['email'],
                'avatar_url': person['avatar_url'],
                'bio': person['bio'],
                'meetings': []
            }
            
            for meeting in meetings:
                meeting_info = {
                    'meeting_title': meeting['meeting_title'],
                    'meeting_date': meeting['meeting_date'],
                    'start_time': meeting['start_time'],
                    'end_time': meeting['end_time'],
                    'location': meeting['location'],
                    'attendees': meeting['attendees'],
                    'description': meeting['description'],
                    'status': meeting['status']
                }
                person_info['meetings'].append(meeting_info)
            
            return person_info
    
    def search_person_by_name(self, name: str) -> List[Dict[str, Any]]:
        """模糊搜索人物信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM person_info 
                WHERE name LIKE ? 
                ORDER BY name
            """, (f'%{name}%',))
            
            persons = cursor.fetchall()
            return [dict(person) for person in persons]
    
    def get_all_persons(self) -> List[Dict[str, Any]]:
        """获取所有人物信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM person_info ORDER BY name")
            persons = cursor.fetchall()
            return [dict(person) for person in persons]
    
    def add_person(self, person_data: Dict[str, Any]) -> bool:
        """添加新人物"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO person_info (name, title, department, office_location, phone, email, bio)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    person_data.get('name'),
                    person_data.get('title'),
                    person_data.get('department'),
                    person_data.get('office_location'),
                    person_data.get('phone'),
                    person_data.get('email'),
                    person_data.get('bio')
                ))
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            return False  # 姓名重复
    
    def add_meeting(self, meeting_data: Dict[str, Any]) -> bool:
        """添加会议日程"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO meeting_schedule (person_id, meeting_title, meeting_date, start_time, end_time, location, attendees, description)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    meeting_data.get('person_id'),
                    meeting_data.get('meeting_title'),
                    meeting_data.get('meeting_date'),
                    meeting_data.get('start_time'),
                    meeting_data.get('end_time'),
                    meeting_data.get('location'),
                    meeting_data.get('attendees'),
                    meeting_data.get('description')
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"添加会议失败: {e}")
            return False
    
    def get_today_meetings(self) -> List[Dict[str, Any]]:
        """获取今天的会议日程"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT m.*, p.name as person_name, p.title as person_title
                FROM meeting_schedule m
                JOIN person_info p ON m.person_id = p.id
                WHERE m.meeting_date = ?
                ORDER BY m.start_time
            """, (today,))
            
            meetings = cursor.fetchall()
            return [dict(meeting) for meeting in meetings]

if __name__ == "__main__":
    # 测试数据库功能
    db = DatabaseManager()
    
    # 测试查询
    person = db.get_person_info("张三")
    if person:
        print(json.dumps(person, ensure_ascii=False, indent=2))
    else:
        print("未找到该人物") 