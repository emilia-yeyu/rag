#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迎宾机器人知识库系统
基于SQLite3数据库的人物信息查询服务
"""

import json
import sys
import os
from typing import Dict, List, Optional, Any
from datetime import datetime

# 导入数据库管理器
from database_manager import DatabaseManager

class GreetingRobotKnowledgeBase:
    def __init__(self, db_path: str = None):
        """初始化迎宾机器人知识库"""
        # 如果没有指定数据库路径，使用当前目录下的数据库文件
        if db_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(current_dir, "greeting_robot.db")
        
        self.db_manager = DatabaseManager(db_path)
        print(f"✓ SQLite数据库系统初始化成功，数据库路径: {db_path}")
    
    def query_person_info(self, name: str) -> Dict[str, Any]:
        """
        查询人物信息
        返回JSON格式的完整人物信息
        """
        result = {
            "query_time": datetime.now().isoformat(),
            "person_name": name,
            "person_data": None,
            "description": ""
        }
        
        # 从SQLite查询人物信息
        person_info = self.db_manager.get_person_info(name)
        if person_info:
            result["person_data"] = person_info
            result["description"] = self._generate_description(person_info)
        
        return result
    
    def _generate_description(self, person_info: Dict[str, Any]) -> str:
        """生成人物描述"""
        basic_info = person_info
        meetings = person_info.get("meetings", [])
        
        description = f"{basic_info.get('name', '')}是{basic_info.get('title', '')}，"
        description += f"在{basic_info.get('department', '')}工作，"
        description += f"办公室位于{basic_info.get('office_location', '')}。"
        
        if basic_info.get('bio'):
            description += f" {basic_info.get('bio')}"
        
        if meetings:
            description += f" 目前有{len(meetings)}个会议安排。"
        
        return description
    
    def search_person(self, name: str) -> List[Dict[str, Any]]:
        """模糊搜索人物"""
        return self.db_manager.search_person_by_name(name)
    
    def get_all_persons(self) -> List[Dict[str, Any]]:
        """获取所有人物列表"""
        return self.db_manager.get_all_persons()
    
    def get_today_meetings(self) -> List[Dict[str, Any]]:
        """获取今天的会议安排"""
        return self.db_manager.get_today_meetings()

def main():
    """主程序入口"""
    print("🤖 迎宾机器人知识库系统")
    print("=" * 50)
    
    # 初始化知识库
    kb = GreetingRobotKnowledgeBase()
    
    print("\n系统已启动，请输入人名进行查询（输入 'quit' 退出）：")
    
    while True:
        try:
            # 获取用户输入
            name = input("\n请输入人名: ").strip()
            
            if name.lower() in ['quit', 'exit', '退出']:
                print("👋 再见！")
                break
            
            if not name:
                print("❌ 请输入有效的人名")
                continue
            
            print(f"\n🔍 正在查询 {name} 的信息...")
            
            # 查询人物信息
            result = kb.query_person_info(name)
            
            # 输出JSON格式结果
            print("\n📋 查询结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            # 如果没找到，提供搜索建议
            if not result["person_data"]:
                print(f"\n❓ 未找到 '{name}' 的精确匹配，是否要搜索相似名字？")
                similar_persons = kb.search_person(name)
                if similar_persons:
                    print("🔍 找到相似的人物：")
                    for person in similar_persons:
                        print(f"  - {person['name']} ({person['title']})")
                else:
                    print("❌ 没有找到相似的人物")
        
        except KeyboardInterrupt:
            print("\n\n👋 程序被中断，再见！")
            break
        except Exception as e:
            print(f"\n❌ 查询过程中出现错误: {e}")

if __name__ == "__main__":
    main() 