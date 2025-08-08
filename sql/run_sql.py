#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SQL执行工具 - 用于执行SQL文件或单个SQL语句
"""

import sqlite3
import sys
import os
from pathlib import Path

def execute_sql_file(db_path, sql_file_path):
    """执行SQL文件"""
    if not os.path.exists(sql_file_path):
        print(f"❌ SQL文件不存在: {sql_file_path}")
        return False
    
    try:
        with open(sql_file_path, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        # 分割SQL语句（简单的分割，按分号分割）
        sql_statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip() and not stmt.strip().startswith('--')]
        
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            for i, statement in enumerate(sql_statements, 1):
                if statement:
                    try:
                        print(f"🔄 执行第 {i} 条SQL语句...")
                        cursor.execute(statement)
                        
                        # 如果是查询语句，显示结果
                        if statement.strip().upper().startswith('SELECT'):
                            results = cursor.fetchall()
                            if results:
                                # 显示列名
                                columns = [description[0] for description in cursor.description]
                                print(f"📊 查询结果 ({len(results)} 行):")
                                print("-" * 80)
                                print(" | ".join(f"{col:^15}" for col in columns))
                                print("-" * 80)
                                
                                # 显示数据
                                for row in results:
                                    print(" | ".join(f"{str(row[col]) if row[col] is not None else 'NULL':^15}" for col in columns))
                                print("-" * 80)
                            else:
                                print("📊 查询结果: 无数据")
                        else:
                            # 对于非查询语句，显示影响的行数
                            affected_rows = cursor.rowcount
                            print(f"✅ 执行成功，影响 {affected_rows} 行")
                        
                        print()
                        
                    except sqlite3.Error as e:
                        print(f"❌ 第 {i} 条SQL语句执行失败: {e}")
                        print(f"SQL: {statement[:100]}...")
                        print()
            
            conn.commit()
            print("🎉 SQL文件执行完成！")
            
    except Exception as e:
        print(f"❌ 执行SQL文件时发生错误: {e}")
        return False
    
    return True

def execute_single_sql(db_path, sql_statement):
    """执行单个SQL语句"""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute(sql_statement)
            
            # 如果是查询语句，显示结果
            if sql_statement.strip().upper().startswith('SELECT'):
                results = cursor.fetchall()
                if results:
                    # 显示列名
                    columns = [description[0] for description in cursor.description]
                    print(f"📊 查询结果 ({len(results)} 行):")
                    print("-" * 80)
                    print(" | ".join(f"{col:^15}" for col in columns))
                    print("-" * 80)
                    
                    # 显示数据
                    for row in results:
                        print(" | ".join(f"{str(row[col]) if row[col] is not None else 'NULL':^15}" for col in columns))
                    print("-" * 80)
                else:
                    print("📊 查询结果: 无数据")
            else:
                # 对于非查询语句，显示影响的行数
                affected_rows = cursor.rowcount
                print(f"✅ 执行成功，影响 {affected_rows} 行")
            
            conn.commit()
            
    except sqlite3.Error as e:
        print(f"❌ SQL执行失败: {e}")
        return False
    
    return True

def interactive_mode(db_path):
    """交互模式"""
    print("🚀 进入SQL交互模式")
    print("提示：")
    print("- 输入SQL语句后按回车执行")
    print("- 输入 'exit' 或 'quit' 退出")
    print("- 输入 'help' 查看帮助")
    print("-" * 50)
    
    while True:
        try:
            sql = input("SQL> ").strip()
            
            if sql.lower() in ['exit', 'quit']:
                print("👋 退出交互模式")
                break
            elif sql.lower() == 'help':
                print("🔍 常用命令示例:")
                print("SELECT * FROM person_info;")
                print("SELECT * FROM meeting_schedule;")
                print("INSERT INTO person_info (name, title, department, office_location, bio) VALUES ('姓名', '职位', '部门', '地点', '简介');")
                print("UPDATE person_info SET title='新职位' WHERE name='姓名';")
                print("DELETE FROM meeting_schedule WHERE id=1;")
                continue
            elif not sql:
                continue
            
            execute_single_sql(db_path, sql)
            
        except KeyboardInterrupt:
            print("\n👋 退出交互模式")
            break
        except EOFError:
            print("\n👋 退出交互模式")
            break

def main():
    """主函数"""
    db_path = "greeting_robot.db"
    
    print("🤖 AMICRO一微半导体数据库SQL执行工具")
    print("=" * 50)
    
    if len(sys.argv) == 1:
        # 无参数，进入交互模式
        interactive_mode(db_path)
    elif len(sys.argv) == 2:
        # 一个参数，可能是SQL文件或SQL语句
        arg = sys.argv[1]
        
        if os.path.exists(arg) and arg.endswith('.sql'):
            # 是SQL文件
            print(f"📁 执行SQL文件: {arg}")
            execute_sql_file(db_path, arg)
        else:
            # 是SQL语句
            print(f"📝 执行SQL语句: {arg}")
            execute_single_sql(db_path, arg)
    else:
        print("使用方法:")
        print("1. python3 run_sql.py                    # 进入交互模式")
        print("2. python3 run_sql.py file.sql          # 执行SQL文件")
        print("3. python3 run_sql.py \"SELECT * FROM person_info\"  # 执行单个SQL语句")

if __name__ == "__main__":
    main()
