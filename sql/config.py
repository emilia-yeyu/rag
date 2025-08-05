#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
迎宾机器人知识库系统配置文件
"""

import os
from typing import Dict, Any

class Config:
    """系统配置类"""
    
    # 数据库配置
    DATABASE_PATH = "greeting_robot.db"
    DATABASE_BACKUP_PATH = "backup/greeting_robot_backup.db"
    
    # 日志配置
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/greeting_robot.log"
    
    # 查询配置
    MAX_SEARCH_RESULTS = 10
    ENABLE_FUZZY_SEARCH = True
    
    # 输出配置
    JSON_INDENT = 2
    ENABLE_COLORED_OUTPUT = True
    
    # 安全配置
    ENABLE_SQL_INJECTION_PROTECTION = True
    MAX_QUERY_LENGTH = 1000
    
    @classmethod
    def get_database_config(cls) -> Dict[str, Any]:
        """获取数据库配置"""
        return {
            "db_path": cls.DATABASE_PATH,
            "backup_path": cls.DATABASE_BACKUP_PATH,
            "enable_backup": True
        }
    
    @classmethod
    def get_logging_config(cls) -> Dict[str, Any]:
        """获取日志配置"""
        return {
            "level": cls.LOG_LEVEL,
            "file": cls.LOG_FILE,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    
    @classmethod
    def create_directories(cls):
        """创建必要的目录"""
        directories = [
            "logs",
            "backup"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def validate_config(cls) -> bool:
        """验证配置"""
        try:
            # 检查数据库路径
            if not os.path.dirname(cls.DATABASE_PATH):
                os.makedirs(os.path.dirname(cls.DATABASE_PATH), exist_ok=True)
            
            # 创建必要目录
            cls.create_directories()
            
            return True
        except Exception as e:
            print(f"配置验证失败: {e}")
            return False

# 开发环境配置
class DevelopmentConfig(Config):
    """开发环境配置"""
    LOG_LEVEL = "DEBUG"
    ENABLE_COLORED_OUTPUT = True

# 生产环境配置
class ProductionConfig(Config):
    """生产环境配置"""
    LOG_LEVEL = "WARNING"
    ENABLE_COLORED_OUTPUT = False
    ENABLE_SQL_INJECTION_PROTECTION = True

# 测试环境配置
class TestConfig(Config):
    """测试环境配置"""
    DATABASE_PATH = "test_greeting_robot.db"
    LOG_LEVEL = "DEBUG"

# 根据环境变量选择配置
def get_config():
    """根据环境变量获取配置"""
    env = os.getenv("GREETING_ROBOT_ENV", "development").lower()
    
    if env == "production":
        return ProductionConfig
    elif env == "test":
        return TestConfig
    else:
        return DevelopmentConfig

# 默认配置
config = get_config()