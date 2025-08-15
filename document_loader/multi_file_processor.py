#!/usr/bin/env python3
"""
多文件文档处理器
专门处理data目录下的多个txt文件，每个文件作为一个独立的chunk
"""

import os
import glob
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document


class MultiFileProcessor:
    """
    处理多个txt文件，每个文件作为一个独立的文档块
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        初始化多文件处理器
        
        Args:
            data_dir: 数据目录路径
        """
        # 如果是相对路径，尝试在脚本所在目录查找
        if not os.path.isabs(data_dir):
            script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(script_dir, data_dir)
            if os.path.exists(full_path):
                data_dir = full_path
        
        self.data_dir = data_dir
        print(f"📁 数据目录: {self.data_dir}")
        
        # 检查目录是否存在
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        if not os.path.isdir(self.data_dir):
            raise ValueError(f"路径不是目录: {self.data_dir}")
    
    def load_documents(self, file_pattern: str = "*.txt") -> List[Document]:
        """
        加载所有txt文件作为独立的文档
        
        Args:
            file_pattern: 文件匹配模式，默认为"*.txt"
            
        Returns:
            List[Document]: 文档列表，每个文件对应一个Document
        """
        # 获取所有txt文件
        pattern = os.path.join(self.data_dir, file_pattern)
        txt_files = glob.glob(pattern)
        
        if not txt_files:
            print(f"⚠️ 在 {self.data_dir} 中未找到匹配 {file_pattern} 的文件")
            return []
        
        # 按文件名排序（支持数字排序）
        txt_files.sort(key=lambda x: self._natural_sort_key(os.path.basename(x)))
        
        print(f"📄 找到 {len(txt_files)} 个txt文件")
        
        documents = []
        for file_path in txt_files:
            try:
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if not content:
                    print(f"⚠️ 文件为空，跳过: {os.path.basename(file_path)}")
                    continue
                
                # 创建Document对象
                filename = os.path.basename(file_path)
                file_number = self._extract_file_number(filename)
                
                # 提取标题（通常是第一行）
                lines = content.split('\n')
                title = lines[0].strip() if lines else filename
                
                document = Document(
                    page_content=content,
                    metadata={
                        'source': file_path,
                        'filename': filename,
                        'file_number': file_number,
                        'title': title,
                        'chunk_type': 'file',
                        'chunk_id': filename.replace('.txt', ''),
                        'char_count': len(content),
                        'line_count': len(lines)
                    }
                )
                
                documents.append(document)
                print(f"✅ 加载文件: {filename} ({len(content)} 字符)")
                
            except Exception as e:
                print(f"❌ 加载文件失败 {os.path.basename(file_path)}: {e}")
                continue
        
        print(f"🎯 成功加载 {len(documents)} 个文档")
        return documents
    
    def _natural_sort_key(self, filename: str) -> tuple:
        """
        自然排序键，支持数字排序（1.txt, 2.txt, ..., 10.txt, 11.txt）
        """
        import re
        # 提取文件名中的数字部分
        parts = re.split(r'(\d+)', filename)
        return tuple(int(part) if part.isdigit() else part for part in parts)
    
    def _extract_file_number(self, filename: str) -> Optional[int]:
        """
        从文件名中提取数字编号
        """
        import re
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else None
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        获取数据目录信息
        """
        pattern = os.path.join(self.data_dir, "*.txt")
        txt_files = glob.glob(pattern)
        
        total_size = 0
        file_info = []
        
        for file_path in txt_files:
            try:
                size = os.path.getsize(file_path)
                total_size += size
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = len(content.split('\n'))
                    chars = len(content)
                
                file_info.append({
                    'filename': os.path.basename(file_path),
                    'size_bytes': size,
                    'char_count': chars,
                    'line_count': lines
                })
            except Exception as e:
                print(f"⚠️ 获取文件信息失败 {os.path.basename(file_path)}: {e}")
        
        return {
            'data_directory': self.data_dir,
            'total_files': len(txt_files),
            'total_size_bytes': total_size,
            'total_size_kb': round(total_size / 1024, 2),
            'files': sorted(file_info, key=lambda x: self._natural_sort_key(x['filename']))
        }


def create_multi_file_processor(data_dir: str = "data") -> MultiFileProcessor:
    """
    创建多文件处理器实例
    
    Args:
        data_dir: 数据目录路径
        
    Returns:
        MultiFileProcessor: 配置好的处理器实例
    """
    return MultiFileProcessor(data_dir=data_dir)


if __name__ == "__main__":
    # 测试多文件处理器
    print("🧪 测试多文件处理器")
    print("=" * 50)
    
    try:
        # 创建处理器
        processor = create_multi_file_processor()
        
        # 获取文件信息
        info = processor.get_file_info()
        print(f"\n📊 数据目录信息:")
        print(f"  目录: {info['data_directory']}")
        print(f"  文件总数: {info['total_files']}")
        print(f"  总大小: {info['total_size_kb']} KB")
        
        # 显示前5个文件信息
        print(f"\n📄 文件列表 (前5个):")
        for file_info in info['files'][:5]:
            print(f"  {file_info['filename']}: {file_info['char_count']} 字符, {file_info['line_count']} 行")
        
        if len(info['files']) > 5:
            print(f"  ... 还有 {len(info['files']) - 5} 个文件")
        
        # 加载文档
        print(f"\n📚 加载文档...")
        documents = processor.load_documents()
        
        if documents:
            print(f"\n✅ 加载完成！")
            print(f"  文档数量: {len(documents)}")
            
            # 显示第一个文档示例
            first_doc = documents[0]
            print(f"\n📖 第一个文档示例:")
            print(f"  文件名: {first_doc.metadata['filename']}")
            print(f"  标题: {first_doc.metadata['title']}")
            print(f"  字符数: {first_doc.metadata['char_count']}")
            print(f"  内容预览: {first_doc.page_content[:100]}...")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
