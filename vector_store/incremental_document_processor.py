import os
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader
from langchain.indexes import SQLRecordManager, index
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .vector_store import VectorStoreManager


class IncrementalDocumentProcessor:
    """
    支持增量更新的文档处理器。
    基于LangChain RecordManager跟踪文档变更，实现智能增量索引。
    """
    
    def __init__(
        self,
        docs_path: str,
        vector_store_manager: VectorStoreManager,
        record_manager: Optional[SQLRecordManager] = None,
        text_splitter: Optional[RecursiveCharacterTextSplitter] = None,
        loader_kwargs: Optional[Dict[str, Any]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        supported_extensions: List[str] = None,
    ):
        """
        初始化增量文档处理器。
        
        参数:
            docs_path: 文档目录路径
            vector_store_manager: VectorStoreManager实例
            record_manager: LangChain记录管理器，用于跟踪文档状态
            text_splitter: 文本分割器，如果为None则使用默认配置
            loader_kwargs: 文档加载器参数
            chunk_size: 文档块大小
            chunk_overlap: 文档块重叠大小
            supported_extensions: 支持的文件扩展名列表
        """
        # 验证文档路径
        if not os.path.isdir(docs_path):
            raise ValueError(f"文档路径不存在或不是一个目录: {docs_path}")
        
        self.docs_path = docs_path
        self.vector_store_manager = vector_store_manager
        self.loader_kwargs = loader_kwargs or {}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 支持的文件扩展名
        self.supported_extensions = supported_extensions or ['.txt', '.md', '.pdf', '.doc', '.docx', '.html']
        
        # 设置记录管理器
        if record_manager is None:
            # 如果没有提供，创建默认的记录管理器
            cache_dir = os.path.join(os.path.dirname(docs_path), '.rag_cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            self.record_manager = SQLRecordManager(
                namespace=f"rag_docs_{os.path.basename(docs_path)}",
                db_url=f"sqlite:///{cache_dir}/record_manager.db"
            )
            self.record_manager.create_schema()
        else:
            self.record_manager = record_manager
        
        # 初始化文本分割器
        if text_splitter is None:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", " ", ""],
                keep_separator=False,
            )
        else:
            self.text_splitter = text_splitter
        
        # 文档状态缓存
        self._doc_metadata_cache: Dict[str, Dict] = {}
        print(f"✅ 增量文档处理器初始化完成，监控目录: {docs_path}")
    
    def load_documents(self) -> List[Document]:
        """从指定目录加载原始文档"""
        try:
            loader = DirectoryLoader(
                self.docs_path,
                loader_kwargs=self.loader_kwargs,
                show_progress=True, 
                use_multithreading=True,
                silent_errors=True  # 跳过无法加载的文件
            )
            documents = loader.load()
            print(f"📚 从 {self.docs_path} 加载了 {len(documents)} 个文档")
            return documents
        except Exception as e:
            print(f"❌ 加载文档时出错: {e}")
            return []
        
    def _compute_file_hash(self, file_path: str) -> str:
        """计算文件的MD5哈希值"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"计算文件哈希失败 {file_path}: {e}")
            return ""
    
    def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """获取文件元数据"""
        stat = os.stat(file_path)
        return {
            "file_path": file_path,
            "file_size": stat.st_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime),
            "hash": self._compute_file_hash(file_path),
        }
    
    def _scan_directory_changes(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        扫描目录变更。
        
        返回:
            Tuple[新增文件, 修改文件, 删除文件]
        """
        current_files = set()
        new_files = set()
        modified_files = set()
        
        # 扫描当前目录中的所有文件
        for root, dirs, files in os.walk(self.docs_path):
            for file in files:
                file_path = os.path.join(root, file)
                # 过滤支持的文件类型
                if any(file.lower().endswith(ext) for ext in self.supported_extensions):
                    current_files.add(file_path)
                    
                    # 获取当前文件元数据
                    current_metadata = self._get_file_metadata(file_path)
                    
                    if file_path not in self._doc_metadata_cache:
                        # 新文件
                        new_files.add(file_path)
                        self._doc_metadata_cache[file_path] = current_metadata
                    else:
                        # 检查是否被修改
                        cached_metadata = self._doc_metadata_cache[file_path]
                        if (current_metadata["hash"] != cached_metadata["hash"] or
                            current_metadata["modified_time"] != cached_metadata["modified_time"]):
                            modified_files.add(file_path)
                            self._doc_metadata_cache[file_path] = current_metadata
        
        # 检查删除的文件
        cached_files = set(self._doc_metadata_cache.keys())
        deleted_files = cached_files - current_files
        
        # 清理已删除文件的缓存
        for deleted_file in deleted_files:
            del self._doc_metadata_cache[deleted_file]
        
        return new_files, modified_files, deleted_files
    
    def _load_specific_files(self, file_paths: Set[str]) -> List[Document]:
        """加载指定的文件列表"""
        documents = []
        
        for file_path in file_paths:
            try:
                # 使用适当的加载器加载单个文件
                if file_path.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "file_hash": self._compute_file_hash(file_path),
                            "load_time": datetime.now().isoformat(),
                        }
                    )
                    documents.append(doc)
                # 可以扩展支持更多文件类型
                else:
                    print(f"暂不支持的文件类型: {file_path}")
                    
            except Exception as e:
                print(f"加载文件失败 {file_path}: {e}")
                
        return documents
    
    def _create_document_ids(self, documents: List[Document]) -> List[str]:
        """为文档创建唯一ID"""
        doc_ids = []
        for i, doc in enumerate(documents):
            # 基于文件路径和内容哈希创建唯一ID
            source = doc.metadata.get("source", "unknown")
            content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
            doc_id = f"{Path(source).stem}_{content_hash}_{i}"
            doc_ids.append(doc_id)
        return doc_ids
    
    async def incremental_index(self, cleanup: str = "incremental") -> Dict[str, Any]:
        """
        执行增量索引更新。
        
        参数:
            cleanup: 清理模式
                - "incremental": 只处理变更的文档
                - "full": 完全重建索引
                - None: 只添加新文档，不删除旧文档
        
        返回:
            包含更新统计信息的字典
        """
        print("🔍 扫描文档变更...")
        new_files, modified_files, deleted_files = self._scan_directory_changes()
        
        stats = {
            "new_files": len(new_files),
            "modified_files": len(modified_files), 
            "deleted_files": len(deleted_files),
            "total_processed": 0,
        }
        
        # 处理新增和修改的文件
        changed_files = new_files | modified_files
        if changed_files:
            print(f"📚 处理 {len(changed_files)} 个变更文件...")
            
            # 加载变更的文档
            documents = self._load_specific_files(changed_files)
            
            if documents:
                # 分割文档
                split_docs = self.text_splitter.split_documents(documents)
                
                # 创建文档ID
                doc_ids = self._create_document_ids(split_docs)
                
                # 使用LangChain的index函数进行增量更新
                result = await index(
                    doc_ids,
                    split_docs,
                    record_manager=self.record_manager,
                    vector_store=self.vector_store_manager._vector_store,
                    cleanup=cleanup,
                    source_id_key="source",
                )
                
                stats["total_processed"] = len(split_docs)
                stats["index_result"] = result
                
                print(f"✅ 成功处理 {len(split_docs)} 个文档块")
        
        # 处理删除的文件
        if deleted_files and cleanup == "incremental":
            print(f"🗑️ 清理 {len(deleted_files)} 个已删除文件的索引...")
            # TODO: 实现删除逻辑
            # 需要根据source字段从记录管理器中删除相关记录
        
        print("🎉 增量索引更新完成!")
        return stats
    
    def full_reindex(self) -> Dict[str, Any]:
        """完全重建索引"""
        print("🔄 执行完全重建索引...")
        
        # 加载所有文档
        documents = self.load_documents()
        
        if not documents:
            return {"total_processed": 0, "message": "没有找到文档"}
        
        # 分割文档
        split_docs = self.text_splitter.split_documents(documents)
        
        # 创建文档ID
        doc_ids = self._create_document_ids(split_docs)
        
        # 完全重建索引
        result = index(
            doc_ids,
            split_docs,
            record_manager=self.record_manager,
            vector_store=self.vector_store_manager._vector_store,
            cleanup="full",
            source_id_key="source",
        )
        
        # 更新元数据缓存
        self._update_metadata_cache()
        
        stats = {
            "total_processed": len(split_docs),
            "index_result": result,
        }
        
        print(f"✅ 完全重建完成，处理了 {len(split_docs)} 个文档块")
        return stats
    
    def _update_metadata_cache(self):
        """更新元数据缓存"""
        self._doc_metadata_cache.clear()
        
        for root, dirs, files in os.walk(self.docs_path):
            for file in files:
                file_path = os.path.join(root, file)
                if any(file.lower().endswith(ext) for ext in self.supported_extensions):
                    self._doc_metadata_cache[file_path] = self._get_file_metadata(file_path)
    
    def get_index_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        try:
            # 从记录管理器获取统计信息
            total_records = len(self.record_manager.list_keys())
            
            return {
                "total_documents": len(self._doc_metadata_cache),
                "total_chunks": total_records,
                "cache_size": len(self._doc_metadata_cache),
            }
        except Exception as e:
            return {"error": str(e)}
    
    def update_documents(self) -> Dict[str, Any]:
        """
        同步版本的增量文档更新方法
        """
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._check_and_update_documents())
        finally:
            loop.close()
    
    async def _check_and_update_documents(self) -> Dict[str, Any]:
        """
        检查并执行增量文档更新（核心逻辑）
        
        返回:
            更新统计信息字典
        """
        try:
            print("🔍 开始检查文档变更...")
            stats = await self.incremental_index(cleanup="incremental")
            
            if stats["new_files"] > 0 or stats["modified_files"] > 0:
                print(f"✅ 检测到变更并已更新: 新增{stats['new_files']}个，修改{stats['modified_files']}个文件")
                stats["status"] = "success"
                stats["message"] = f"成功更新 {stats['total_processed']} 个文档块"
            else:
                print("ℹ️ 没有检测到文档变更")
                stats["status"] = "no_changes"
                stats["message"] = "没有检测到文档变更"
            
            return stats
            
        except Exception as e:
            print(f"❌ 增量更新失败: {e}")
            return {
                "status": "error",
                "message": f"增量更新失败: {str(e)}"
            }
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """
        获取增量处理器的综合状态信息
        """
        try:
            # 基础统计
            index_stats = self.get_index_stats()
            
            # 文件监控统计
            file_stats = {
                "monitored_directory": self.docs_path,
                "supported_extensions": self.supported_extensions,
                "cached_files_count": len(self._doc_metadata_cache),
            }
            
            # 配置信息
            config_info = {
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "record_manager_namespace": self.record_manager.namespace,
            }
            
            # 合并所有信息
            comprehensive_status = {
                **index_stats,
                **file_stats,
                **config_info
            }
            
            return comprehensive_status
            
        except Exception as e:
            return {"comprehensive_status_error": str(e)}
    
    @classmethod
    def create_with_vector_store(
        cls,
        docs_path: str,
        vector_store_manager: VectorStoreManager,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        **kwargs
    ) -> "IncrementalDocumentProcessor":
        """
        便捷方法：使用VectorStoreManager创建增量处理器
        
        参数:
            docs_path: 文档目录路径
            vector_store_manager: VectorStoreManager实例
            chunk_size: 文档块大小
            chunk_overlap: 文档块重叠大小
            **kwargs: 其他参数
        
        返回:
            IncrementalDocumentProcessor实例
        """
        return cls(
            docs_path=docs_path,
            vector_store_manager=vector_store_manager,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **kwargs
        )
