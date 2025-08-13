# backend/embedding/adapter.py
import os
import importlib
from typing import Any, Optional, TYPE_CHECKING, Dict, Set, Callable, List

# 使用 TYPE_CHECKING 来避免循环导入，并允许类型提示
if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings

class EmbeddingAdapterError(Exception):
    """嵌入模型适配器相关的自定义错误。"""
    pass

class EmbeddingAdapter:
    """
    嵌入模型适配器工具类，提供静态方法用于创建LangChain Embedding实例。
    按需加载相应的依赖库。
    """
    
    # 支持的嵌入模型提供商列表
    SUPPORTED_PROVIDERS: Set[str] = {
        "openai",
        "dashscope",
        "google",
        "bge",  # 添加 bge 支持
    }
    
    # 提供商到构建方法名称的映射
    _PROVIDER_BUILDERS: Dict[str, str] = {
        "openai": "_build_openai_embedding",
        "dashscope": "_build_dashscope_embedding",
        "google": "_build_google_embedding",
        "bge": "_build_bge_embedding",  # 添加 bge 构建方法
    }

    @staticmethod
    def get_embedding(provider: str, model_name: Optional[str] = None) -> 'Embeddings':
        """
        获取指定提供商和模型的 LangChain Embedding 实例。

        Args:
            provider: 嵌入模型提供商名称 (例如 "openai", "dashscope", "google")。
            model_name: 具体的模型名称，如果为None则使用默认模型
                        (例如 "text-embedding-3-small", "text-embedding-v1", "embedding-001")。

        Returns:
            一个配置好的 LangChain Embeddings 实例。

        Raises:
            EmbeddingAdapterError: 如果提供商不支持、缺少 API 密钥、无法加载依赖或实例化失败。
            ImportError: 如果无法导入所需的 LangChain 库。
        """
        if not provider:
             raise EmbeddingAdapterError("必须提供 Embedding provider 名称。")

        provider = provider.lower() # 统一转为小写处理
        
        # 检查提供商是否支持
        if provider not in EmbeddingAdapter.SUPPORTED_PROVIDERS:
            supported_list = ', '.join([f'"{p}"' for p in sorted(EmbeddingAdapter.SUPPORTED_PROVIDERS)])
            raise EmbeddingAdapterError(f"不支持的嵌入模型提供商: '{provider}'。支持的提供商: {supported_list}")
        
        # 使用映射获取构建方法
        if provider not in EmbeddingAdapter._PROVIDER_BUILDERS:
            # 这种情况理论上不应该发生，因为我们已经检查了SUPPORTED_PROVIDERS
            raise EmbeddingAdapterError(f"内部错误: 提供商 '{provider}' 在支持列表中但缺少构建方法")
            
        # 获取对应的构建方法名称
        builder_name = EmbeddingAdapter._PROVIDER_BUILDERS[provider]
        
        # 动态获取并调用构建方法
        try:
            # 使用getattr获取类的静态方法
            builder_method = getattr(EmbeddingAdapter, builder_name)
            # 调用构建方法
            return builder_method(model_name)
        except AttributeError:
            raise EmbeddingAdapterError(f"内部错误: 未找到构建方法 '{builder_name}'")

    @staticmethod
    def _lazy_import(module_name: str, class_name: str) -> Any:
        """按需导入模块和类。"""
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except ImportError:
            raise ImportError(f"无法导入 '{class_name}' 从 '{module_name}'。请确保安装了必要的库 (例如 langchain-openai)。")
        except AttributeError:
             raise EmbeddingAdapterError(f"在模块 '{module_name}' 中未找到类 '{class_name}'。")

    @staticmethod
    def _build_openai_embedding(model_name: Optional[str] = None) -> 'Embeddings':
        """
        构建 OpenAI 的 LangChain Embedding 实例。
        支持通过环境变量自定义 base_url。
        
        Args:
            model_name: OpenAI嵌入模型名称，如果为None则使用默认模型"text-embedding-3-small"
        
        Returns:
            配置好的OpenAI嵌入模型实例
        """
        # 1. 按需导入
        OpenAIEmbeddings = EmbeddingAdapter._lazy_import('langchain_openai', 'OpenAIEmbeddings')

        # 2. 获取 API 密钥
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EmbeddingAdapterError("无法构建 OpenAI Embedding: 环境变量 'OPENAI_API_KEY' 未设置或为空。")
          
        # 3. 检查是否有自定义的 base_url
        base_url = os.getenv("OPENAI_BASE_URL")
        
        # 4. 实例化
        try:
            kwargs = {
                "model": model_name if model_name else "text-embedding-3-small",
                "openai_api_key": api_key,
            }
            
            # 如果设置了自定义 base_url，则添加到参数中
            if base_url:
                kwargs["base_url"] = base_url
                print(f"使用自定义 OpenAI API 端点: {base_url}")
            
            embedding = OpenAIEmbeddings(**kwargs)
            
            print(f"成功构建 OpenAI Embedding: 模型='{kwargs['model']}'")
            return embedding
        except Exception as e:
            raise EmbeddingAdapterError(f"实例化 OpenAI Embedding (模型: {model_name}) 时出错: {e}")
            
    @staticmethod
    def _build_dashscope_embedding(model_name: Optional[str] = None) -> 'Embeddings':
        """
        构建 阿里云百炼(DashScope) 的 LangChain Embedding 实例。
        
        Args:
            model_name: DashScope嵌入模型名称，如果为None则使用默认模型"text-embedding-v1"
        
        Returns:
            配置好的DashScope嵌入模型实例
        """
        # 1. 按需导入
        DashScopeEmbeddings = EmbeddingAdapter._lazy_import('langchain_community.embeddings', 'DashScopeEmbeddings')

        # 2. 获取 API 密钥
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise EmbeddingAdapterError("无法构建 DashScope Embedding: 环境变量 'DASHSCOPE_API_KEY' 未设置或为空。")
        
        # 3. 实例化
        try:
            model = model_name if model_name else "text-embedding-v3"
            embedding = DashScopeEmbeddings(
                model=model,
                dashscope_api_key=api_key
            )
            
            print(f"成功构建 DashScope Embedding: 模型='{model}'")
            return embedding
        except Exception as e:
            raise EmbeddingAdapterError(f"实例化 DashScope Embedding (模型: {model_name}) 时出错: {e}")
            
    @staticmethod
    def _build_google_embedding(model_name: Optional[str] = None) -> 'Embeddings':
        """
        构建 Google Generative AI 的 LangChain Embedding 实例。
        
        Args:
            model_name: Google嵌入模型名称，如果为None则使用默认模型"embedding-001"
        
        Returns:
            配置好的Google嵌入模型实例
        """
        # 1. 按需导入
        GoogleGenerativeAIEmbeddings = EmbeddingAdapter._lazy_import('langchain_google_genai', 'GoogleGenerativeAIEmbeddings')

        # 2. 获取 API 密钥
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EmbeddingAdapterError("无法构建 Google Embedding: 环境变量 'GOOGLE_API_KEY' 未设置或为空。")
        
        # 3. 实例化
        try:
            model = model_name if model_name else "text-embedding-004"
            embedding = GoogleGenerativeAIEmbeddings(
                model=model,
                google_api_key=api_key
            )
            
            print(f"成功构建 Google Embedding: 模型='{model}'")
            return embedding
        except Exception as e:
            raise EmbeddingAdapterError(f"实例化 Google Embedding (模型: {model_name}) 时出错: {e}")

    @staticmethod
    def _get_local_model_path(model_name: str) -> Optional[str]:
        """
        获取本地模型路径，如果不存在则返回None。
        
        Args:
            model_name: 模型名称，例如 "BAAI/bge-large-zh-v1.5"
        
        Returns:
            本地模型路径或None
        """
        # 将huggingface模型名称转换为本地缓存目录名称
        cache_name = model_name.replace("/", "--")
        cache_name = f"models--{cache_name}"
        
        # 获取项目根目录并构建模型路径
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 回到 RAG 目录
        model_base_path = os.path.join(script_dir, "models", "embeddings", cache_name)
        
        # 检查是否存在main引用文件
        main_ref_path = os.path.join(model_base_path, "refs", "main")
        if os.path.exists(main_ref_path):
            # 读取main引用指向的snapshot
            try:
                with open(main_ref_path, 'r') as f:
                    snapshot_hash = f.read().strip()
                
                # 构建完整的snapshot路径
                snapshot_path = os.path.join(model_base_path, "snapshots", snapshot_hash)
                
                # 验证snapshot路径存在且包含必要的配置文件
                if os.path.exists(snapshot_path) and os.path.exists(os.path.join(snapshot_path, "config.json")):
                    return snapshot_path
            except Exception:
                pass
        
        return None

    @staticmethod
    def _build_bge_embedding(model_name: Optional[str] = None) -> 'Embeddings':
        """
        构建 BGE (BAAI General Embedding) 的 LangChain Embedding 实例。
        优先使用本地模型，如果不存在则使用 HuggingFace 在线模型。
        
        Args:
            model_name: BGE嵌入模型名称，如果为None则使用默认模型"BAAI/bge-large-zh-v1.5"
        
        Returns:
            配置好的BGE嵌入模型实例
        """
        # 1. 按需导入
        HuggingFaceEmbeddings = EmbeddingAdapter._lazy_import('langchain_huggingface', 'HuggingFaceEmbeddings')

        # 2. 设置模型名称
        model = model_name if model_name else "BAAI/bge-large-zh-v1.5"
        
        # 3. 尝试获取本地模型路径
        local_model_path = EmbeddingAdapter._get_local_model_path(model)
        actual_model_path = local_model_path if local_model_path else model
        ## 3. 尝试获取本地模型路径
        #local_model_path = EmbeddingAdapter._get_local_model_path(model)
        #actual_model_path = local_model_path if local_model_path else model

        # 获取项目根目录并设置模型缓存目录
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 回到 RAG 目录
        cache_dir = os.path.join(script_dir, "models", "embeddings")
        os.makedirs(cache_dir, exist_ok=True)
        
        # 4. 实例化
        try:
            # BGE 模型的推荐配置
            encode_kwargs = {
                'normalize_embeddings': True,  # BGE 推荐归一化
                'batch_size': 32,  # 合适的批处理大小
            }
            
            # 模型配置
            model_kwargs = {
                'device': 'cpu',  # 可以根据需要改为 'cuda'
                'trust_remote_code': True,  # BGE 模型需要此选项
            }
            
            embedding = HuggingFaceEmbeddings(
                model_name=actual_model_path,#model
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                show_progress=True,  # 显示下载进度
                cache_folder=cache_dir if not local_model_path else None,  # 本地模型不需要额外缓存
            )
            
            if local_model_path:
                print(f"✅ 成功构建 BGE Embedding (本地模型): 路径='{local_model_path}'")
            else:
                print(f"✅ 成功构建 BGE Embedding (在线模型): 模型='{model}'")
                print(f"📁 模型缓存目录: {cache_dir}")
                print(f"注意: 首次使用会下载模型到指定目录，请耐心等待")
            
            return embedding
        except Exception as e:
            error_msg = f"实例化 BGE Embedding 时出错: {e}"
            if local_model_path:
                error_msg += f"\n本地模型路径: {local_model_path}"
            else:
                error_msg += f"\n在线模型: {model}"
            raise EmbeddingAdapterError(error_msg)




"""     @staticmethod
    def _get_local_model_path(model_name: str) -> Optional[str]:
        '''
        获取本地模型路径，如果不存在则返回None。
        
        Args:
            model_name: 模型名称，例如 "BAAI/bge-large-zh-v1.5"
        
        Returns:
            本地模型路径或None
        '''
        # 将huggingface模型名称转换为本地缓存目录名称
        cache_name = model_name.replace("/", "--")
        cache_name = f"models--{cache_name}"
        
        # 获取项目根目录并构建模型路径
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 回到 RAG 目录
        model_base_path = os.path.join(script_dir, "models", "embeddings", cache_name)
        
        # 检查是否存在main引用文件
        main_ref_path = os.path.join(model_base_path, "refs", "main")
        if os.path.exists(main_ref_path):
            # 读取main引用指向的snapshot
            try:
                with open(main_ref_path, 'r') as f:
                    snapshot_hash = f.read().strip()
                
                # 构建完整的snapshot路径
                snapshot_path = os.path.join(model_base_path, "snapshots", snapshot_hash)
                
                # 验证snapshot路径存在且包含必要的配置文件
                if os.path.exists(snapshot_path) and os.path.exists(os.path.join(snapshot_path, "config.json")):
                    return snapshot_path
            except Exception:
                pass
        
        return None

    @staticmethod
    def _build_bge_embedding(model_name: Optional[str] = None) -> 'Embeddings':
        '''
        构建 BGE (BAAI General Embedding) 的 LangChain Embedding 实例。
        优先使用本地微调模型，如果不存在则使用 HuggingFace 在线模型。
        
        Args:
            model_name: BGE嵌入模型名称，如果为None则使用默认模型"BAAI/bge-large-zh-v1.5"
        
        Returns:
            配置好的BGE嵌入模型实例
        '''
        # 1. 按需导入
        HuggingFaceEmbeddings = EmbeddingAdapter._lazy_import('langchain_huggingface', 'HuggingFaceEmbeddings')

        # 2. 设置模型名称
        model = model_name if model_name else "BAAI/bge-large-zh-v1.5"
        
        # 3. 优先检查微调模型路径
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 回到 RAG 目录
        fine_tuned_model_path = os.path.join(script_dir, "models", "ft_BAAI_bge-large-zh-v1.5")
        
        # 检查微调模型是否存在
        if os.path.exists(fine_tuned_model_path) and os.path.exists(os.path.join(fine_tuned_model_path, "config.json")):
            actual_model_path = fine_tuned_model_path
            is_fine_tuned = True
            print(f"🎯 发现微调模型: {fine_tuned_model_path}")
        else:
            # 降级到原来的逻辑：尝试获取本地模型路径
            local_model_path = EmbeddingAdapter._get_local_model_path(model)
            actual_model_path = local_model_path if local_model_path else model
            is_fine_tuned = False
        
        # 设置模型缓存目录（微调模型不需要缓存）
        cache_dir = os.path.join(script_dir, "models", "embeddings")
        if not is_fine_tuned:
            os.makedirs(cache_dir, exist_ok=True)
        
        # 4. 实例化
        try:
            # BGE 模型的推荐配置
            encode_kwargs = {
                'normalize_embeddings': True,  # BGE 推荐归一化
                'batch_size': 32,  # 合适的批处理大小
            }
            
            # 模型配置
            model_kwargs = {
                'device': 'cpu',  # 可以根据需要改为 'cuda'
                'trust_remote_code': True,  # BGE 模型需要此选项
            }
            
            embedding = HuggingFaceEmbeddings(
                model_name=actual_model_path,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                show_progress=True,  # 显示下载进度
                cache_folder=cache_dir if not is_fine_tuned else None,  # 微调模型不需要额外缓存
            )
            
            if is_fine_tuned:
                print(f"✅ 成功构建 BGE Embedding (微调模型): 路径='{actual_model_path}'")
                print(f"🔥 使用您训练的微调模型！")
            elif actual_model_path != model:
                print(f"✅ 成功构建 BGE Embedding (本地模型): 路径='{actual_model_path}'")
            else:
                print(f"✅ 成功构建 BGE Embedding (在线模型): 模型='{model}'")
                print(f"📁 模型缓存目录: {cache_dir}")
                print(f"注意: 首次使用会下载模型到指定目录，请耐心等待")
            
            return embedding
        except Exception as e:
            error_msg = f"实例化 BGE Embedding 时出错: {e}"
            if is_fine_tuned:
                error_msg += f"\n微调模型路径: {actual_model_path}"
            elif actual_model_path != model:
                error_msg += f"\n本地模型路径: {actual_model_path}"
            else:
                error_msg += f"\n在线模型: {model}"
            raise EmbeddingAdapterError(error_msg) """
