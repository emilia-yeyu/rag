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
    }
    
    # 提供商到构建方法名称的映射
    _PROVIDER_BUILDERS: Dict[str, str] = {
        "openai": "_build_openai_embedding",
        "dashscope": "_build_dashscope_embedding",
        "google": "_build_google_embedding",
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