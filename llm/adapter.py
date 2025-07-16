# backend/llm/adapter.py
import os
import importlib
from typing import Any, Optional, TYPE_CHECKING, Dict, Set, Callable

# 加载环境变量
from dotenv import load_dotenv
# 获取项目根目录的 .env 文件路径
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
load_dotenv(env_path)

# 使用 TYPE_CHECKING 来避免循环导入，并允许类型提示
if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

class LLMAdapterError(Exception):
    """LLM 适配器相关的自定义错误。"""
    pass

class LLMAdapter:
    """
    LLM适配器工具类，提供静态方法用于创建LangChain LLM实例。
    按需加载相应的依赖库。
    """
    
    # 支持的LLM提供商列表
    SUPPORTED_PROVIDERS: Set[str] = {
        "openrouter",
        "deepseek",
        "dashscope",
        "google",
        "openai",
        "xai",
        "oneapi",
    }
    
    # 提供商到构建方法名称的映射
    _PROVIDER_BUILDERS: Dict[str, str] = {
        "openrouter": "_build_openrouter_llm",
        "deepseek": "_build_deepseek_llm",
        "dashscope": "_build_dashscope_llm",
        "google": "_build_google_llm",
        "openai": "_build_openai_llm",
        "xai": "_build_xai_llm",
        "oneapi": "_build_oneapi_llm",
    }

    @staticmethod
    def get_llm(provider: str, model_name: str, temperature: float = 0.7, **kwargs) -> 'BaseChatModel':
        """
        获取指定提供商和模型的 LangChain LLM 实例。

        Args:
            provider: LLM 提供商名称 (例如 "openrouter", "deepseek", "dashscope", "google", "openai", "oneapi")。
            model_name: 具体的模型名称 (例如 "google/gemini-2.0-flash-001", "deepseek-chat", "qwen-plus-latest", "gemini-2.0-flash", "gpt-4o")。
            temperature: 温度参数，控制输出的随机性。默认为 0.7。
            **kwargs: 其他特定于提供商的配置参数。

        Returns:
            一个配置好的 LangChain BaseChatModel 实例。

        Raises:
            LLMAdapterError: 如果提供商不支持、缺少 API 密钥、无法加载依赖或实例化失败。
            ImportError: 如果无法导入所需的 LangChain 库。
        """
        if not provider:
             raise LLMAdapterError("必须提供 LLM provider 名称。")
        if not model_name:
             raise LLMAdapterError("必须提供 LLM model_name。")

        provider = provider.lower() # 统一转为小写处理
        
        # 检查提供商是否支持
        if provider not in LLMAdapter.SUPPORTED_PROVIDERS:
            supported_list = ', '.join([f'"{p}"' for p in sorted(LLMAdapter.SUPPORTED_PROVIDERS)])
            raise LLMAdapterError(f"不支持的 LLM 提供商: '{provider}'。支持的提供商: {supported_list}")
        
        # 使用映射获取构建方法
        if provider not in LLMAdapter._PROVIDER_BUILDERS:
            # 这种情况理论上不应该发生，因为我们已经检查了SUPPORTED_PROVIDERS
            raise LLMAdapterError(f"内部错误: 提供商 '{provider}' 在支持列表中但缺少构建方法")
            
        # 获取对应的构建方法名称
        builder_name = LLMAdapter._PROVIDER_BUILDERS[provider]
        
        # 动态获取并调用构建方法
        try:
            # 使用getattr获取类的静态方法
            builder_method = getattr(LLMAdapter, builder_name)
            # 调用构建方法，传递所有参数
            return builder_method(model_name, temperature, **kwargs)
        except AttributeError:
            raise LLMAdapterError(f"内部错误: 未找到构建方法 '{builder_name}'")

    @staticmethod
    def _lazy_import(module_name: str, class_name: str) -> Any:
        """按需导入模块和类。"""
        try:
            module = importlib.import_module(module_name)
            return getattr(module, class_name)
        except ImportError:
            raise ImportError(f"无法导入 '{class_name}' 从 '{module_name}'。请确保安装了必要的库 (例如 langchain-openai)。")
        except AttributeError:
             raise LLMAdapterError(f"在模块 '{module_name}' 中未找到类 '{class_name}'。")


    @staticmethod
    def _build_openrouter_llm(model_name: str, temperature: float, **kwargs) -> 'BaseChatModel':
        ChatOpenAI = LLMAdapter._lazy_import('langchain_openai', 'ChatOpenAI')
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise LLMAdapterError("无法构建 OpenRouter LLM: 环境变量 'OPENROUTER_API_KEY' 未设置或为空。")
        try:
            # 默认配置
            config = {
                "model": model_name,
                "openai_api_key": api_key,
                "base_url": "https://gateway.ai.cloudflare.com/v1/ea80a58fda8dc229df038ef64d3ce2ee/openrouter/openrouter",
                "temperature": temperature,
                "streaming": True
            }
            # 使用 kwargs 覆盖或添加额外配置
            config.update(kwargs)
            
            llm = ChatOpenAI(**config)
            return llm
        except Exception as e:
            raise LLMAdapterError(f"实例化 OpenRouter LLM (模型: {model_name}) 时出错: {e}")

    @staticmethod
    def _build_deepseek_llm(model_name: str, temperature: float, **kwargs) -> 'BaseChatModel':
        ChatDeepSeek = LLMAdapter._lazy_import('langchain_deepseek', 'ChatDeepSeek')
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise LLMAdapterError("无法构建 DeepSeek LLM: 环境变量 'DEEPSEEK_API_KEY' 未设置或为空。")
        try:
            # 默认配置
            config = {
                "model": model_name,
                "api_key": api_key,
                "temperature": temperature,
                "max_tokens": None, 
                "timeout": None,
                "max_retries": 2, 
                "streaming": True 
            }
            # 使用 kwargs 覆盖或添加额外配置
            config.update(kwargs)
            
            llm = ChatDeepSeek(**config)
            return llm
        except Exception as e:
            raise LLMAdapterError(f"实例化 DeepSeek LLM (模型: {model_name}) 时出错: {e}")

    @staticmethod
    def _build_dashscope_llm(model_name: str, temperature: float, **kwargs) -> 'BaseChatModel':
        ChatTongyi = LLMAdapter._lazy_import('langchain_community.chat_models.tongyi', 'ChatTongyi')
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise LLMAdapterError("无法构建 DashScope LLM: 环境变量 'DASHSCOPE_API_KEY' 未设置或为空。")
        try:
            # 默认配置
            config = {
                "model_name": model_name,
                "dashscope_api_key": api_key,
                "temperature": temperature,
                "streaming": True
            }
            # 使用 kwargs 覆盖或添加额外配置
            config.update(kwargs)
            
            llm = ChatTongyi(**config)
            return llm
        except Exception as e:
            raise LLMAdapterError(f"实例化 DashScope (ChatTongyi) LLM (模型: {model_name}) 时出错: {e}")

    @staticmethod
    def _build_google_llm(model_name: str, temperature: float, **kwargs) -> 'BaseChatModel':
        ChatGoogleGenerativeAI = LLMAdapter._lazy_import('langchain_google_genai', 'ChatGoogleGenerativeAI')
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise LLMAdapterError("无法构建 Google LLM: 环境变量 'GOOGLE_API_KEY' 未设置或为空。")
        try:
            # 默认配置
            config = {
                "model": model_name,
                "google_api_key": api_key,
                "temperature": temperature
            }
            # 使用 kwargs 覆盖或添加额外配置
            config.update(kwargs)
            
            llm = ChatGoogleGenerativeAI(**config)
            return llm
        except Exception as e:
            raise LLMAdapterError(f"实例化 Google LLM (模型: {model_name}) 时出错: {e}")

    @staticmethod
    def _build_openai_llm(model_name: str, temperature: float, **kwargs) -> 'BaseChatModel':
        ChatOpenAI = LLMAdapter._lazy_import('langchain_openai', 'ChatOpenAI')
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMAdapterError("无法构建 OpenAI LLM: 环境变量 'OPENAI_API_KEY' 未设置或为空。")
        base_url = os.getenv("OPENAI_BASE_URL")
        try:
            # 默认配置
            config = {
                "model": model_name,
                "openai_api_key": api_key,
                "temperature": temperature,
                "streaming": True
            }
            if base_url:
                config["base_url"] = base_url
            # 使用 kwargs 覆盖或添加额外配置
            config.update(kwargs)
            
            llm = ChatOpenAI(**config)
            return llm
        except Exception as e:
            raise LLMAdapterError(f"实例化 OpenAI LLM (模型: {model_name}) 时出错: {e}")

    @staticmethod
    def _build_xai_llm(model_name: str, temperature: float, **kwargs) -> 'BaseChatModel':
        ChatXAI = LLMAdapter._lazy_import('langchain_xai', 'ChatXAI')
        api_key = os.getenv("XAI_API_KEY")
        if not api_key:
            raise LLMAdapterError("无法构建 xAI LLM: 环境变量 'XAI_API_KEY' 未设置或为空。")
        try:
            # 默认配置
            config = {
                "model": model_name,
                "xai_api_key": api_key,
                "temperature": temperature,
            }
            # 使用 kwargs 覆盖或添加额外配置
            config.update(kwargs)
            
            llm = ChatXAI(**config)
            return llm
        except Exception as e:
            raise LLMAdapterError(f"实例化 xAI LLM (模型: {model_name}) 时出错: {e}")

    @staticmethod
    def _build_oneapi_llm(model_name: str, temperature: float, **kwargs) -> 'BaseChatModel':
        ChatOpenAI = LLMAdapter._lazy_import('langchain_openai', 'ChatOpenAI')
        api_key = os.getenv("ONEAPI_API_KEY")
        base_url = os.getenv("ONEAPI_BASE_URL")

        try:
            # 默认配置
            config = {
                "model": model_name,
                "openai_api_key": api_key,
                "base_url": base_url, # 使用从环境变量获取或指定的 base_url
                "temperature": temperature,
                "streaming": True
            }
            # 使用 kwargs 覆盖或添加额外配置
            config.update(kwargs)
            
            llm = ChatOpenAI(**config)
            return llm
        except Exception as e:
            raise LLMAdapterError(f"实例化 One API LLM (模型: {model_name}) 时出错: {e}")