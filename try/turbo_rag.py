#!/usr/bin/env python3
"""
优化版RAG系统 - 集成TurboRAG思路
在保持原有架构的基础上，添加KV缓存优化功能
"""

import os
import time
from typing import Dict, Any, List
import argparse

# 导入原有RAG系统
from rag import SimpleRAG

# 导入优化模块
from kv_cache_optimizer import create_optimized_rag, KVCacheManager


class TurboRAG:
    """
    TurboRAG - 优化版RAG系统
    基于现有SimpleRAG，集成KV缓存优化
    """
    
    def __init__(self, document_path: str = "1.txt", 
                 enable_optimization: bool = True,
                 local_model_name: str = None):
        """
        初始化TurboRAG系统
        
        Args:
            document_path: 文档文件路径
            enable_optimization: 是否启用KV缓存优化
            local_model_name: 本地模型名称（可选，用于KV缓存生成）
        """
        print(f"🚀 初始化TurboRAG系统...")
        print(f"📄 文档路径: {document_path}")
        print(f"⚡ 优化模式: {'启用' if enable_optimization else '禁用'}")
        
        self.document_path = document_path
        self.enable_optimization = enable_optimization
        
        # 初始化原始RAG系统
        print(f"🔧 初始化基础RAG系统...")
        self.base_rag = SimpleRAG(document_path)
        
        # 如果启用优化，初始化优化组件
        self.optimized_rag = None
        if enable_optimization:
            print(f"⚡ 初始化优化组件...")
            self.optimized_rag = create_optimized_rag(
                self.base_rag, 
                model_name=local_model_name
            )
        
        print(f"✅ TurboRAG系统就绪！")
    
    def preprocess_for_optimization(self) -> Dict[str, Any]:
        """
        预处理文档以启用优化
        这是一次性操作，会为所有文档块生成KV缓存
        
        Returns:
            Dict[str, Any]: 预处理结果统计
        """
        if not self.optimized_rag:
            return {
                "success": False,
                "message": "优化功能未启用，请在初始化时设置enable_optimization=True"
            }
        
        print(f"🔄 开始预处理文档以生成KV缓存...")
        print(f"⏰ 这是一次性操作，可能需要几分钟时间...")
        
        start_time = time.time()
        
        try:
            processed_count = self.optimized_rag.preprocess_knowledge_base()
            
            # 获取缓存统计
            cache_stats = self.optimized_rag.cache_manager.get_cache_stats()
            
            preprocessing_time = time.time() - start_time
            
            result = {
                "success": True,
                "processed_chunks": processed_count,
                "preprocessing_time": f"{preprocessing_time:.2f}秒",
                "cache_stats": cache_stats,
                "message": "预处理完成！后续查询将享受显著的性能提升。"
            }
            
            print(f"✅ 预处理完成！")
            print(f"📊 处理的文档块: {processed_count}")
            print(f"⏱️ 预处理耗时: {result['preprocessing_time']}")
            print(f"💾 缓存大小: {cache_stats['total_size_mb']:.2f} MB")
            print(f"🚀 现在可以享受快速查询了！")
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "message": f"预处理失败: {str(e)}"
            }
            print(f"❌ 预处理失败: {e}")
            return error_result
    
    def query(self, question: str, show_sources: bool = False, 
              force_optimized: bool = None) -> Dict[str, Any]:
        """
        智能查询方法
        
        Args:
            question: 用户问题
            show_sources: 是否显示来源信息
            force_optimized: 强制使用优化模式（None=自动选择）
            
        Returns:
            Dict[str, Any]: 查询结果
        """
        # 决定使用哪种查询方式
        use_optimized = self._should_use_optimized(force_optimized)
        
        if use_optimized and self.optimized_rag:
            print(f"⚡ 使用优化模式查询")
            return self.optimized_rag.optimized_query(question, show_sources)
        else:
            print(f"🔧 使用基础模式查询")
            result = self.base_rag.query(question, show_sources)
            # 添加优化信息
            result["optimization_used"] = False
            result["cache_hit_rate"] = "N/A"
            return result
    
    def _should_use_optimized(self, force_optimized: bool = None) -> bool:
        """决定是否使用优化模式"""
        if force_optimized is not None:
            return force_optimized
        
        # 自动决策：如果启用了优化且有缓存则使用
        if not self.enable_optimization or not self.optimized_rag:
            return False
        
        # 检查是否有可用的缓存
        cache_stats = self.optimized_rag.cache_manager.get_cache_stats()
        return cache_stats['total_caches'] > 0
    
    def benchmark(self, questions: List[str] = None, rounds: int = 3) -> Dict[str, Any]:
        """
        性能基准测试
        
        Args:
            questions: 测试问题列表
            rounds: 测试轮数
            
        Returns:
            Dict[str, Any]: 基准测试结果
        """
        if questions is None:
            questions = [
                "一微半导体是什么公司？",
                "员工迟到会有什么处罚？",
                "公司的核心价值观是什么？",
                "公司有多少员工？",
                "公司的考勤时间是怎样的？"
            ]
        
        print(f"🏁 开始性能基准测试")
        print(f"📝 测试问题数: {len(questions)}")
        print(f"🔄 测试轮数: {rounds}")
        print(f"="*60)
        
        # 测试基础模式
        print(f"🔧 测试基础模式...")
        base_times = []
        for round_num in range(rounds):
            round_start = time.time()
            for i, question in enumerate(questions):
                result = self.base_rag.query(question)
                print(f"  轮次{round_num+1} 问题{i+1}: {result['response_time']}")
            round_time = time.time() - round_start
            base_times.append(round_time)
        
        base_avg = sum(base_times) / len(base_times)
        base_per_query = base_avg / len(questions)
        
        # 测试优化模式（如果可用）
        optimized_times = []
        optimized_avg = 0
        optimized_per_query = 0
        optimization_available = self._should_use_optimized()
        
        if optimization_available:
            print(f"\n⚡ 测试优化模式...")
            for round_num in range(rounds):
                round_start = time.time()
                for i, question in enumerate(questions):
                    result = self.query(question, force_optimized=True)
                    print(f"  轮次{round_num+1} 问题{i+1}: {result['response_time']} (缓存命中: {result['cache_hit_rate']})")
                round_time = time.time() - round_start
                optimized_times.append(round_time)
            
            optimized_avg = sum(optimized_times) / len(optimized_times)
            optimized_per_query = optimized_avg / len(questions)
        
        # 计算性能提升
        speedup = base_per_query / optimized_per_query if optimized_per_query > 0 else 1.0
        
        # 准备结果
        result = {
            "test_config": {
                "questions_count": len(questions),
                "rounds": rounds,
                "optimization_available": optimization_available
            },
            "base_mode": {
                "total_avg_time": f"{base_avg:.2f}秒",
                "per_query_avg_time": f"{base_per_query:.2f}秒",
                "all_round_times": [f"{t:.2f}秒" for t in base_times]
            },
            "optimized_mode": {
                "total_avg_time": f"{optimized_avg:.2f}秒" if optimization_available else "N/A",
                "per_query_avg_time": f"{optimized_per_query:.2f}秒" if optimization_available else "N/A",
                "all_round_times": [f"{t:.2f}秒" for t in optimized_times] if optimization_available else [],
                "average_speedup": f"{speedup:.2f}x"
            }
        }
        
        # 打印总结
        print(f"\n" + "="*60)
        print(f"📊 基准测试结果总结")
        print(f"="*60)
        print(f"🔧 基础模式平均查询时间: {base_per_query:.2f}秒")
        if optimization_available:
            print(f"⚡ 优化模式平均查询时间: {optimized_per_query:.2f}秒")
            print(f"🚀 性能提升: {speedup:.2f}倍")
            print(f"⏱️ 时间节省: {((base_per_query - optimized_per_query) / base_per_query * 100):.1f}%")
        else:
            print(f"⚠️ 优化模式不可用 - 请先运行预处理")
        
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态信息"""
        base_info = {
            "document_path": self.document_path,
            "base_rag_initialized": self.base_rag is not None,
            "vector_store_size": len(self.base_rag.vector_store) if self.base_rag else 0
        }
        
        if self.optimized_rag:
            cache_stats = self.optimized_rag.cache_manager.get_cache_stats()
            optimization_info = {
                "optimization_enabled": True,
                "cache_stats": cache_stats,
                "optimization_ready": cache_stats['total_caches'] > 0
            }
        else:
            optimization_info = {
                "optimization_enabled": False,
                "cache_stats": None,
                "optimization_ready": False
            }
        
        return {**base_info, **optimization_info}
    
    def demo(self, use_optimization: bool = None):
        """
        演示功能
        
        Args:
            use_optimization: 是否使用优化模式（None=自动选择）
        """
        print("\n" + "="*60)
        print("🎯 TurboRAG系统演示")
        print("="*60)
        
        # 显示系统状态
        status = self.get_system_status()
        print(f"📊 系统状态:")
        print(f"  📄 文档块数量: {status['vector_store_size']}")
        print(f"  ⚡ 优化功能: {'启用' if status['optimization_enabled'] else '禁用'}")
        if status['optimization_enabled']:
            print(f"  💾 缓存数量: {status['cache_stats']['total_caches']}")
            print(f"  📊 缓存大小: {status['cache_stats']['total_size_mb']:.2f} MB")
        
        # 演示查询
        questions = [
            "一微半导体是什么公司？",
            "员工迟到会有什么处罚？",
            "公司的核心价值观是什么？",
            "公司有多少员工？",
            "公司的考勤时间是怎样的？"
        ]
        
        for i, question in enumerate(questions, 1):
            print(f"\n🔍 问题{i}: {question}")
            print("-" * 40)
            
            result = self.query(question, force_optimized=use_optimization)
            
            print(f"💬 回答: {result['answer']}")
            print(f"⏱️ 耗时: {result['response_time']}")
            if result.get('optimization_used'):
                print(f"⚡ 缓存命中率: {result['cache_hit_rate']}")
        
        print(f"\n✅ 演示完成！")
    
    def interactive(self):
        """交互模式"""
        print("\n" + "="*60)
        print("💬 TurboRAG交互模式")
        print("🔧 命令: 'status' - 查看状态, 'benchmark' - 性能测试, 'quit' - 退出")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n🤔 请输入问题或命令: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', '退出']:
                    print("👋 再见！")
                    break
                elif user_input.lower() == 'status':
                    status = self.get_system_status()
                    print(f"\n📊 系统状态:")
                    for key, value in status.items():
                        print(f"  {key}: {value}")
                elif user_input.lower() == 'benchmark':
                    self.benchmark()
                else:
                    # 普通查询
                    result = self.query(user_input, show_sources=True)
                    print(f"\n💬 回答:\n{result['answer']}")
                    print(f"\n⏱️ 耗时: {result['response_time']}")
                    if result.get('optimization_used'):
                        print(f"⚡ 缓存命中率: {result['cache_hit_rate']}")
                    
                    # 显示来源
                    if result.get('sources'):
                        print(f"\n📚 相关文档片段:")
                        for i, source in enumerate(result['sources'], 1):
                            print(f"  {i}. {source['content']}")
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TurboRAG - 优化版RAG系统')
    parser.add_argument('--document', '-d', type=str, default='1.txt', 
                       help='文档文件路径')
    parser.add_argument('--disable-optimization', action='store_true',
                       help='禁用优化功能')
    parser.add_argument('--local-model', type=str, default=None,
                       help='本地模型名称（用于KV缓存生成）')
    parser.add_argument('--preprocess', action='store_true',
                       help='执行预处理（生成KV缓存）')
    parser.add_argument('--benchmark', action='store_true',
                       help='运行性能基准测试')
    parser.add_argument('--mode', choices=['demo', 'interactive'], default='interactive',
                       help='运行模式')
    
    args = parser.parse_args()
    
    try:
        # 初始化TurboRAG系统
        turbo_rag = TurboRAG(
            document_path=args.document,
            enable_optimization=not args.disable_optimization,
            local_model_name=args.local_model
        )
        
        # 如果需要预处理
        if args.preprocess:
            print(f"\n🔄 执行预处理操作...")
            result = turbo_rag.preprocess_for_optimization()
            if result['success']:
                print(f"✅ 预处理成功！")
            else:
                print(f"❌ 预处理失败: {result['message']}")
                return
        
        # 如果需要基准测试
        if args.benchmark:
            print(f"\n🏁 执行性能基准测试...")
            turbo_rag.benchmark()
            return
        
        # 选择运行模式
        if args.mode == "demo":
            turbo_rag.demo()
        else:
            turbo_rag.interactive()
            
    except Exception as e:
        print(f"❌ 系统错误: {e}")


if __name__ == "__main__":
    main()