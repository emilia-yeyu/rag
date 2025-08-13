#!/usr/bin/env python3
"""
测试可配置RAG系统的脚本
"""
import os
import sys
import time

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_sql_components():
    """测试SQL组件"""
    print("🧪 测试SQL组件...")
    
    try:
        from sql_retriever import SQLRetriever, DatabaseManager
        
        # 测试数据库管理器
        print("\n1. 测试数据库管理器...")
        db_manager = DatabaseManager()
        
        # 获取统计信息
        stats = db_manager.execute_query("SELECT COUNT(*) as total FROM guests")
        total_count = stats[0]['total'] if stats else 0
        print(f"   ✅ 数据库记录总数: {total_count}")
        
        # 测试SQL检索器
        print("\n2. 测试SQL检索器...")
        sql_retriever = SQLRetriever()
        
        # 获取统计信息
        db_stats = sql_retriever.get_stats()
        print(f"   📊 数据库统计: {db_stats}")
        
        print(f"✅ SQL组件测试完成")
        
    except Exception as e:
        print(f"❌ SQL组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_retrieval_configs():
    """测试不同的检索配置"""
    print("\n🧪 测试检索配置...")
    
    try:
        from retrieval_config import list_configs, get_config
        
        print("📋 可用配置:")
        configs = list_configs()
        for name, description in configs.items():
            print(f"   • {name}: {description}")
        
        # 测试获取配置
        print("\n测试配置加载:")
        for config_name in ['semantic', 'keyword', 'structured', 'comprehensive']:
            try:
                config = get_config(config_name)
                print(f"   ✅ {config_name}: {config.get_description()}")
                print(f"      启用方法: {', '.join(config.get_enabled_methods())}")
            except Exception as e:
                print(f"   ❌ {config_name}: {e}")
        
        print(f"✅ 检索配置测试完成")
        
    except Exception as e:
        print(f"❌ 检索配置测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_different_modes():
    """测试不同检索模式"""
    print("\n🧪 测试不同检索模式...")
    
    try:
        from configurable_rag import ConfigurableRAG
        
        # 测试配置列表
        test_configs = [
            'semantic',          # 仅向量检索
            'keyword',           # 仅BM25检索
            'structured',        # 仅SQL检索
            'semantic_keyword',  # 向量+BM25
            'comprehensive'      # 全部混合
        ]
        
        # 测试查询
        test_queries = [
            "贾宝玉多大年纪？",           # 适合SQL检索
            "红楼梦的主要人物关系？",     # 适合向量检索
            "荣国府的联系方式？",         # 适合SQL检索
            "林黛玉的性格特点？"          # 适合混合检索
        ]
        
        for config_name in test_configs:
            print(f"\n--- 测试配置: {config_name} ---")
            
            try:
                # 初始化RAG系统
                rag = ConfigurableRAG("2.txt", retrieval_config=config_name)
                
                # 测试一个查询
                test_query = test_queries[0]  # 使用第一个查询
                print(f"测试查询: {test_query}")
                
                result = rag.query(test_query, show_sources=False, show_config=True)
                
                if 'error' not in result:
                    answer_preview = str(result['answer'].content)[:100] + "..." if len(str(result['answer'].content)) > 100 else str(result['answer'].content)
                    print(f"   ✅ 回答预览: {answer_preview}")
                    print(f"   ⏱️ 耗时: {result['response_time']}")
                    print(f"   🔍 检索模式: {result['performance_stats'].get('retrieval_mode', 'unknown')}")
                else:
                    print(f"   ❌ 查询失败: {result.get('error', 'unknown error')}")
                
            except Exception as e:
                print(f"   ❌ 配置 {config_name} 测试失败: {e}")
                
        print(f"\n✅ 不同检索模式测试完成")
        
    except Exception as e:
        print(f"❌ 检索模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_config_switching():
    """测试配置切换功能"""
    print("\n🧪 测试配置切换功能...")
    
    try:
        from configurable_rag import ConfigurableRAG
        
        # 初始化RAG系统
        print("初始化RAG系统...")
        rag = ConfigurableRAG("2.txt", retrieval_config="comprehensive")
        
        # 测试配置切换
        switch_configs = ['semantic', 'structured', 'comprehensive']
        test_query = "贾宝玉多大了？"
        
        for config_name in switch_configs:
            print(f"\n--- 切换到配置: {config_name} ---")
            
            try:
                # 切换配置
                rag.switch_config(config_name)
                
                # 测试查询
                print(f"测试查询: {test_query}")
                result = rag.query(test_query, show_sources=False)
                
                if 'error' not in result:
                    print(f"   ✅ 配置切换成功")
                    print(f"   🔍 当前模式: {result['performance_stats'].get('retrieval_mode', 'unknown')}")
                    print(f"   ⏱️ 耗时: {result['response_time']}")
                else:
                    print(f"   ❌ 查询失败: {result.get('error', 'unknown error')}")
                
            except Exception as e:
                print(f"   ❌ 配置切换失败: {e}")
        
        print(f"\n✅ 配置切换测试完成")
        
    except Exception as e:
        print(f"❌ 配置切换测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def performance_comparison():
    """性能对比测试"""
    print("\n🧪 性能对比测试...")
    
    try:
        from configurable_rag import ConfigurableRAG
        
        # 测试配置
        configs_to_test = ['semantic', 'structured', 'comprehensive']
        test_query = "贾宝玉的基本信息？"
        
        results = {}
        
        for config_name in configs_to_test:
            print(f"\n--- 性能测试: {config_name} ---")
            
            try:
                # 初始化RAG系统
                rag = ConfigurableRAG("2.txt", retrieval_config=config_name)
                
                # 执行多次查询取平均值
                times = []
                for i in range(3):
                    start_time = time.time()
                    result = rag.query(test_query, show_sources=False)
                    end_time = time.time()
                    
                    if 'error' not in result:
                        times.append(end_time - start_time)
                    else:
                        print(f"   ❌ 第{i+1}次查询失败")
                
                if times:
                    avg_time = sum(times) / len(times)
                    results[config_name] = {
                        'avg_time': avg_time,
                        'description': rag.config.get_description()
                    }
                    print(f"   ✅ 平均耗时: {avg_time:.2f}秒")
                else:
                    print(f"   ❌ 所有查询都失败了")
                    
            except Exception as e:
                print(f"   ❌ 性能测试失败: {e}")
        
        # 输出性能对比结果
        if results:
            print(f"\n📊 性能对比结果:")
            sorted_results = sorted(results.items(), key=lambda x: x[1]['avg_time'])
            
            for i, (config_name, data) in enumerate(sorted_results, 1):
                print(f"   {i}. {config_name} ({data['description']}): {data['avg_time']:.2f}秒")
        
        print(f"\n✅ 性能对比测试完成")
        
    except Exception as e:
        print(f"❌ 性能对比测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def demo_interactive():
    """演示交互模式"""
    print("\n🎮 交互模式演示...")
    print("这将启动交互模式，你可以:")
    print("  - 输入问题进行查询")
    print("  - 输入 'list' 查看所有配置")
    print("  - 输入 'switch <配置名>' 切换配置")
    print("  - 输入 'quit' 退出")
    
    try:
        from configurable_rag import ConfigurableRAG
        
        # 初始化RAG系统
        rag = ConfigurableRAG("2.txt", retrieval_config="comprehensive")
        
        # 启动交互模式
        rag.interactive()
        
    except Exception as e:
        print(f"❌ 交互模式演示失败: {e}")
        import traceback
        traceback.print_exc()

def main():
    """主测试函数"""
    print("🚀 开始测试可配置RAG系统...")
    print("="*60)
    
    # 运行各种测试
    tests = [
        ("SQL组件测试", test_sql_components),
        ("检索配置测试", test_retrieval_configs),
        ("不同检索模式测试", test_different_modes),
        ("配置切换测试", test_config_switching),
        ("性能对比测试", performance_comparison)
    ]
    
    passed_tests = 0
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"🧪 {test_name}")
        print(f"{'='*60}")
        
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} 通过")
            else:
                print(f"❌ {test_name} 失败")
        except Exception as e:
            print(f"❌ {test_name} 异常: {e}")
    
    # 总结
    print(f"\n{'='*60}")
    print(f"📊 测试总结: {passed_tests}/{len(tests)} 通过")
    print(f"{'='*60}")
    
    # 询问是否要运行交互演示
    if passed_tests > 0:
        print(f"\n🎮 所有测试完成! 是否要运行交互演示? (y/n): ", end="")
        try:
            response = input().strip().lower()
            if response in ['y', 'yes', '是']:
                demo_interactive()
        except (KeyboardInterrupt, EOFError):
            print("跳过交互演示")
    
    print(f"\n🎉 测试完成！")

if __name__ == "__main__":
    main() 