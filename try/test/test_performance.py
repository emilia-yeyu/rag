#!/usr/bin/env python3
"""
RAG性能测试脚本
用于测试各个环节的耗时分布
"""

from rag import SimpleRAG
import json

def test_rag_performance():
    """测试RAG系统性能"""
    print("🚀 启动RAG性能测试")
    print("="*60)
    
    # 初始化RAG系统
    rag = SimpleRAG("1.txt")
    
    # 测试问题
    test_questions = [
        "一微半导体是什么公司？",
        "员工迟到会有什么处罚？", 
        "公司的核心价值观是什么？"
    ]
    
    all_results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n🔍 测试 {i}/{len(test_questions)}: {question}")
        print("-"*50)
        
        # 执行查询（包含性能统计）
        result = rag.query(question, show_sources=True)
        all_results.append(result)
        
        print(f"💬 回答: {result['answer'].content}")

        
    # 汇总性能统计
    print(f"\n" + "="*60)
    print(f"📊 性能测试总结")
    print(f"="*60)
    
    total_retrieval_time = 0
    total_llm_time = 0
    total_time = 0
    
    for i, result in enumerate(all_results, 1):
        if "performance_stats" in result:
            stats = result["performance_stats"]
            print(f"问题{i}: 总耗时 {stats['total_time']}, LLM推理 {stats['llm_time']}, 瓶颈: {stats['bottleneck']}")
            
            # 累计统计（去掉"秒"字符）
            total_time += float(stats['total_time'].replace('秒', ''))
            total_llm_time += float(stats['llm_time'].replace('秒', ''))
            total_retrieval_time += float(stats['retrieval_time'].replace('秒', ''))
    
    avg_total_time = total_time / len(all_results)
    avg_llm_time = total_llm_time / len(all_results)
    avg_retrieval_time = total_retrieval_time / len(all_results)
    
    print(f"\n🎯 平均性能指标:")
    print(f"  平均总耗时: {avg_total_time:.2f}秒")
    print(f"  平均LLM耗时: {avg_llm_time:.2f}秒 ({avg_llm_time/avg_total_time*100:.1f}%)")
    print(f"  平均检索耗时: {avg_retrieval_time:.2f}秒 ({avg_retrieval_time/avg_total_time*100:.1f}%)")
    
    # 优化建议
    print(f"\n💡 优化建议:")
    if avg_llm_time / avg_total_time > 0.8:
        print(f"  🎯 LLM推理占{avg_llm_time/avg_total_time*100:.1f}%，是主要瓶颈")
        print(f"  🚀 建议使用TurboRAG的KV缓存优化，预期提升70-80%")
        print(f"  📝 运行命令: python turbo_rag.py --preprocess")
    elif avg_retrieval_time / avg_total_time > 0.3:
        print(f"  🔍 向量检索占{avg_retrieval_time/avg_total_time*100:.1f}%，检索较慢")
        print(f"  📊 建议优化向量数据库配置或减少检索数量")
    else:
        print(f"  ✅ 各环节性能较为均衡")

if __name__ == "__main__":
    test_rag_performance() 