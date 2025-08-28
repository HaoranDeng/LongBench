#!/usr/bin/env python3
"""
简单的 vLLM 模型加载和测试脚本 - 完全独立，无外部依赖
"""

from vllm import LLM, SamplingParams
import argparse
import json
import time

def main():
    parser = argparse.ArgumentParser(description="使用 vLLM 加载和测试 HuggingFace 模型")
    
    # 必需参数
    parser.add_argument("--model_path", type=str, required=True, 
                       help="HuggingFace 模型路径 (本地路径或 hub 模型名)")
    
    # 可选参数
    parser.add_argument("--prompt", type=str, default="Hello, how are you?", 
                       help="测试提示词")
    parser.add_argument("--max_tokens", type=int, default=100, 
                       help="生成的最大 token 数")
    parser.add_argument("--temperature", type=float, default=0.8, 
                       help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.95, 
                       help="Top-p 采样参数")
    
    # vLLM 参数
    parser.add_argument("--tensor_parallel_size", type=int, default=1, 
                       help="张量并行大小")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, 
                       help="GPU 内存使用率")
    parser.add_argument("--max_model_len", type=int, default=4096, 
                       help="模型最大序列长度")
    
    args = parser.parse_args()
    
    print(f"正在加载模型: {args.model_path}")
    print(f"张量并行大小: {args.tensor_parallel_size}")
    print(f"GPU 内存使用率: {args.gpu_memory_utilization}")
    print("-" * 50)
    
    try:
        # 初始化 vLLM 模型
        start_time = time.time()
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=args.max_model_len,
        )
        load_time = time.time() - start_time
        print(f"✅ 模型加载成功! 耗时: {load_time:.2f} 秒")
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )
        
        print(f"\n测试提示词: {args.prompt}")
        print(f"采样参数: temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}")
        print("-" * 50)
        
        # 生成文本
        start_time = time.time()
        outputs = llm.generate([args.prompt], sampling_params)
        generation_time = time.time() - start_time
        
        # 显示结果
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"提示词: {prompt}")
            print(f"生成结果: {generated_text}")
            print(f"生成耗时: {generation_time:.2f} 秒")
            print(f"生成 token 数: {len(output.outputs[0].token_ids)}")
            
            # 计算一些基本统计
            if generation_time > 0:
                tokens_per_second = len(output.outputs[0].token_ids) / generation_time
                print(f"生成速度: {tokens_per_second:.2f} tokens/秒")
        
        print("\n✅ 测试完成!")
        
        # 可选：批量测试多个提示词
        test_prompts = [
            "What is artificial intelligence?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about the ocean.",
        ]
        
        print(f"\n正在进行批量测试 ({len(test_prompts)} 个提示词)...")
        batch_start = time.time()
        batch_outputs = llm.generate(test_prompts, sampling_params)
        batch_time = time.time() - batch_start
        
        print(f"批量生成耗时: {batch_time:.2f} 秒")
        total_tokens = sum(len(output.outputs[0].token_ids) for output in batch_outputs)
        print(f"总 token 数: {total_tokens}")
        print(f"批量平均速度: {total_tokens / batch_time:.2f} tokens/秒")
        
        for i, output in enumerate(batch_outputs):
            print(f"\n--- 测试 {i+1} ---")
            print(f"提示词: {output.prompt}")
            print(f"生成结果: {output.outputs[0].text[:100]}..." if len(output.outputs[0].text) > 100 else f"生成结果: {output.outputs[0].text}")
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
