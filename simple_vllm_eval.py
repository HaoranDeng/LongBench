#!/usr/bin/env python3
"""
使用 vLLM 在 LongBench 上做预测并评估（参考 pred.py）

此脚本会：
- 使用 vLLM 对 LongBench 的每个数据集做逐条预测
- 将结果写入 `pred/` 或 `pred_e/` 目录下，格式与 LongBench 官方 pred.py 一致
- 在完成后调用 LongBench 自带的 eval.py 计算并保存得分

注意：需要安装 vllm、transformers、datasets 等依赖，且模型需与 vLLM 兼容。
"""

import os
import json
import argparse
import subprocess
import time
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="HuggingFace 模型路径或 id，用于 vLLM 加载")
    parser.add_argument('--model_name', type=str, default=None, help="用于输出文件夹的模型名字，默认取 model_path 的 basename")
    parser.add_argument('--e', action='store_true', help="在 LongBench-E 数据集上评测")
    parser.add_argument('--temperature', type=float, default=0.0, help="生成温度")
    parser.add_argument('--top_p', type=float, default=1.0, help="top_p")
    parser.add_argument('--max_model_len', type=int, default=131072, help="最大输入 token 数（用于截断）")
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.95, help="vLLM 的 GPU 内存使用率")
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help="vLLM 张量并行大小")
    return parser.parse_args(args)


def load_longbench_configs():
    cfg_dir = os.path.join("LongBench", "config")
    dataset2prompt = json.load(open(os.path.join(cfg_dir, "dataset2prompt.json"), "r", encoding="utf-8"))
    dataset2maxlen = json.load(open(os.path.join(cfg_dir, "dataset2maxlen.json"), "r", encoding="utf-8"))
    return dataset2prompt, dataset2maxlen


def main():
    args = parse_args()
    model_path = args.model_path
    model_name = args.model_name or os.path.basename(model_path)

    # LongBench 任务列表（与 LongBench/LongBench/pred.py 保持一致）
    if args.e:
        datasets_list = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", \
                         "trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]
    else:
        datasets_list = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
                         "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
                         "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    dataset2prompt, dataset2maxlen = load_longbench_configs()

    # 加载 tokenizer（用于截断）
    print(f"Loading tokenizer from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    # 初始化 vLLM
    print(f"Loading model for vLLM: {model_path} ...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )

    pred_root = "pred_e" if args.e else "pred"
    os.makedirs(pred_root, exist_ok=True)
    os.makedirs(os.path.join(pred_root, model_name), exist_ok=True)

    for dataset in datasets_list:
        ds_name = f"{dataset}_e" if args.e else dataset
        print(f"\nProcessing dataset: {ds_name}")
        try:
            data = load_dataset('THUDM/LongBench', ds_name, split='test')
        except Exception as e:
            print(f"Failed to load dataset {ds_name}: {e}")
            continue

        out_path = os.path.join(pred_root, model_name, f"{dataset}.jsonl")
        # open file for appending so partially generated results survive
        with open(out_path, 'a', encoding='utf-8') as fout:
            # iterate samples
            for item in tqdm(data, desc=f"gen:{dataset}"):
                # prepare prompt
                prompt_format = dataset2prompt.get(dataset)
                if prompt_format is None:
                    print(f"No prompt template for dataset {dataset}, skipping")
                    break
                try:
                    prompt = prompt_format.format(**item)
                except Exception:
                    # some items may not contain expected keys, fall back to simple concatenation
                    prompt = "\n\n".join(str(v) for v in item.values())

                # truncate by tokens if necessary (截断策略同 pred.py：两端保留)
                try:
                    encoded = tokenizer(prompt, truncation=False, return_tensors='pt', add_special_tokens=False).input_ids[0]
                    if len(encoded) > args.max_model_len:
                        half = args.max_model_len // 2
                        prompt = tokenizer.decode(list(encoded[:half]) + list(encoded[-half:]), skip_special_tokens=True)
                except Exception:
                    # 如果 tokenizer 无法处理则按字符截断（保守）
                    if len(prompt) > 200000:
                        prompt = prompt[:100000] + prompt[-100000:]

                max_gen = dataset2maxlen.get(dataset, 128)
                sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=max_gen)

                # vLLM 生成（逐条生成以降低显存与实现简单）
                try:
                    outputs = llm.generate([prompt], sampling_params)
                    gen_text = ''
                    for out in outputs:
                        # 输出结构参照 simple_vllm_eval.py 示例
                        if len(out.outputs) > 0:
                            gen_text = out.outputs[0].text
                        else:
                            gen_text = ''
                        break
                except Exception as e:
                    print(f"Generation failed for one sample: {e}")
                    gen_text = ''

                # prepare record in the same format as LongBench 官方 pred.py
                record = {
                    "pred": gen_text,
                    "answers": item.get("answers", item.get("answer", [])),
                    "all_classes": item.get("all_classes", item.get("all_choices", [])),
                }
                if "length" in item:
                    record["length"] = item["length"]

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

        print(f"Wrote predictions for {dataset} to {out_path}")

    # 调用 LongBench 自带的 eval.py 来计算得分
    eval_script = os.path.join("LongBench", "eval.py")
    cmd = ["python3", eval_script, "--model", model_name]
    if args.e:
        cmd.append("--e")
    print("\nRunning evaluation via:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
        print("Evaluation finished. Results written to pred/ or pred_e/ folder.")
    except Exception as e:
        print(f"Evaluation script failed: {e}")


if __name__ == '__main__':
    main()
