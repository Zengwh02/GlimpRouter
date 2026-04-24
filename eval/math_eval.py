import sys
import os
import argparse
from datasets import load_dataset, load_from_disk
import json
import re
import math
from math_verify import parse, verify


def get_dataset(dataset_name):
    if dataset_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
    elif dataset_name == "aime25":
        dataset = load_dataset("math-ai/aime25")["test"]
    elif dataset_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif dataset_name == "gpqa":
        if os.getenv("HF_HUB_OFFLINE", "1") == "0":
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond")["train"]
        else:  
            dataset = load_dataset(
            'json',
            data_files="../data/gpqa/gpqa_diamond_test.jsonl",
            split="train",  
        )
    else:
        raise NotImplementedError
    return dataset


def is_correct(answer, gold):
    gold = parse(gold)
    answer = parse(answer)

    return verify(gold=gold, target=answer)


def is_exact_correct(answer, gold):
    start_token = r'\boxed{'
    start = answer.find(start_token)
    if start == -1:
        return False

    i = start + len(start_token)
    brace_count = 1
    content = []

    while i < len(answer):
        char = answer[i]
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1

        if brace_count == 0:
            break

        content.append(char)
        i += 1

    if brace_count == 0:
        answer = ''.join(content)
        return is_correct(answer, gold)
    return False


def check(args):
    dataset = get_dataset(args.dataset_name)
    generations = load_dataset("json", data_files=args.answer_path, split="train")

    results = []
    for i in range(len(dataset)):
        gold = dataset[i]['answer']
        answer = generations[i]['answer']

        
        result = is_correct(answer, gold)
        
        results.append(result)

    # calculate pass rate
    pass_rate = sum(results) / len(results)
    print(f"Pass Rate: {pass_rate * 100:.2f}%")

    if not os.path.exists(args.output_file):
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    with open(args.output_file, 'w') as f:
        d = {"pass_rate": pass_rate, "results": results}
        f.write(json.dumps(d))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_path', type=str, help="generations path")
    parser.add_argument('--dataset_name', type=str, help="dataset path")
    parser.add_argument('--output_file', type=str, help="output file")
    args = parser.parse_args()

    check(args)
