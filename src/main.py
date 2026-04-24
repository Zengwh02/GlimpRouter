import argparse
from glimp_router import glimprouter
from datasets import load_dataset
import os
from tqdm import tqdm
from pprint import pprint
import time
import json
import re


# Optional JSON config for defaults; CLI args override these values.
config_path = "config.json"
config = {}
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)


def get_dataset(dataset_name):
    # Map dataset name to loader and option tag.
    options = None
    if dataset_name == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
        options = "aime"
    elif dataset_name == "aime25":
        dataset = load_dataset("math-ai/aime25")["test"]
        options = "aime"
    elif dataset_name == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
        options = "math"
    elif dataset_name == "gpqa":
        dataset = load_dataset(
            'json',
            data_files="YOUR_DIRECTORY_OF_GPQA_DATASET",  # NOTE: change to the directory of your GPQA dataset, e.g. "../data/gpqa/gpqa_diamond_test.jsonl"
            split="train",
        )
        options = "gpqa"
    elif dataset_name == "lcbv5":
        dataset = load_dataset(
            'json',
            data_files="YOUR_DIRECTORY_OF_LCB_DATASET",  # NOTE: change to the directory of your LCB dataset, e.g. "../data/lcbv5/test5.jsonl"
            split="train",
        )
        options = "lcb"
    elif dataset_name == "lcbv6":
        dataset = load_dataset(
            'json',
            data_files="YOUR_DIRECTORY_OF_LCB_DATASET",  # NOTE: change to the directory of your LCB dataset, e.g. "../data/lcbv6/test6.jsonl"
            split="train",
        )
        options = "lcb"
    else:
        raise NotImplementedError
    return dataset, options


def extract_answer(result, options):
    step_str = result[-1]['step_str']
    if options == "lcb":
        # LCB returns code blocks; extract the first code block if possible.
        try:
            s = re.findall(r'```(?:python)?\n(.*?)```', step_str, re.DOTALL | re.IGNORECASE)[0]
        except Exception as ex:
            print(f"Exception: {ex}. Failed to extract codeblock:\n{step_str}")
            s = step_str
    elif options == "aime" or options == "math" or options == "gpqa":
        s = step_str
    else:
        raise NotImplementedError
    return s


def main(args):
    print(f"Args:")
    pprint(vars(args))

    dataset, options = get_dataset(args.dataset_name)

    # Choose scoring strategy and thresholds.
    score_method = "first_token_entropy"
    score_threshold = 0.9
    first_n_steps_base_model = 0
    
    if args.score_method == 'zeroshot':
        first_n_steps_base_model = 16384
    elif args.score_method == 'first_token_entropy':
        score_method = 'first_token_entropy'
        if args.score_threshold != 0.0:
            score_threshold = args.score_threshold
        else:
            score_threshold = 1.0
    else:
        raise NotImplementedError

    answers = {}
    generation_time = []
    # Generate answers and record timing per problem.
    for problem_id, problem in tqdm(enumerate(dataset), total=len(dataset), position=0, leave=True, desc="Generation"):
        answer_repeat = []
        time_repeat = []
        for repeat_id in range(args.repeat_num):
            
            s_time = time.time()
            result = glimprouter(
                problem=problem,
                options=options,
                dataset_name=args.dataset_name,
                token_budget=args.token_budget,
                repeat_id=repeat_id,
                score_method=score_method,
                output_dir=args.output_dir,
                problem_id=problem_id,
                score_threshold=score_threshold,
                first_n_steps_base_model=first_n_steps_base_model,
                model_size=args.model_size,
                small_model_size=args.small_model_size,
            )
            e_time = time.time()

            answer_repeat.append({'answer': extract_answer(result, options), 'id': problem_id, "question_id": problem.get("question_id", "")})
            time_repeat.append(e_time - s_time)
        answers[problem_id] = answer_repeat
        generation_time.append(sum(time_repeat))

    final_results = []
    for i in range(args.repeat_num):
        item = [{'id': v[i]['id'], 'answer': v[i]['answer'], "question_id": v[i]['question_id']} for k, v in answers.items()]
        final_results.append(item)

        # Write per-repeat results for downstream evaluation.
        output_dir = os.path.join(args.output_dir, args.dataset_name)
        output_filename = os.path.join(output_dir, f"result_{i+1}.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=4)
    
    print(f"Total Time: {sum(generation_time)}; Avg Time: {sum(generation_time)/len(generation_time)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GlimpRouter experiment")

    parser.add_argument("--dataset_name", type=str, choices=["aime24", "math500", "gpqa", "aime25", "lcbv5", "lcbv6"], 
                        default=config.get("dataset_name", "aime24"),
                        help="Dataset name")
    parser.add_argument("--token_budget", type=int, 
                        default=config.get("token_budget", 8192),
                        help="Max num of total output tokens in each step")
    parser.add_argument("--repeat_num", type=int, 
                        default=config.get("repeat_num", 1),
                        help="Repeat Num")
    parser.add_argument("--score_method", type=str, choices=["zeroshot", "first_token_entropy"], 
                        default=config.get("score_method", "first_token_entropy"),
                        help="Scoring method")
    parser.add_argument("--output_dir", type=str, 
                        default=config.get("output_dir", "./main_results"),
                        help="Where result pickle files will be written to")
    parser.add_argument("--model_size", type=str, 
                        default=config.get("model_size", "32b"),
                        help="Large model size")
    parser.add_argument("--small_model_size", type=str, 
                        default=config.get("small_model_size", "4b"),
                        help="Small model size")
    parser.add_argument("--score_threshold", type=float, 
                        default=config.get("score_threshold", 0.0),
                        help="Acceptance threshold")

    args = parser.parse_args()
    main(args)