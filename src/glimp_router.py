# %%
import os
import pickle
import pprint
import logging
import numpy as np
from openai import OpenAI
import statistics
from collections import Counter
from tqdm import tqdm
import json
import random
import re

def get_avg_score(scores):
    # Mean over non-null scores.
    return statistics.mean([x for x in scores if x is not None])

def get_frequency(scores):
    # Count frequency of score values.
    return dict(Counter(scores))

def get_model(model_size):
    return model_names[model_size]

# %%
model_names = {
    "32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",     # OR change to the name of your 32b large model, e.g. "org/model-32b"
    "4b": "Qwen/Qwen3-4B-Thinking-2507",    # OR change to the name of your 4b small model
}
ports = {
    "32b": "17125",  # NOTE: change to the port of your 32b large model, e.g. "11125"
    "4b": "17130",  # NOTE: change to the port of your 4b small model, e.g. "11130"
}

clients = {}
for size, full_name in model_names.items():
    clients[size] = OpenAI(
        api_key="glimp_router",  # OR change to the api key of your model
        base_url=f"http://localhost:{ports[size]}/v1",  # OR change to the base url of your model, e.g. f"http://localhost:{ports[size]}/v1"
    )

def get_first_user_msg(problem, options=None):
    if options == "aime" or options == "math":
        system_prompt = "Solve the following math problem and return ONLY the final answer.\nPlease reason step by step, separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.\n\n"
        system_prompt += f"Problem: {problem['problem']}\n\n"
        return system_prompt
    elif options == "lcb":
        raw_prompt = problem["question_content"]
        starter = problem["starter_code"]
        system_prompt = "Write code to solve the following problem and return ONLY the code.\nYou will generate a correct Python program that matches the specification and passes all tests.\n\n"
        system_prompt += f"Question: {raw_prompt}\n\n"
        if starter:
            system_prompt += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
            system_prompt += f"```python\n{starter}\n```\n\n"
        else:
            system_prompt += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n"
            system_prompt += f"```python\n# YOUR CODE HERE\n```\n\n"
        return system_prompt
    elif options == "gpqa":
        system_prompt = "What is the correct answer to the following problem? Please reason step by step.\nSeparate logical reasoning steps with two newline characters (\n\n).\nPut the final answer **strictly** in the format \\boxed{{X}}, where X is a single letter (A, B, C, or D).\n\n**Example output:** \\boxed{{A}}\n\n"
        system_prompt += f"Problem: {problem['problem']}\n\n"
        return system_prompt
    else:
        raise NotImplementedError

# %%
def generate_new_step(problem, steps_so_far, model_size, options=None, stop_token="\n\n"):
    client = clients[model_size]
    
    if steps_so_far == []:  # first step
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options)},
        ]
        extra_body = {"add_generation_prompt": True}
    else:  # continuing on from a previous message
        steps_so_far_str = "\n\n".join(steps_so_far) + "\n\n"
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options)},
            {"role": "assistant", "content": f"<think>{steps_so_far_str}"},
        ]
        extra_body = {"add_generation_prompt": False, "continue_final_message": True}
    
    response = client.chat.completions.create(
        model=get_model(model_size),
        messages=messages,
        temperature=0.6, top_p=0.95,
        max_tokens=512,
        stop=[stop_token],
        extra_body=extra_body,
    )

    step_str = response.choices[0].message.content
    num_output_tokens = response.usage.completion_tokens
    finished = "</think>" in step_str
    
    return step_str, finished, num_output_tokens


def generate_answer(problem, steps_so_far, model_size, options=None):
    client = clients[model_size]
    
    steps_so_far_str = "\n\n".join(steps_so_far)
    steps_so_far_str = steps_so_far_str.split("</think>")[0] if "</think>" in steps_so_far_str else steps_so_far_str

    # Always finalize with the large model to produce the answer.
    messages = [
        {"role": "user", "content": get_first_user_msg(problem, options)},
        {"role": "assistant", "content": f"<think>{steps_so_far_str}\n</think>\n\n"},
    ]
    extra_body = {"add_generation_prompt": False, "continue_final_message": True}
    
    response = client.chat.completions.create(
        model=get_model(model_size),
        messages=messages,
        temperature=0.6, top_p=0.95,
        max_tokens=2048,
        extra_body=extra_body,
    )

    step_str = response.choices[0].message.content
    num_output_tokens = response.usage.completion_tokens
    if options == "lcb":
        s = re.findall(r'```(?:python)?\n(.*?)```', step_str, re.DOTALL | re.IGNORECASE)
        finished = len(s) >= 1
    else:
        finished = any([x in step_str for x in ["boxed", "Answer:", "ANSWER:"]])
    
    return step_str, finished, num_output_tokens


def process_logprobs(response, method, temp=1.0):
    # Extract logprobs for the first generated token.
    assert len(response.choices[0].logprobs.content) == 1
    token = response.choices[0].logprobs.content[0].token
    token_logprobs = {t.token: t.logprob for t in response.choices[0].logprobs.content[0].top_logprobs}
    token_logprobs = {k: v for k, v in token_logprobs.items() if k.isdigit()}  # filter out non-digit values

    if method == "greedy":
        # return the vanilla response
        if not token.isdigit():
            return 0
        return int(token)
    elif method == "average":
        # Convert log probabilities to probabilities and normalize each distribution.
        probs = {tok: np.exp(lp / temp) for tok, lp in token_logprobs.items()}
        total_probs = sum(probs.values())
        for tok in probs:
            probs[tok] /= total_probs
        for i in range(10):
            if i not in probs:
                probs[i] = 0
        return sum([int(t) * p for t, p in probs.items()])
    else:
        raise NotImplementedError


def get_score_first_token_entropy(problem, steps_so_far, model_size="4b", options=None):
    client = clients[model_size]
    
    if steps_so_far == []:  # first step
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options)},
        ]
        extra_body = {"add_generation_prompt": True}
    else:  # continuing on from a previous message
        steps_so_far_str = "\n\n".join(steps_so_far) + "\n\n"
        messages = [
            {"role": "user", "content": get_first_user_msg(problem, options)},
            {"role": "assistant", "content": f"<think>{steps_so_far_str}"},
        ]
        extra_body = {"add_generation_prompt": False, "continue_final_message": True}
    
    response = client.chat.completions.create(
        model=get_model(model_size),
        messages=messages,
        temperature=0.0, 
        max_tokens=1,
        logprobs=True,
        top_logprobs=20,
        extra_body=extra_body,
    )

    content = response.choices[0].message.content

    # Calcute Score
    assert len(response.choices[0].logprobs.content) == 1
    token = response.choices[0].logprobs.content[0].token
    token_logprobs = {t.token: t.logprob for t in response.choices[0].logprobs.content[0].top_logprobs}

    # Convert log probabilities to probabilities and normalize each distribution.
    probs = {tok: np.exp(lp) for tok, lp in token_logprobs.items()}
    total_probs = sum(probs.values())
    for tok in probs:
        probs[tok] /= total_probs

    entropy = -sum([p * np.log(p) for p in probs.values()])

    return entropy, content, response


def get_score(score_method, problem, steps_so_far, model_size="32b", options=None):
    if score_method=='first_token_entropy':
        return get_score_first_token_entropy(problem, steps_so_far, model_size=model_size, options=options)
    else:
        raise NotImplementedError


def glimprouter(
    problem,
    options=None,
    dataset_name="aime24",
    score_threshold=1.0,
    token_budget=8192,
    problem_id=0,
    repeat_id=0,
    score_method="first_token_entropy",
    output_dir="./results",
    first_n_steps_base_model=0,
    model_size="32b",
    small_model_size="4b",
):
    problem_uid = f"{dataset_name}/{problem_id}"
    output_filename = os.path.join(output_dir, f"{problem_uid}/{repeat_id}")

    if os.path.exists(f"{output_filename}.json"):
        with open(f"{output_filename}.json", "r", encoding="utf-8") as f:
            metadata_list = json.load(f)
        return metadata_list

    steps_so_far = []
    step_id = 0
    metadata_list = []

    try:
        while True:
            warning_flag = False
            if step_id < first_n_steps_base_model: # zeroshot
                base_model_step, finished, num_output_tokens_base = generate_new_step(
                    problem, steps_so_far, model_size, options=options
                )
                small_model_step, num_output_tokens_small = None, None
                score, justification = None, None
                step_str = base_model_step
                steps_so_far.append(step_str)
            elif score_method == "first_token_entropy": # first-token entropy
                score, justification, response = get_score(
                    score_method,
                    problem,
                    steps_so_far,
                    model_size=small_model_size,
                    options=options,
                )

                if score is not None and score >= score_threshold:
                    # large model generates
                    base_model_step, finished, num_output_tokens_base = generate_new_step(
                        problem, steps_so_far, model_size, options=options
                    )
                    small_model_step, num_output_tokens_small = None, None
                    step_str = base_model_step
                else:
                    # small model generates
                    small_model_step, finished, num_output_tokens_small = generate_new_step(
                        problem, steps_so_far, small_model_size, options=options
                    )
                    base_model_step, num_output_tokens_base = None, None
                    step_str = small_model_step
                steps_so_far.append(step_str)
            else:
                raise NotImplementedError

            # collect metadata
            metadata = {
                "step_id": step_id,
                "step_str": step_str,
                "small_model_step": small_model_step,
                "num_output_tokens_small": num_output_tokens_small,
                "score": score,
                "base_model_step": base_model_step,
                "num_output_tokens_base": num_output_tokens_base,
                "final_num_output_tokens": (
                    num_output_tokens_base if num_output_tokens_base is not None else num_output_tokens_small
                ),
                "justification": justification,
            }
            metadata_list.append(metadata)
            step_id += 1

            # Check if finished
            if len(steps_so_far) > 2:
                finished = finished or steps_so_far[-1] == steps_so_far[-2]

            if finished or sum(m["final_num_output_tokens"] for m in metadata_list) >= token_budget:
                if sum(m["final_num_output_tokens"] for m in metadata_list) >= token_budget:
                    metadata_list[-1]["stop_reason"] = "budget"
                else:
                    metadata_list[-1]["stop_reason"] = "finished"
                break

        # Generation of Final Answer
        base_model_step, finished, num_output_tokens_base = generate_answer(
            problem, steps_so_far, model_size, options=options
        )
        small_model_step, num_output_tokens_small = None, None
        score, justification = None, None
        step_str = base_model_step
        steps_so_far.append(step_str)

        metadata = {
            "step_id": step_id,
            "step_str": step_str,
            "small_model_step": small_model_step,
            "num_output_tokens_small": num_output_tokens_small,
            "score": score,
            "base_model_step": base_model_step,
            "num_output_tokens_base": num_output_tokens_base,
            "final_num_output_tokens": (
                num_output_tokens_base if num_output_tokens_base is not None else num_output_tokens_small
            ),
            "justification": justification,
            "answer": finished,
        }
        metadata_list.append(metadata)

    except ValueError:
        logging.error("ValueError caught in chat template application, continuing")

    # save results
    os.makedirs(os.path.dirname(f"{output_filename}.json"), exist_ok=True)

    with open(f"{output_filename}.json", "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=4)

    with open(f"{output_filename}.txt", "w") as f:
        pprint.pprint(metadata_list, stream=f)

    return metadata_list