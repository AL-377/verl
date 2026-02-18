import pandas as pd
import json

train_path = "hdfs://haruna/home/byte_data_seed/lf_lq/user/heqianyu.work/heqianyu/datasets/rl_data/training/riddle/final_riddle_1106p1.parquet"
eval_path = "hdfs://haruna/home/byte_data_seed/lf_lq/user/heqianyu.work/heqianyu/datasets/rl_data/evaluation/riddle/eval_riddle_commonsense_math_gpqa_251104a.parquet"

# {
#     "data_source": data_source,
#     "prompt": [
#         {
#             "role": "user",
#             "content": question,
#         }
#     ],
#     "ability": "math",
#     "reward_model": {"style": "rule", "ground_truth": solution},
#     "extra_info": {
#         "split": split,
#         "index": idx,
#         "answer": answer_raw,
#         "question": question_raw,
#     },
# }
def insert_mc_format_req(raw_prompt:str):
    old_format_str = "Answer the following Multiple Choice Problem, answer in the format of \\boxed{option} (e.g. \\boxed{A})"
    format_str = """You are an intelligent multi choice solver. You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think> </think> tags, and then output the final option, which should be enclosed within \\boxed{} (e.g. \\boxed{A})
**OUTPUT FORMAT:**:
<think> [Your thinking about what answer should be for this riddle] </think>
\\boxed{Your final option here.}
---
"""
    new_prompt = raw_prompt.replace(old_format_str,format_str)
    return new_prompt

def insert_puzzle_format_req(raw_prompt:str):
    format_str = """You are an intelligent puzzle solver. You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think> </think> tags, and then output the final option, which should be enclosed within \\boxed{} (e.g. \\boxed{A})
**OUTPUT FORMAT:**:
<think> [Your thinking about what answer should be for this riddle] </think>
\\boxed{Your final option here.}
---
"""
    new_prompt = format_str + raw_prompt
    return new_prompt


def insert_math_format_req(raw_prompt:str):
    old_format_str = "Answer the following math problem and return the answer in the format of \\boxed{answer}"
    format_str = """You are an intelligent math solver. You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think> </think> tags, and then output the final answer, which should be enclosed within \\boxed{}.
**OUTPUT FORMAT:**:
<think> [Your thinking about what answer should be for this riddle] </think>
\\boxed{Your final answer here.}
---
"""
    new_prompt = raw_prompt.replace(old_format_str,format_str)
    return new_prompt

def insert_format_req(raw_prompt:str):
    old_format_str = "You are an intelligent riddle solver. Provide the answer enclosed in \\boxed{}."
    format_str = """You are an intelligent riddle solver. You should begin by detailing the internal reasoning process, and then present the answer to the user. The reasoning process should be enclosed within <think> </think> tags, and then output the final answer, which should be enclosed within \\boxed{}.
**OUTPUT FORMAT:**:
<think> [Your thinking about what answer should be for this riddle] </think>
\\boxed{Your final answer here.}
---
"""
    new_prompt = raw_prompt.replace(old_format_str,format_str)
    return new_prompt
def proprocess_dataset(source_path: str,split: str,save_path: str) -> pd.DataFrame:
    df = pd.read_parquet(source_path)
    new_data_items = []
    for _, row in df.iterrows():
        if row["data_source"] == "metaphor_riddle":
            data_source = row["data_source"]
            new_prompt_str = insert_format_req(row["prompt"][0]["content"])
            prompt = {"role": "user", "content": new_prompt_str}
            ability = "math"
            try:
                ground_truth = json.loads(row["reward_model"]["ground_truth"])["answer"]
            except Exception:
                ground_truth = row["reward_model"]["ground_truth"]
            print(ground_truth)
            reward_model = {"ground_truth": ground_truth, "style": "rule"}
            extra_info = row["extra_info"]
            extra_info["split"] = split
            extra_info["question"] = row["raw_problem"]
            extra_info["answer"] = ground_truth
        elif row["reward_model"]["style"]=="rule-lighteval/MATH_v2":
            if "AIME" not in row["data_source"]:
                continue
            data_source = "metaphor_riddle_" + row["data_source"]
            new_prompt_str = insert_math_format_req(row["prompt"][0]["content"])
            prompt = {"role": "user", "content": new_prompt_str}
            ability = "math"
            ground_truth = row["reward_model"]["ground_truth"]
            print(ground_truth)
            reward_model = {"ground_truth": ground_truth, "style": "rule"}
            extra_info = row["extra_info"]
            extra_info["split"] = split
            extra_info["question"] = row["raw_problem"]
            extra_info["answer"] = ground_truth
        elif row["reward_model"]["style"]=="rule-boxed_gpqa":
            if "GPQA" not in row["data_source"]:
                continue
            data_source = "metaphor_riddle_" + row["data_source"]
            new_prompt_str = insert_mc_format_req(row["prompt"][0]["content"])
            prompt = {"role": "user", "content": new_prompt_str}
            ability = "math"
            ground_truth = row["reward_model"]["ground_truth"]
            print(ground_truth)
            reward_model = {"ground_truth": ground_truth, "style": "rule"}
            extra_info = row["extra_info"]
            extra_info["split"] = split
            extra_info["question"] = row["raw_problem"]
            extra_info["answer"] = ground_truth
        elif row["reward_model"]["style"]=="rule-logic_puzzle":
            data_source = "metaphor_riddle_" + row["data_source"]
            new_prompt_str = insert_puzzle_format_req(row["prompt"][0]["content"])
            prompt = {"role": "user", "content": new_prompt_str}
            ability = "math"
            ground_truth = json.dumps(json.loads(row["reward_model"]["ground_truth"])["answer"])
            print(ground_truth)
            reward_model = {"ground_truth": ground_truth, "style": "rule"}
            extra_info = row["extra_info"]
            extra_info["split"] = split
            extra_info["question"] = row["raw_problem"]
            extra_info["answer"] = ground_truth
        else:
            continue
        new_data_items.append({
            "data_source": data_source,
            "prompt": [prompt],
            "ability": ability,
            "reward_model": reward_model,
            "extra_info": extra_info,
        })
    print(new_data_items[-1])
    df = pd.DataFrame(new_data_items)
    df.to_parquet(save_path, index=False, engine="pyarrow", row_group_size=8)

if __name__ == "__main__":
    # proprocess_dataset(train_path, "train", "data/train_verl_compat_add_think.parquet")
    proprocess_dataset(eval_path, "test", "data/eval_verl_only_gpqa_aime_puzzle.parquet")