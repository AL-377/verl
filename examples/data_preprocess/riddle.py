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

def proprocess_dataset(source_path: str,split: str,save_path: str) -> pd.DataFrame:
    df = pd.read_parquet(source_path)
    new_data_items = []
    for _, row in df.iterrows():
        data_source = row["data_source"]
        prompt = row["prompt"][0]
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
    proprocess_dataset(train_path, "train", "train_verl_compat.parquet")
    # proprocess_dataset(eval_path, "eval", "eval_verl_compat.parquet")