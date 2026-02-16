# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import os
import json
import asyncio
from mathruler.grader import extract_boxed_content, grade_answer

from .llm import run_gpt_oss


def format_general_em_message(answer:str, response:str):
    system_prompt = f"""# Role
你是一个严格的评测专家。你的任务是对比“待测回答（Candidate Answer）”与“标准答案（Ground Truth）”，是否指向同一个答案。

## Evaluation Rules
1. 缩写/全称 都被认为是同一个答案
2. 大小写不敏感 含有相同即可
3. 特殊字符如果不影响含义, 也被认为是同一个答案

请判断模型的回答是否正确。

## Output Format
请仅输出一个 JSON 对象，不要包含任何解释性文本。格式如下：
{{
  "reason": "判断理由, 请详细判断这两个答案是否指向同一个答案",
  "score": true或false, true表示指向同一个答案, false表示指向不同的答案
}}"""
    user_prompt = f"""现在需要判断的信息如下:
题目的ground_truth是: {answer}
模型的response是: {response}
---
请按照格式要求回复, 判断模型的回答是否正确。"""
    return system_prompt,user_prompt




def format_reward(predict_str: str) -> float:
    print("raw predict_str: ", predict_str)
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    match_result = re.fullmatch(pattern, predict_str)
    return 1.0 if match_result else 0.0


async def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = True) -> float:
    if use_boxed:
        answer = extract_boxed_content(predict_str)
    else:
        answer = predict_str
    print("Predict str: ", answer)        
    print("Ground truth: ", ground_truth)
    system_prompt,user_prompt = format_general_em_message(ground_truth, answer)
    eval_url = os.environ.get('EVAL_OSS_MODEL_URL', "http://[2605:340:cd51:7700:980d:6504:c72e:240d]:8000/v1/completions")
    eval_max_tokens = int(os.environ.get('EVAL_LLM_MAX_TOKENS', '4096'))
    eval_temperature = float(os.environ.get('EVAL_LLM_TEMPERATURE', '0.0'))
    eval_res = None
    try:
        eval_output = await run_gpt_oss(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            url=eval_url,
            max_tokens=eval_max_tokens,
            temperature=eval_temperature,
        )
        print("raw eval_output: ", eval_output)
        eval_output = eval_output.strip()
        eval_output = json.loads(eval_output)
        eval_res = eval_output.get("score")
    except Exception as e:
        print(f"Error evaluating task: {e}")
        return 0.0
    return 1.0 if eval_res=="true" or eval_res==True else 0.0


async def compute_score(predict_str: str, ground_truth: str, use_boxed: bool = True, format_score: float = 0.1) -> float:
    if format_reward(predict_str) == 0.0:
        return 0.0
    return await acc_reward(predict_str, ground_truth, use_boxed)


if __name__ == '__main__':
    print(asyncio.run(compute_score("\\boxed{1}", "1")))
    print(asyncio.run(compute_score("<think>I think the answer is</think>\\boxed{cereal grain}", "wheat")))
    print(asyncio.run(compute_score("<think>I think the answer is</think>\\boxed{Phrygan Cap}", "phrygan cap")))
    