import asyncio

import aiohttp
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
    ReasoningEffort,
    RenderConversationConfig
)
import os
os.environ["TIKTOKEN_RS_CACHE_DIR"] = os.path.dirname(os.path.abspath(__file__))
files_in_current_directory = os.listdir(os.path.dirname(os.path.abspath(__file__)))  # List files in the current directory

encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

VLLM_STOP_STRINGS = ["<|call|>", "<|return|>"]  # Harmony 常用停止符

llm_semaphore = asyncio.Semaphore(256)
async def complete_once(
            prompt_text: str,
            max_tokens: int = 8192,
            temperature: float = 1.0,
            top_p: float = 0.75,
            url: str = None,
    ) -> str:
        payload = {
            "model": "gpt-oss-120b",
            "prompt": prompt_text,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": VLLM_STOP_STRINGS, # NOTE dont know whether to use
            # 兼容字段（vLLM 常见扩展）
            "skip_special_tokens": False,
            "add_special_tokens": False,
            "include_stop_str_in_output": True,
        }
        async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60)
            ) as session:  # 先创建并管理ClientSession的生命周期
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=7200)) as resp:  # 再发起请求
                    resp.raise_for_status()  # 检查HTTP错误状态码
                    data = await resp.json()
                    return data["choices"][0]["text"]

render_cfg = RenderConversationConfig(auto_drop_analysis=False)

def _render_prompt_text(convo: Conversation) -> str:
    tokens = encoding.render_conversation_for_completion(convo, Role.ASSISTANT, config=render_cfg)
    return encoding.decode(tokens)
# ---------- 与 responses 版一致的对外方法 ----------
async def run_gpt_oss(system_prompt: str, user_prompt: str, url: str = None,max_tokens: int = 1024, temperature: float = 0.0):
    convo = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, system_prompt),
            # Message.from_role_and_content(Role.DEVELOPER, self.developer_message),
            Message.from_role_and_content(Role.USER, user_prompt),
        ]
    )
    prompt_text = _render_prompt_text(convo)
    chunk = await complete_once(
        prompt_text,
        url = url,
        max_tokens=max_tokens,
        temperature=temperature
    )
    # print_with_verbose(colored(f"Assistant: {chunk}", "yellow"), verbose=self.verbose)
    assistant_messages = encoding.parse_messages_from_completion_tokens(encoding.encode(chunk, allowed_special='all'), Role.ASSISTANT)

    assert assistant_messages[-1].channel == 'final', assistant_messages[-1]
    return assistant_messages[-1].content[0].text

if __name__ == "__main__":
    asyncio.run(run_gpt_oss("你是一个助手", "你好"))