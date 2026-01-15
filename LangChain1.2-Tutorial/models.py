import os
from langchain_openai import ChatOpenAI


# 千问模型
qwen = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus-2025-12-01",  # "qwen3-max"
    temperature=0.7,
    max_tokens=1000,
    timeout=30
)

# deepseek模型:
#    deepseek-chat 对应 DeepSeek-V3.2-Exp 的非思考模式，
#    deepseek-reasoner 对应 DeepSeek-V3.2-Exp 的思考模式.
ds = ChatOpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com",
    model="deepseek-chat",
    temperature=0.7
)