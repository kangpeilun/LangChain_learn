# -*- coding: utf-8 -*-
#        Data: 2026-01-16 13:44
#     Project: LangChain_learn
#   File Name: 2propmt_template.py
#      Author: KangPeilun
#       Email: 374774222@qq.com 
# Description:

from langchain_core.prompts import ChatPromptTemplate
from model_config import models

"""
ChatPromptTemplate 提供角色设置
    system: 系统角色，定义整体行为和规则
    user: 用户角色，表示用户输入
    assistant: 助手角色，表示模型的回应
"""

prompt = ChatPromptTemplate([
    ("system", "把用户的输入翻译成{language}"),
    ("user", "{text}")
])

prompt = prompt.format_prompt(language="法语", text="你好，世界！")
print(prompt)
print("-"*50)

result = models["qwen3-max"].invoke(prompt)
print(result)