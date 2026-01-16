# -*- coding: utf-8 -*-
#        Data: 2026-01-16 11:37
#     Project: LangChain_learn
#   File Name: model_zoo.py
#      Author: KangPeilun
#       Email: 374774222@qq.com 
# Description:

from langchain_openai import ChatOpenAI

api_key = "sk-11f927fe64e64886ae1364e132663e9a"
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/"

model_list = ["qwen3-max", "qwen3-32b"]

qwen3_series_model = {}
for model in model_list:
    qwen3_series_model[model] = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )