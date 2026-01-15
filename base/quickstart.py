# -*- coding: utf-8 -*-
#        Data: 2026-01-15 16:11
#     Project: LangChain_learn
#   File Name: start.py
#      Author: KangPeilun
#       Email: 374774222@qq.com 
# Description:

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool


model = ChatOpenAI(
    model="qwen3-max",
    api_key="sk-11f927fe64e64886ae1364e132663e9a",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1/"
)

@tool
def search(query: str) -> str:
    """A search engine to find information about current events."""
    return "Search results for: " + query

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return "The current weather in " + location + " is sunny."

agent = create_agent(
    model=model,
    tools=[search, get_weather],
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "纽约的天气怎么样？"}]}
)
print(result)