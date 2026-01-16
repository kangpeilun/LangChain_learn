# -*- coding: utf-8 -*-
#        Data: 2026-01-16 11:40
#     Project: LangChain_learn
#   File Name: 1first_agent.py
#      Author: KangPeilun
#       Email: 374774222@qq.com 
# Description:

from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.tools import tool, ToolRuntime
from langchain.agents.structured_output import ToolStrategy
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from typing import Literal

from model_config import models

# 定义系统提示
SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。
你可以使用两个工具：
- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置
如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。"""


# 定义上下文模式
@dataclass
class Context:
    """自定义运行时上下文模式"""
    user_id: str


# 使用 Pydantic 模型定义复杂输入
class WeatherInput(BaseModel):
    """天气查询的输入定义"""
    location: str = Field(description="城市名称或坐标")
    units: Literal["celsius", "fahrenheit"] = Field(
        default="celsius",
        description="温度单位"
    )  # 通过Field给定一些默认值和描述
    include_forecast: bool = Field(
        default=False,
        description="包括五天天气预报"
    )


# json模型定义复杂输入
weather_schema_json = {
    "type": "object",
    "properties": {
        "location": {"type": "string"},
        "units": {"type": "string"},
        "include_forecast": {"type": "boolean"}
    },
    "required": ["location", "units", "include_forecast"]
}


# 定义工具
@tool("get_weather", description="获取当前天气和可选的天气预报",
      args_schema=WeatherInput)  # 自定义工具名称, description 描述工具用途方便模型理解
def get_weather_for_location(location: str, units: str, include_forecast: bool) -> str:
    # 下面的函数说明有助于模型理解工具的用途，同时也可以定义参数的类型，方便大模型识别
    """获取当前天气和可选的天气预报"""
    temp = 22 if units == "celsius" else 72
    result = f"城市 {location} 的当前温度是: {temp} {units[0].upper()}"
    if include_forecast:
        result += "\n未来5天: Sunny"
    return result


@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户ID获取位置信息"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"


# 定义响应格式
@dataclass
class ResponseFormat:
    """定义响应格式"""
    # 带双关语的回应（始终必需）
    punny_response: str
    # 实际天气信息（可选）
    weather_info: str | None = None


# 根据对话长度动态选择模型
basic_model = models["qwen3-max"]
advanced_model = models["qwen3-max"]


@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """根据会话复杂度选择模型."""
    message_count = len(request.state["messages"])
    # print("message_count", message_count)
    # 根据会话长度，修改模型
    if message_count > 10:
        # 使用高级模式进行更长的对话
        model = advanced_model
    else:
        model = basic_model
    return handler(request.override(model=model))


# 设置记忆
# InMemorySaver可以自动将过往的所有对话加入到对话历史中，如果不使用InMemorySaver，则每次对话都不会记忆之前的内容，每条对话都是新对话
checkpointer = InMemorySaver()
# thread_id 用于标识对话线程，可以帮助记忆管理
config = {"configurable": {"thread_id": "3"}}

# 创建Agent
agent = create_agent(
    model=models["qwen3-max"],
    system_prompt=SYSTEM_PROMPT,
    tools=[get_weather_for_location, get_user_location],
    context_schema=Context,  # context_schema 上下文描述
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer,
    middleware=[dynamic_model_selection],
)

# 运行Agent
messages_list = [
    {"role": "user", "content": "今天天气怎么样？"},
    {"role": "user", "content": "明天呢？"},
    {"role": "user", "content": "后天会下雨吗？"},
    {"role": "user", "content": "这个周末适合户外活动吗？"},
    {"role": "user", "content": "下周一的天气预报如何？"},
]
for messages in messages_list:
    response = agent.invoke(
        {"messages": messages},
        config=config,
        context=Context(user_id="1"),
    )
    print(response['structured_response'])
