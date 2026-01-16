# -*- coding: utf-8 -*-
#        Data: 2026-01-16 11:38
#     Project: LangChain_learn
#   File Name: __init__.py
#      Author: KangPeilun
#       Email: 374774222@qq.com 
# Description:

import os
from importlib import import_module

models = {}
model_type_list = [f[:-3] for f in os.listdir(os.path.dirname(__file__)) if f.endswith('.py') and f != '__init__.py']
for model_type in model_type_list:
    module = import_module(f"model_config.{model_type}")
    value = getattr(module, f"{model_type}_series_model", None)
    models.update(value)

print("系统已有的模型:", models.keys())