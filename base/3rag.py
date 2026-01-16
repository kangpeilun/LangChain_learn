# -*- coding: utf-8 -*-
#        Data: 2026-01-16 17:17
#     Project: LangChain_learn
#   File Name: 3rag.py
#      Author: KangPeilun
#       Email: 374774222@qq.com 
# Description:

import bs4
from langchain_community.document_loaders import WebBaseLoader, Docx2txtLoader, PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, MarkdownHeaderTextSplitter

# 从网页中加载文档
web_path = "https://www.news.cn/fortune/20251218/93d4a4212fd34985ad05d178a27c7c75/c.html"

loader = WebBaseLoader(
    web_path=[web_path],
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("main-left left", "title")))
)

docs = loader.load()
print(docs)  # Document对象
print('-' * 100)

# 从txt文件中加载文档
loader = TextLoader("./docs/deepseek.txt", encoding="utf-8")
docs = loader.load()
print(docs)
print('-' * 100)

# 从docx文件中加载文档
loader = Docx2txtLoader("./docs/行业.docx")
docs = loader.load()
print(docs)
print('-' * 100)

# 从pdf文件中加载文档
loader = PyMuPDFLoader("./docs/Reflexion.pdf")
docs = loader.load()
print(docs)
print('-' * 100)

# -------------------------------------------------------------------
# 文档切分
# 1. RecursiveCharacterTextSplitter：智能递归分割
# 用的最多的文本切分器
spliter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]   # 从左到右优先级逐渐降低，依次用这些分割符进行对文档进行切分
)
documents = spliter.split_documents(docs)
# for doc in documents:
#     print(doc, end="\n --- \n")
print(len(documents))
print('-' * 100)


# 2. CharacterTextSplitter：文本按分隔符简单分割
spliter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separator=" "   # 只能用一个分割符
)
documents = spliter.split_documents(docs)
# for doc in documents:
#     print(doc, end="\n --- \n")
print(len(documents))
print('-' * 100)


# 3. MarkdownHeaderTextSplitter：按Markdown标题切分
markdown = """
# 第一章
## 第一节
这是第一节内容
## 第二节
这是第二节内容
# 第二章
这是第二章内容
"""
headers = [
    ("#", "Header1"),
    ("##", "Header2")
]  # 定义哪些是一级标题，哪些是二级标题
# 按headers拆分
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(markdown)
print(chunks)
